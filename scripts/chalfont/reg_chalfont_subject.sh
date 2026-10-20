#!/usr/bin/env bash
# Register one staged Chalfont subject to the symmetric template and bring its
# ASL maps into that space.
#
#   a) asl_ref (PD/M0, bet-masked) -> T1_brain      flirt 6 dof normmi   -> <ID>_asl2t1.mat
#   b) T1_brain -> T1_sym                           flirt 6 dof (full search) then 12 dof -> aff
#   c) fnirt T1_brain -> T1_sym (parameters identical to symreg/reg_subject.sh) -> <ID>_warp.nii.gz
#   d) applywarp --premat=asl2t1 --warp=warp on tcflow (PRIMARY -> <ID>_perf_sym.nii.gz)
#      and flow (-> <ID>_perfflow_sym.nii.gz); spline, float32, nothing clamped
#   e) applywarp nn gif_parc -> <ID>_gifparc_sym.nii.gz
#   f) warped T1 on the exact-symmetric 2 mm QC grid -> <ID>_t1_sym_2mm.nii.gz
#
# The ASL->sym composition is done inside applywarp (--premat + --warp), which is
# the same transform convertwarp would produce without writing the 170 MB
# full-resolution combined warp field.
#
# Usage: reg_chalfont_subject.sh <ID>
set -euo pipefail

ID="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../.." && pwd)"
ST="$BASE/external_chalfont/staged/$ID"
OUT="$BASE/external_chalfont/sym_perf"
SYM="$BASE/symreg/templates/T1_sym.nii.gz"
TMP="$OUT/${ID}_tmp"
mkdir -p "$OUT"

export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=1

for f in T1_brain.nii.gz asl_ref.nii.gz flow.nii.gz tcflow.nii.gz; do
  [ -f "$ST/$f" ] || { echo "ERROR $ID: missing staged $f" >&2; exit 1; }
done

t0=$(date +%s)
log() { echo "[$ID $(( $(date +%s) - t0 ))s] $*"; }

# a) ASL reference -> T1 (rigid). Brain-mask the PD/M0 image with a quick bet;
#    fall back to the unmasked image if bet removes (nearly) everything.
bet "$ST/asl_ref.nii.gz" "${TMP}_aslref_brain" -f 0.3 -m >/dev/null 2>&1 || true
MOV="${TMP}_aslref_brain.nii.gz"
if [ ! -f "$MOV" ] || [ "$(fslstats "$MOV" -V | awk '{print $1}')" -lt 20000 ]; then
  log "bet on asl_ref failed/too small; using unmasked asl_ref"
  MOV="$ST/asl_ref.nii.gz"
fi
flirt -in "$MOV" -ref "$ST/T1_brain.nii.gz" -dof 6 -cost normmi -usesqform \
      -searchrx -30 30 -searchry -30 30 -searchrz -30 30 \
      -omat "$OUT/${ID}_asl2t1.mat" -out "${TMP}_aslref_in_t1.nii.gz"
log "asl2t1 done"

# b) T1_brain -> T1_sym: 6-dof full search, then 12 dof refinement.
flirt -in "$ST/T1_brain.nii.gz" -ref "$SYM" -dof 6 -omat "${TMP}_init6.mat"
flirt -in "$ST/T1_brain.nii.gz" -ref "$SYM" -dof 12 -init "${TMP}_init6.mat" -nosearch \
      -omat "${TMP}_aff.mat"
log "affine done"

# c) Nonlinear warp (same parameters as symreg/reg_subject.sh).
fnirt --in="$ST/T1_brain.nii.gz" --ref="$SYM" --aff="${TMP}_aff.mat" \
      --cout="$OUT/${ID}_warp.nii.gz" --iout="${TMP}_t1warped.nii.gz" \
      --subsamp=8,4,2 --miter=5,5,3 --infwhm=8,4,2 --reffwhm=6,2,0 \
      --estint=1,1,1 --applyinmask=1,1,1 --applyrefmask=1,1,1 \
      --warpres=10,10,10 --regmod=bending_energy --intmod=global_non_linear_with_bias >/dev/null 2>&1
log "fnirt done"

# d) ASL maps -> symmetric space (ASL->T1 premat composed with the T1->sym warp).
applywarp --in="$ST/tcflow.nii.gz" --ref="$SYM" --premat="$OUT/${ID}_asl2t1.mat" \
          --warp="$OUT/${ID}_warp.nii.gz" --out="$OUT/${ID}_perf_sym.nii.gz" \
          --interp=spline --datatype=float
applywarp --in="$ST/flow.nii.gz" --ref="$SYM" --premat="$OUT/${ID}_asl2t1.mat" \
          --warp="$OUT/${ID}_warp.nii.gz" --out="$OUT/${ID}_perfflow_sym.nii.gz" \
          --interp=spline --datatype=float
log "perfusion warped"

# e) GIF parcellation (nearest neighbour), when present.
if [ -f "$ST/gif_parc.nii.gz" ]; then
  applywarp --in="$ST/gif_parc.nii.gz" --ref="$SYM" --warp="$OUT/${ID}_warp.nii.gz" \
            --out="$OUT/${ID}_gifparc_sym.nii.gz" --interp=nn --datatype=short
fi

# f) Warped T1 on the exact flip-symmetric 2 mm grid (QC only).
python "$HERE/chalfont_io.py" resample2mm "${TMP}_t1warped.nii.gz" "$OUT/${ID}_t1_sym_2mm.nii.gz" 1
log "t1 2mm written"

rm -f "${TMP}_aslref_brain.nii.gz" "${TMP}_aslref_brain_mask.nii.gz" "${TMP}_aslref_in_t1.nii.gz" \
      "${TMP}_init6.mat" "${TMP}_aff.mat" "${TMP}_t1warped.nii.gz"
log "total $(( $(date +%s) - t0 ))s"
echo "DONE $ID"
