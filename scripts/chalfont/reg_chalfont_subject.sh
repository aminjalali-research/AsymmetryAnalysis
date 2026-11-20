#!/usr/bin/env bash
# Register one staged Chalfont subject to the symmetric template and bring its
# ASL maps into that space.
#
#   a) asl_ref (PD/M0) -> T1_brain, brain to brain:  flirt 6 dof normmi, moving image weighted by the T1 brain
#      mask carried into ASL space with the scanner headers (-> <ID>_asl2t1_init.mat), refined by boundary-based
#      registration on the GIF white matter (flirt -cost bbr, FSL's ASL/EPI -> T1 method) -> <ID>_asl2t1.mat.
#      Until 2026-11-18 the moving image was "bet -f 0.3" of the PD, which keeps the whole head (bright scalp
#      fat), so the fit was head-to-brain normmi: adequate in most subjects, 8.5 deg off in D042.
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
# Usage: reg_chalfont_subject.sh <ID> [step1]
#   step1: redo only a) and d) (ASL -> T1 and the resampling of the ASL maps), keeping the T1 -> T1_sym warp.
set -euo pipefail

ID="$1"
MODE="${2:-full}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../.." && pwd)"
ST="$BASE/external_chalfont/staged/$ID"
OUT="${REG_OUT:-$BASE/external_chalfont/sym_perf}"     # REG_OUT: alternative output folder (tests, dry runs)
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

# a) ASL reference -> T1 (rigid). bet cannot strip the PD/M0 image (its scalp fat is brighter than brain, so it
#    keeps the whole head), so the +-30 deg normmi search runs on the whole head, as it effectively always did,
#    and the result is only the initialisation. The brain-to-brain fit is the refinement: boundary-based
#    registration on the GIF white matter (labels 81-94, lobar white matter; bbrtype global_abs because the
#    PD/T1 contrast is inverted), FSL's ASL/EPI -> T1 method. Without a GIF parcellation the normmi fit is kept.
#    The refinement is accepted only if it stays within 15 deg and 8 mm (at the ASL volume's centre) of the normmi
#    fit AND its CBF-cortex correlation in the T1 is not lower than the normmi fit's (chalfont_io.py step1pick);
#    otherwise the normmi fit is kept (logged). Audit 2026-11-18: genuine refinements moved <= 9.7 deg / 4.8 mm;
#    the one divergence (N052) moved 14 mm and halved the correlation. (Weighting the normmi search by the
#    projected T1 brain mask was tried and is unstable: D042 converged 60 deg off.)
flirt -in "$ST/asl_ref.nii.gz" -ref "$ST/T1_brain.nii.gz" -dof 6 -cost normmi -usesqform \
      -searchrx -30 30 -searchry -30 30 -searchrz -30 30 \
      -omat "$OUT/${ID}_asl2t1_init.mat" -out "${TMP}_aslref_in_t1.nii.gz"
log "asl2t1 normmi (whole head, initialisation) done"
if [ -f "$ST/gif_parc.nii.gz" ]; then
  fslmaths "$ST/gif_parc.nii.gz" -thr 81 -uthr 94 -bin "${TMP}_wm.nii.gz"
  flirt -in "$ST/asl_ref.nii.gz" -ref "$ST/T1_brain.nii.gz" -dof 6 -cost bbr -wmseg "${TMP}_wm.nii.gz" \
        -init "$OUT/${ID}_asl2t1_init.mat" -schedule "$FSLDIR/etc/flirtsch/bbr.sch" -bbrtype global_abs \
        -omat "${TMP}_bbr.mat" -out "${TMP}_aslref_in_t1_bbr.nii.gz"
  if python "$HERE/chalfont_io.py" step1pick "$OUT/${ID}_asl2t1_init.mat" "${TMP}_bbr.mat" "$ST" 15 8; then
    cp "${TMP}_bbr.mat" "$OUT/${ID}_asl2t1.mat"
    log "asl2t1 BBR refinement done"
  else
    cp "$OUT/${ID}_asl2t1_init.mat" "$OUT/${ID}_asl2t1.mat"
    log "asl2t1 BBR rejected (moved > 15 deg / 8 mm, or a lower CBF-cortex correlation); normmi fit kept"
  fi
else
  cp "$OUT/${ID}_asl2t1_init.mat" "$OUT/${ID}_asl2t1.mat"
  log "asl2t1: no GIF parcellation, normmi fit kept"
fi
#    No further polish: a coordinate descent on the CBF-cortex correlation (chalfont_io.py step1polish) was tried on
#    2026-11-19 and rejected because it made independent checks worse (CBF outside the brain worse in 42/80, in the
#    ventricles worse in 61/80; external_chalfont/qc/asl2t1_polish_independent_check.csv). BBR passes the same checks.

if [ "$MODE" = "step1" ]; then
  [ -f "$OUT/${ID}_warp.nii.gz" ] || { echo "ERROR $ID: step1 mode needs the existing warp" >&2; exit 1; }
  log "step1 mode: keeping the existing T1 -> T1_sym warp"
else
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
fi

# d) ASL maps -> symmetric space (ASL->T1 premat composed with the T1->sym warp).
applywarp --in="$ST/tcflow.nii.gz" --ref="$SYM" --premat="$OUT/${ID}_asl2t1.mat" \
          --warp="$OUT/${ID}_warp.nii.gz" --out="$OUT/${ID}_perf_sym.nii.gz" \
          --interp=spline --datatype=float
applywarp --in="$ST/flow.nii.gz" --ref="$SYM" --premat="$OUT/${ID}_asl2t1.mat" \
          --warp="$OUT/${ID}_warp.nii.gz" --out="$OUT/${ID}_perfflow_sym.nii.gz" \
          --interp=spline --datatype=float
log "perfusion warped"

if [ "$MODE" != "step1" ]; then
# e) GIF parcellation (nearest neighbour), when present.
if [ -f "$ST/gif_parc.nii.gz" ]; then
  applywarp --in="$ST/gif_parc.nii.gz" --ref="$SYM" --warp="$OUT/${ID}_warp.nii.gz" \
            --out="$OUT/${ID}_gifparc_sym.nii.gz" --interp=nn --datatype=short
fi

# f) Warped T1 on the exact flip-symmetric 2 mm grid (QC only).
python "$HERE/chalfont_io.py" resample2mm "${TMP}_t1warped.nii.gz" "$OUT/${ID}_t1_sym_2mm.nii.gz" 1
log "t1 2mm written"
fi

rm -f "${TMP}_wm.nii.gz" "${TMP}_bbr.mat" \
      "${TMP}_aslref_in_t1.nii.gz" "${TMP}_aslref_in_t1_bbr.nii.gz" \
      "${TMP}_init6.mat" "${TMP}_aff.mat" "${TMP}_t1warped.nii.gz"
log "total $(( $(date +%s) - t0 ))s"
echo "DONE $ID"
