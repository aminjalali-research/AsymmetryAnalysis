#!/usr/bin/env bash
# Register one subject's T1 brain to the symmetric template, then apply the
# resulting warp to that subject's perfusion image -> perfusion in symmetric
# space (flip0-symmetric anatomy => valid mirror-AI).
#
# Usage: reg_subject.sh <T1_brain> <perfusion> <out_prefix> [aparc+aseg]
set -euo pipefail

T1="$1"; PERF="$2"; OUT="$3"; APARC="${4:-}"
SYM="$(dirname "$0")/templates/T1_sym.nii.gz"
TMP="${OUT}_tmp"
mkdir -p "$(dirname "$OUT")"

export FSLOUTPUTTYPE=NIFTI_GZ

# 1) Affine init (12 dof). Both already in MNI grid so this is a small refine.
flirt -in "$T1" -ref "$SYM" -omat "${TMP}_aff.mat" -dof 12 \
      -searchrx -10 10 -searchry -10 10 -searchrz -10 10 -nosearch 2>/dev/null

# 2) Nonlinear warp T1 -> symmetric template. Subsampling tuned for 0.8mm ref:
#    subsamp 8,4,2 => ~6.4, 3.2, 1.6 mm sampling. Smooth large-scale asymmetry
#    correction; we do not need sub-mm warp precision.
fnirt --in="$T1" --ref="$SYM" --aff="${TMP}_aff.mat" --cout="${OUT}_warp.nii.gz" \
      --subsamp=8,4,2 --miter=5,5,3 --infwhm=8,4,2 --reffwhm=6,2,0 \
      --estint=1,1,1 --applyinmask=1,1,1 --applyrefmask=1,1,1 \
      --warpres=10,10,10 --regmod=bending_energy --intmod=global_non_linear_with_bias 2>/dev/null

# 3) Apply warp to perfusion (spline interp, output on template grid).
applywarp --in="$PERF" --ref="$SYM" --warp="${OUT}_warp.nii.gz" \
          --out="${OUT}_perf_sym.nii.gz" --interp=spline 2>/dev/null

# 4) Warp the parcellation too (nearest-neighbour) so GM masking + region
#    labelling happen in the same symmetric space as the zAI map.
if [ -n "$APARC" ] && [ -f "$APARC" ]; then
  applywarp --in="$APARC" --ref="$SYM" --warp="${OUT}_warp.nii.gz" \
            --out="${OUT}_aparc_sym.nii.gz" --interp=nn 2>/dev/null
fi

rm -f "${TMP}_aff.mat" "${TMP}_t1warped.nii.gz"
echo "DONE $OUT"
