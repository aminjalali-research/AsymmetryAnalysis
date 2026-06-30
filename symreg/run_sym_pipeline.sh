#!/usr/bin/env bash
# Consolidate Pipeline B in symmetric-template space (2026-06-22 zAI fix).
# Prereq: all 197 subjects registered (symreg/sym_perf/{ID}_{perf,aparc}_sym.nii.gz).
# Archives the original-space results_zscore outputs, then rebuilds everything
# from the symmetric-space perfusion/parcellation.
set -euo pipefail
cd "$(dirname "$0")/.."
export FSLOUTPUTTYPE=NIFTI_GZ
LOG=symreg/logs/sym_pipeline.log
exec > >(tee "$LOG") 2>&1

BANDS=(FM_20_39 FM_40_59 FM_60_79)
ARCH="_archived_origspace_zai_20260622"

echo "=== [0/6] sanity: count registered subjects ==="
np=$(ls symreg/sym_perf/*_perf_sym.nii.gz 2>/dev/null | wc -l)
na=$(ls symreg/sym_perf/*_aparc_sym.nii.gz 2>/dev/null | wc -l)
echo "perf_sym=$np aparc_sym=$na (expect 197 each)"
[ "$np" -ge 197 ] && [ "$na" -ge 197 ] || { echo "ABORT: registration incomplete"; exit 1; }

echo "=== [1/6] archive original-space results_zscore ==="
mkdir -p "$ARCH"
for d in groups asymmetry patients clinical summary; do
  if [ -e "results_zscore/$d" ] && [ ! -e "$ARCH/$d" ]; then
    mv "results_zscore/$d" "$ARCH/$d" && echo "  archived results_zscore/$d"
  fi
done

echo "=== [2/6] 01_build_control_normative.py --sym --rebuild-group (all bands) ==="
python 01_build_control_normative.py --sym --rebuild-group

echo "=== [3/6] 02_compute_zai.py --sym --rebuild (per band) ==="
for b in "${BANDS[@]}"; do
  echo "--- band $b ---"
  python 02_compute_zai.py --sym --rebuild --group "$b"
done

echo "=== [4/6] 03_clinical_maps.py --sym (all patients) ==="
python 03_clinical_maps.py --sym

echo "=== [5/6] 06_clinical_interpretation.py --all ==="
python 06_clinical_interpretation.py --all

echo "=== [6/6] scripts/rerun_thresholding_methods.py --sym ==="
python scripts/rerun_thresholding_methods.py --sym || echo "  (thresholding step returned nonzero; inspect)"

echo "=== SYM PIPELINE COMPLETE ==="
