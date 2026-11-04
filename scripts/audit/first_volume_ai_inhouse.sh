#!/usr/bin/env bash
# FSL FIRST hippocampus + amygdala segmentation for the in-house cohort (17 patients, 180 HCP controls) on
# the native whole-head T1w_acpc_dc_restore (0.8 mm), in parallel -> ez_analysis_mdt/audit/first_inhouse/<ID>_all_fast_firstseg.nii.gz
# Usage: bash scripts/audit/first_volume_ai_inhouse.sh [jobs]     (run through scripts/run_capped.sh; after the Chalfont batch)
set -u
BASE="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$BASE/ez_analysis_mdt/audit/first_inhouse"; mkdir -p "$OUT"
JOBS="${1:-8}"
export FSLOUTPUTTYPE=NIFTI_GZ
one() {
  id="$1"; t1="$2"
  [ -f "$t1" ] || { echo "$id: no T1"; return; }
  [ -f "$OUT/${id}_all_fast_firstseg.nii.gz" ] && { echo "$id: done"; return; }
  run_first_all -i "$t1" -o "$OUT/$id" -s L_Hipp,R_Hipp,L_Amyg,R_Amyg -d > "$OUT/$id.log" 2>&1
  [ -f "$OUT/${id}_all_fast_firstseg.nii.gz" ] && echo "$id: ok" || echo "$id: FAILED"
}
export -f one; export BASE OUT FSLOUTPUTTYPE
{
  for p in "$BASE"/Dataset/P0*; do echo "$(basename "$p") $p/T1w_acpc_dc_restore.nii.gz"; done
  for b in FM_20_39 FM_40_59 FM_60_79; do for c in "$BASE"/DatasetControls/$b/*/; do id=$(basename "$c"); echo "$id ${c}T1w/T1w_acpc_dc_restore.nii.gz"; done; done
} | xargs -P "$JOBS" -L 1 bash -c 'one "$0" "$1"'
echo DONE
