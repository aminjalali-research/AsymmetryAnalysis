#!/usr/bin/env bash
# FSL FIRST hippocampus + amygdala segmentation for the Chalfont cohort (patients and controls) on the
# staged brain-extracted T1, in parallel, producing <out>/<ID>_all_fast_firstseg.nii.gz per subject.
# Structural prior for lateralization (hippocampal volume AI; Ndagijimana 2026, AID-HS 2025).
# Usage: bash scripts/audit/first_volume_ai.sh [jobs]     (run through scripts/run_capped.sh)
set -u
BASE="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$BASE/ez_analysis_mdt/audit/first_chalfont"; mkdir -p "$OUT"
JOBS="${1:-8}"
export FSLOUTPUTTYPE=NIFTI_GZ
one() {
  id="$1"; t1="$BASE/external_chalfont/staged/$id/T1_brain.nii.gz"
  [ -f "$t1" ] || { echo "$id: no T1"; return; }
  [ -f "$OUT/${id}_all_fast_firstseg.nii.gz" ] && { echo "$id: done"; return; }
  run_first_all -i "$t1" -o "$OUT/$id" -s L_Hipp,R_Hipp,L_Amyg,R_Amyg -b -d > "$OUT/$id.log" 2>&1
  [ -f "$OUT/${id}_all_fast_firstseg.nii.gz" ] && echo "$id: ok" || echo "$id: FAILED"
}
export -f one; export BASE OUT FSLOUTPUTTYPE
ls "$BASE/external_chalfont/staged" | xargs -P "$JOBS" -I{} bash -c 'one {}'
echo DONE
