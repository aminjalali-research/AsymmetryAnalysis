#!/usr/bin/env bash
# Drive the full post-SD-fix re-run for the 5 F_20_39 patients.
#   Assumes 02_compute_zai.py --group F_20_39 --rebuild has already
#   regenerated the group stats AND the patient zAI maps.
set -euo pipefail
cd "$(dirname "$0")/.."

PATIENTS=(P013 P014 P015 P020 P026)

echo "=== Step 1: clinical maps (03_clinical_maps.py) ==="
for p in "${PATIENTS[@]}"; do
    echo "  -> $p"
    python3 03_clinical_maps.py --patient "$p"
done

echo "=== Step 2: per-index discrimination ==="
python3 scripts/per_index_discrimination.py

echo "=== Step 3: localization metrics ==="
python3 scripts/localization_metrics.py

echo "=== Step 4: thresholding methods rerun ==="
python3 scripts/rerun_thresholding_methods.py

echo "=== ALL DONE ==="
