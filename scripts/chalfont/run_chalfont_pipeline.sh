#!/usr/bin/env bash
# Chalfont external validation: run Pipeline B unchanged in logic on the
# external cohort (mirrors symreg/run_sym_pipeline.sh), with the per-subject
# stages fanned out with xargs -P.
#
# Prereqs:
#   external_chalfont/sym_perf/<ID>_perf_sym.nii.gz (+ _perfflow_sym for the flow pass)
#   external_chalfont/metadata/subjects.csv, groups.json, qc_exclusions.json
#   external_chalfont/atlas/consensus_parcellation_sym.nii.gz
#       (python scripts/chalfont/build_template_atlas.py)
#
# Usage: bash scripts/chalfont/run_chalfont_pipeline.sh [tc|flow|both] [all|specific]
#   variant  tc   = transit-time-corrected CBF (primary)     -> results_zscore/
#            flow = plain CBF                                  -> results_zscore_flow/
#   policy   all      = pooled CH_ALL controls for everyone (primary, default)
#            specific = age-band sensitivity analysis           -> results_zscore[_flow]_bands/
#   env      CHALFONT_JOBS=10                    parallel per-subject jobs (default 10)
#            CHALFONT_BANDS="CH_20_39 CH_40_59"  restrict the bands that are run
#            CHALFONT_ROOT=<dir>                  alternative root (tests)
# Stages: serial group builds (01, 02) -> per patient 01->02->03 in parallel ->
# 06 --all -> LOO calibration -> pseudo-controls (prepare, then parallel) ->
# summary collector -> stratified analysis + evaluation. No thresholding step.
set -euo pipefail
cd "$(dirname "$0")/../.."
export FSLOUTPUTTYPE=NIFTI_GZ
WHICH="${1:-both}"
POLICY="${2:-all}"
ROOT="${CHALFONT_ROOT:-external_chalfont}"
JOBS="${CHALFONT_JOBS:-10}"
mkdir -p "$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/logs/chalfont_pipeline_${WHICH}_${POLICY}_${STAMP}.log"
exec > >(tee "$LOG") 2>&1

run_variant() {
  local V="$1"                       # tc | flow
  local DS=(--dataset chalfont --perf-variant "$V" --band-policy "$POLICY")
  local SUBLOG="$ROOT/logs/subjects_${V}_${POLICY}_${STAMP}"
  mkdir -p "$SUBLOG"
  echo "################ perf-variant=$V band-policy=$POLICY jobs=$JOBS ################"
  echo "=== [0] configuration ==="
  python dataset_config.py chalfont --perf-variant "$V" --band-policy "$POLICY"
  mapfile -t BANDS < <(python dataset_config.py chalfont --perf-variant "$V" --band-policy "$POLICY" --list-run-bands)
  if [ -n "${CHALFONT_BANDS:-}" ]; then
    local SEL=()
    for b in "${BANDS[@]}"; do for w in $CHALFONT_BANDS; do [ "$b" = "$w" ] && SEL+=("$b"); done; done
    BANDS=("${SEL[@]}")
  fi
  [ "${#BANDS[@]}" -gt 0 ] || { echo "ABORT: no control bands with perfusion (groups.json / sym_perf / CHALFONT_BANDS)"; exit 1; }
  [ -f "$ROOT/atlas/consensus_parcellation_sym.nii.gz" ] || { echo "ABORT: template atlas missing"; exit 1; }
  echo "bands: ${BANDS[*]}"

  echo "=== [1] group statistics (serial): 01 --group-only, 02 --group-only per band ==="
  for b in "${BANDS[@]}"; do
    echo "--- band $b ---"
    python 01_build_control_normative.py "${DS[@]}" --rebuild-group --group-only --group "$b"
    python 02_compute_zai.py "${DS[@]}" --rebuild --group-only --group "$b"
  done

  echo "=== [2] per patient 01 -> 02 -> 03 (parallel, $JOBS jobs; logs in $SUBLOG) ==="
  for b in "${BANDS[@]}"; do
    echo "--- band $b ---"
    python dataset_config.py chalfont --perf-variant "$V" --band-policy "$POLICY" --patients-for-group "$b" \
    | xargs -P "$JOBS" -I{} bash -c '
        pid={}; b='"$b"'; log='"$SUBLOG"'/${pid}.log
        { python 01_build_control_normative.py '"${DS[*]}"' --patient "$pid" --group "$b" \
       && python 02_compute_zai.py '"${DS[*]}"' --patient "$pid" --group "$b" \
       && python 03_clinical_maps.py '"${DS[*]}"' --patient "$pid"; } > "$log" 2>&1 \
       && echo "  done $pid" || echo "  FAILED $pid (see $log)"'
  done

  echo "=== [3] 06_clinical_interpretation.py --all ==="
  python 06_clinical_interpretation.py "${DS[@]}" --all

  echo "=== [4] calibration: leave-one-out controls + patients ==="
  python scripts/chalfont/calibration_loo.py "${DS[@]}" --bands "${BANDS[@]}"
  python symreg/check_full_calibration.py "${DS[@]}"

  echo "=== [5] controls as pseudo-patients (leave-one-out): prepare, then parallel ==="
  for b in "${BANDS[@]}"; do
    echo "--- band $b ---"
    python 02_compute_zai.py "${DS[@]}" --group "$b" --pseudo-controls --prepare-only
    python dataset_config.py chalfont --perf-variant "$V" --band-policy "$POLICY" --pseudo-controls-for-group "$b" \
    | xargs -P "$JOBS" -I{} bash -c '
        cid={}; b='"$b"'; log='"$SUBLOG"'/${cid}.log
        python 02_compute_zai.py '"${DS[*]}"' --group "$b" --pseudo-controls --patient "$cid" > "$log" 2>&1 \
        && echo "  done $cid" || echo "  FAILED $cid (see $log)"'
  done

  echo "=== [6] collect summaries ==="
  python scripts/chalfont/collect_summaries.py "${DS[@]}"

  echo "=== [7] lateralization evaluation ==="
  python scripts/clinical_stratified_analysis.py "${DS[@]}"
  python scripts/chalfont/evaluate_lateralization.py "${DS[@]}"
  echo "################ perf-variant=$V band-policy=$POLICY DONE ################"
}

case "$WHICH" in
  tc)   run_variant tc ;;
  flow) run_variant flow ;;
  both) run_variant tc; run_variant flow ;;
  *) echo "usage: $0 [tc|flow|both] [all|specific]"; exit 2 ;;
esac
echo "=== CHALFONT PIPELINE COMPLETE (log: $LOG) ==="
