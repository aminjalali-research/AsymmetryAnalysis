#!/usr/bin/env bash
# Stage all Chalfont subjects, then register every complete subject to the
# symmetric template (xargs -P), skipping subjects whose outputs already exist,
# then build the QC table/images.
#
# Usage: run_registration.sh [-P 10] [--skip-stage] [--redo-step1]
#   --redo-step1  re-estimate ASL -> T1 (step a) and re-resample the ASL maps (step d) for every registered
#                 subject, keeping the T1 -> T1_sym warps (reg_chalfont_subject.sh <ID> step1); then the QC.
#                 Downstream outputs (results_zscore, v2 features, audits) must be re-run afterwards.
# Logs: external_chalfont/logs/stage.log, reg_<ID>.log, batch.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../.." && pwd)"
EXT="$BASE/external_chalfont"
NPROC=10
SKIP_STAGE=0
MODE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -P) NPROC="$2"; shift 2;;
    --skip-stage) SKIP_STAGE=1; shift;;
    --redo-step1) MODE="step1"; SKIP_STAGE=1; shift;;
    *) echo "unknown arg $1" >&2; exit 1;;
  esac
done
mkdir -p "$EXT/logs" "$EXT/sym_perf" "$EXT/qc"
cd "$BASE"

echo "[$(date)] batch start (P=$NPROC)"
if [ "$SKIP_STAGE" -eq 0 ]; then
  python "$HERE/stage_chalfont.py" -j 6 > "$EXT/logs/stage.log" 2>&1
  echo "[$(date)] staging done: $(tail -n 1 "$EXT/logs/stage.log")"
fi

# Subjects staged successfully and not yet registered.
TODO="$EXT/logs/todo_ids.txt"
python - "$EXT" "$MODE" > "$TODO" <<'EOF'
import csv, os, sys
ext, mode = sys.argv[1], sys.argv[2]
for r in csv.DictReader(open(os.path.join(ext, "metadata", "staging_manifest.csv"))):
    if r["status"] != "staged":
        continue
    sid = r["id"]
    outs = [os.path.join(ext, "sym_perf", f"{sid}_{s}") for s in ("perf_sym.nii.gz", "perfflow_sym.nii.gz", "warp.nii.gz", "t1_sym_2mm.nii.gz")]
    done = all(os.path.exists(o) for o in outs)
    if (mode == "step1" and done) or (mode != "step1" and not done):
        print(sid)
EOF
echo "[$(date)] $(wc -l < "$TODO") subjects to register${MODE:+ ($MODE only)}"

export OMP_NUM_THREADS=1
xargs -P "$NPROC" -I{} bash -c \
  'bash "'"$HERE"'/reg_chalfont_subject.sh" {} '"$MODE"' > "'"$EXT"'/logs/reg_{}.log" 2>&1 || echo "FAILED {}"' < "$TODO"
echo "[$(date)] registration finished: $(grep -l '^DONE' "$EXT"/logs/reg_*.log 2>/dev/null | wc -l) DONE logs"

python "$HERE/make_qc.py" > "$EXT/logs/qc.log" 2>&1
echo "[$(date)] QC done: $(tail -n 1 "$EXT/logs/qc.log")"
