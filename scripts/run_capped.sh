#!/usr/bin/env bash
# Run a long pipeline in its own memory-capped systemd user scope, detached from
# the terminal, with BLAS/OpenMP threads pinned.
#
# Why (2026-10-20): two pipelines ran concurrently with 10-12 parallel per-subject
# jobs on the 227x272x227 grid; the kernel OOM killer and systemd-oomd then killed
# desktop processes (GNOME session, IDE scope), which logged the user out and
# took the pipelines (children of the IDE scope) down with them.
#
# With this launcher:
#   - MemoryMax caps the pipeline cgroup: hitting it kills one python job inside
#     the pipeline (xargs reports it FAILED), never the desktop.
#   - MemorySwapMax=0 stops the pipeline from swapping, so it cannot create the
#     memory-pressure condition that makes systemd-oomd kill user scopes.
#   - setsid + nohup: the pipeline survives the terminal / IDE / Claude session.
#   - nice / ionice keep the desktop responsive.
#   - *_NUM_THREADS=1: N parallel numpy jobs must not each spawn 24 BLAS threads
#     (load average hit 21 on 24 cores during the failed runs).
#
# Usage: scripts/run_capped.sh <name> <MemoryMax, e.g. 30G> <logfile> -- <command...>
#   e.g. scripts/run_capped.sh chalfont-tc 30G external_chalfont/logs/capped.out -- \
#            env CHALFONT_JOBS=6 bash scripts/chalfont/run_chalfont_pipeline.sh tc all
# Follow with: tail -f <logfile>; status: systemctl --user status <name>-*.scope
set -euo pipefail
[ $# -ge 5 ] || { sed -n 2,25p "$0"; exit 2; }
NAME="$1"; MAX="$2"; LOGFILE="$3"; shift 3
[ "${1:-}" = "--" ] && shift
mkdir -p "$(dirname "$LOGFILE")"
UNIT="${NAME}-$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export FSLOUTPUTTYPE="${FSLOUTPUTTYPE:-NIFTI_GZ}"
systemd-run --user --scope --quiet --unit="$UNIT" \
  -p MemoryMax="$MAX" -p MemorySwapMax=0 \
  setsid nohup nice -n 10 ionice -c 2 -n 7 "$@" > "$LOGFILE" 2>&1 < /dev/null &
disown
echo "launched unit=${UNIT}.scope MemoryMax=${MAX} log=${LOGFILE}"
echo "  stop:   systemctl --user stop ${UNIT}.scope"
echo "  memory: cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/${UNIT}.scope/memory.current"
