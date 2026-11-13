#!/usr/bin/env bash
# Run one script of the paper analysis: paper/run.sh SCRIPT.py [ARGS...]
#
# - paper's own environment (paper/pyproject.toml, Python 3.14);
# - working directory = the output root ($PAPER_OUT), so any relative path in
#   the ported code lands there, never in a repository;
# - MPLBACKEND=Agg: the old scripts' plt.show() calls do not block;
# - log to $PAPER_OUT/logs/<script>.log.
# Heavy steps are wrapped by run_all.sh in scripts/guarded (RAM cap).
set -euo pipefail
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script="$1"; shift
: "${CHIRP_DATA:=/mnt/data1/chirp_v2}"
: "${PAPER_DETECTIONS:=rtdetr}"
: "${PAPER_OUT:=$CHIRP_DATA/paper_rerun/$PAPER_DETECTIONS}"
export CHIRP_DATA PAPER_DETECTIONS PAPER_OUT
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$PAPER_OUT/logs"
log="$PAPER_OUT/logs/$(basename "$script" .py).log"
unset VIRTUAL_ENV
cd "$PAPER_OUT"
echo "[$(date -Is)] $script $* (detections $PAPER_DETECTIONS, out $PAPER_OUT, paper $(git -C "$PAPER" rev-parse --short HEAD))" | tee "$log"
uv run --project "$PAPER" --python 3.14 python "$PAPER/scripts/$script" "$@" < /dev/null 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
echo "[$(date -Is)] exit $status" | tee -a "$log"
exit "$status"
