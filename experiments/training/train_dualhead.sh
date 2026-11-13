#!/usr/bin/env bash
# Train a dual-head detector under the GPU guard, one log per attempt.
#   train_dualhead.sh CONFIG DATASET RUN_DIR [VRAM]
# (dualhead_r18_sim_v4_r1 was trained with the root environment at
# chirpdetector 0a6d789, the dual-head merge.)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
config=$1 dataset=$2 run=$3 vram=${4:-0.35}
logs=$CHIRP_DATA/runs/dualhead_logs
mkdir -p "$logs"
log="$logs/$(basename "$run")_$(date +%Y%m%d-%H%M%S).log"
echo "log: $log"
cd "$CHIRP_REPO"
unset VIRTUAL_ENV
exec "$CHIRP_REPO/scripts/guarded" \
  --name "train-$(basename "$run")" --mem 24G --vram "$vram" -- \
  uv run chirpdetector train detector "$config" --dataset "$dataset" --run-dir "$run" \
  > "$log" 2>&1
