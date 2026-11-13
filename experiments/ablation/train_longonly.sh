#!/usr/bin/env bash
# Arm B of the front-end ablation: train the dual-head on sim_v4_longonly
# with the released model's config (same seed, schedule, early stopping).
# As experiments/training/train_dualhead.sh, but named ablation-train and at
# vram 0.4 (a cap only; the released run used 0.35 and peaked at 3.1 GiB).
#   experiments/ablation/train_longonly.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
config=$CHIRP_REPO/experiments/ablation/dualhead_r18_sim_v4_longonly.toml
dataset=$CHIRP_FAST/datasets/sim_v4_longonly
run=$CHIRP_DATA/runs/dualhead_r18_sim_v4_longonly_r1
logs=$CHIRP_DATA/runs/dualhead_logs
mkdir -p "$logs"
log="$logs/$(basename "$run")_$(date +%Y%m%d-%H%M%S).log"
echo "log: $log"
cd "$CHIRP_REPO"
unset VIRTUAL_ENV
exec "$HOME/wrk/analyses/chirp_competition_upgrade/scripts/guarded" \
  --name ablation-train --mem 24G --vram 0.4 -- \
  uv run chirpdetector train detector "$config" --dataset "$dataset" --run-dir "$run" \
  > "$log" 2>&1
