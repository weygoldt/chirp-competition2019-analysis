#!/usr/bin/env bash
# Evaluate one arm of the front-end ablation with the unchanged
# experiments/evaluation/eval_postprocess.py (current chirpdetector main):
# sim_v4 test tiles of the arm's own dataset (box-midpoint matching,
# post-processing, assignment to the scene tracks) and the 335 real labels
# (predict_span over the labelled regions). Output to the report folder, so
# that the released run's own eval_postprocess/ stays untouched.
#
#   experiments/ablation/run_eval.sh A|B
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
root=$CHIRP_DATA/runs/ablation_frontend
case $1 in
  A) run=$CHIRP_DATA/runs/dualhead_r18_sim_v4_r1;          sim=$CHIRP_FAST/datasets/sim_v4 ;;
  B) run=$CHIRP_DATA/runs/dualhead_r18_sim_v4_longonly_r1; sim=$CHIRP_FAST/datasets/sim_v4_longonly ;;
  *) echo "usage: $0 A|B" >&2; exit 2 ;;
esac
mkdir -p "$root/$1" "$root/logs"
cd "$CHIRP_REPO"
unset VIRTUAL_ENV
"$HOME/wrk/analyses/chirp_competition_upgrade/scripts/guarded" \
  --name "ablation-eval-$1" --mem 24G --vram 0.4 -- \
  uv run python experiments/evaluation/eval_postprocess.py "$run" --sim "$sim" \
  --out "$root/$1/eval_postprocess" > "$root/logs/eval_postprocess_$1.log" 2>&1
