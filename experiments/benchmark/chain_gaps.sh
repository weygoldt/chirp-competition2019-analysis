#!/usr/bin/env bash
# Both v2 models on the gap scenes (scenes_gap.txt), after chain_v2_models.sh
# (its output was logged to $CHIRP_DATA/logs/bench_chain_v2.log).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$CHIRP_REPO"
S=$CHIRP_REPO/experiments/benchmark/run_models_v2.py
L=$CHIRP_REPO/experiments/benchmark
scenes() { sed "s|^|$CHIRP_DATA/|" "$L/scenes_$1.txt"; }
until grep -q "CHAIN DONE" "$CHIRP_DATA/logs/bench_chain_v2.log"; do sleep 15; done
scripts/guarded --name bench-dualhead-gaps --mem 16G --vram 0.2 -- uv run python "$S" dualhead $(scenes gap)
scripts/guarded --name bench-yolo-gaps --mem 16G --vram 0.2 -- uv run python "$S" yolo26s $(scenes gap)
echo GAPS DONE
