#!/usr/bin/env bash
# dual-head on the new scenes, then YOLO26s on old and new (one GPU job at a
# time, vram 0.2). Scene lists: scenes_{new,old}.txt (relative to $CHIRP_DATA).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$CHIRP_REPO"
S=$CHIRP_REPO/experiments/benchmark/run_models_v2.py
L=$CHIRP_REPO/experiments/benchmark
scenes() { sed "s|^|$CHIRP_DATA/|" "$L/scenes_$1.txt"; }
# (the original waited here for the dual-head run on the old scenes to finish)
until ! pgrep -f "run_models_v2.py dualhead $CHIRP_DATA/scenes/train_v4" > /dev/null; do sleep 10; done
scripts/guarded --name bench-dualhead-new --mem 16G --vram 0.2 -- uv run python "$S" dualhead $(scenes new)
scripts/guarded --name bench-yolo-old --mem 16G --vram 0.2 -- uv run python "$S" yolo26s $(scenes old)
scripts/guarded --name bench-yolo-new --mem 16G --vram 0.2 -- uv run python "$S" yolo26s $(scenes new)
echo CHAIN DONE
