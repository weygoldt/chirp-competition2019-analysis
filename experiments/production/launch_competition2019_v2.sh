#!/usr/bin/env bash
# Full 2019 competition run with the final pipeline (2026-11-13). Re-running resumes.
# The run config names its output directory and the bundles by absolute path.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$CHIRP_REPO"
unset VIRTUAL_ENV
LOG=$CHIRP_DATA/detections/competition2019_v2_launch_$(date +%Y%m%d-%H%M%S).log
echo "chirpdetector $(git -C packages/chirpdetector rev-parse --short HEAD), root $(git rev-parse --short HEAD)" | tee "$LOG"
scripts/guarded --name pipeline-competition2019 --mem 24G --vram 0.4 -- \
  uv run chirpdetector run configs/detect/competition2019_v2.toml --dataset "$CHIRP_REC_2019" \
  2>&1 | tee -a "$LOG"
echo "exit: ${PIPESTATUS[0]}" | tee -a "$LOG"
