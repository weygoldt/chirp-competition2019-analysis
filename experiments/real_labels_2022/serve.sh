#!/usr/bin/env bash
# Start (or resume) the 2022 labelling session real_v2_2022: the blind browser
# labeller with real_v1's view, as a transient systemd user service (survives
# the terminal and the Claude session).
#
#   experiments/real_labels_2022/serve.sh            # port 8767
#   PORT=8770 experiments/real_labels_2022/serve.sh
#   systemctl --user stop chirp-labeller-real-v2-2022  # stop; nothing is lost
#
# Then, on the laptop: ssh -N -L 8767:localhost:8767 USER@HOST and open
# http://localhost:8767/ . Every click is saved at once; running this again
# resumes the same queue (same --out).
#
# chirpdetector: main of the main checkout's packages/chirpdetector (55b993a,
# = branch v2-real-labels that served real_v1, merged 2026-11-16).
# Unlike real_v1, the server is not wrapped in scripts/guarded: the watchdog
# stops guarded (chirpv2-*) scopes, newest first, whenever VRAM use exceeds
# its limit, and a GPU training runs in parallel. The service caps its own
# memory instead (8 GB) and sees no GPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PORT=${PORT:-8767}
UPGRADE=${UPGRADE:-$HOME/wrk/analyses/chirp_competition_upgrade}
CHIRPDETECTOR_SRC=${CHIRPDETECTOR_SRC:-$UPGRADE/packages/chirpdetector/src}
CANDIDATES=$CHIRP_DATA/labels/candidates/real_v2_2022.parquet
OUT=$CHIRP_DATA/labels/real_v2_2022_labels
LOG=$CHIRP_DATA/logs/real_labels_2022/labeller_$(date +%Y%m%d-%H%M%S).log
UNIT=chirp-labeller-real-v2-2022
if systemctl --user is-active --quiet "$UNIT"; then
  echo "$UNIT is already running (systemctl --user status $UNIT)"; exit 0
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
if ss -ltn "sport = :$PORT" | /usr/bin/grep -q LISTEN; then
  echo "port $PORT is in use; choose another: PORT=... $0" >&2; exit 1
fi
mkdir -p "$(dirname "$LOG")"
systemd-run --user --unit="$UNIT" --working-directory="$UPGRADE" \
  -p MemoryMax=8G -p MemorySwapMax=0 \
  -E PATH="$PATH" -E HOME="$HOME" -E PYTHONPATH="$CHIRPDETECTOR_SRC" -E CUDA_VISIBLE_DEVICES= \
  bash -c "unset VIRTUAL_ENV; exec $UPGRADE/.venv/bin/chirpdetector label-emitters \
    --candidates $CANDIDATES --out $OUT \
    --blind --span-s 5 --span-hz 600 --keep-order --serve --port $PORT > $LOG 2>&1"
echo "started $UNIT on 127.0.0.1:$PORT; log $LOG"
