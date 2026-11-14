#!/usr/bin/env bash
# Start (or resume) the real_v1 labelling session: the blind browser
# labeller with the simulated benchmark's view, as a transient systemd user
# service (survives the terminal; stop it with
# `systemctl --user stop mining-labeller-real-v1`).
#
#   experiments/real_labels/serve.sh            # port 8765
#   PORT=8770 experiments/real_labels/serve.sh
#
# Then, on the laptop: ssh -N -L 8765:localhost:8765 USER@HOST and open
# http://localhost:8765/ . Every click is saved at once; running this again
# resumes the same queue (same --out).
#
# The labeller needs chirpdetector's window items (branch v2-real-labels):
# CHIRPDETECTOR_SRC points at that checkout until the branch is merged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PORT=${PORT:-8765}
UPGRADE=${UPGRADE:-$HOME/wrk/analyses/chirp_competition_upgrade}
CHIRPDETECTOR_SRC=${CHIRPDETECTOR_SRC:-$UPGRADE/.worktrees/real-labels/chirpdetector/src}
CANDIDATES=$CHIRP_DATA/labels/candidates/real_v1.parquet
OUT=$CHIRP_DATA/labels/real_v1_labels
LOG=$CHIRP_DATA/logs/real_labels/labeller_$(date +%Y%m%d-%H%M%S).log
UNIT=mining-labeller-real-v1
if systemctl --user is-active --quiet "$UNIT"; then
  echo "$UNIT is already running (systemctl --user status $UNIT)"; exit 0
fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "port $PORT is in use; choose another: PORT=... $0" >&2; exit 1
fi
mkdir -p "$(dirname "$LOG")"
systemd-run --user --unit="$UNIT" --working-directory="$UPGRADE" \
  -E PATH="$PATH" -E HOME="$HOME" -E PYTHONPATH="$CHIRPDETECTOR_SRC" \
  bash -c "unset VIRTUAL_ENV; $UPGRADE/scripts/guarded --name labeller-real-v1 --mem 8G --vram 0 -- \
    $UPGRADE/.venv/bin/chirpdetector label-emitters --candidates $CANDIDATES --out $OUT \
    --blind --span-s 5 --span-hz 600 --keep-order --serve --port $PORT > $LOG 2>&1; \
    echo \"exit \$?\" >> $LOG"
echo "started $UNIT on 127.0.0.1:$PORT; log $LOG"
