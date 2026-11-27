#!/usr/bin/env bash
# Start (or resume) the short labelling session real_sw (sw_session.py build):
# the new single-window detector's boxes without an old counterpart, plus a few
# real_v1 items again, in the blind browser labeller with real_v1's view
# (--blind --span-s 5 --span-hz 600 --keep-order), as a transient systemd user
# service (stop it with `systemctl --user stop mining-labeller-real-sw`).
#
#   experiments/real_labels/serve_sw.sh            # port 8766
#   PORT=8770 experiments/real_labels/serve_sw.sh
#
# Then, on the laptop: ssh -N -L 8766:localhost:8766 USER@HOST and open
# http://localhost:8766/ . Every click is saved at once; running this again
# resumes the same queue (same --out). Keys as in real_v1 (labels/real_v1/README.md):
# click on the emitter's baseline = chirp of that fish, 0 = chirp, emitter
# unclear, x = not a chirp, s = skip, u = undo.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
PORT=${PORT:-8766}
UPGRADE=${UPGRADE:-$HOME/wrk/analyses/chirp_competition_upgrade}
CANDIDATES=$CHIRP_DATA/labels/candidates/real_sw.parquet
OUT=$CHIRP_DATA/labels/real_sw_labels
LOG=$CHIRP_DATA/logs/real_labels/labeller_sw_$(date +%Y%m%d-%H%M%S).log
UNIT=mining-labeller-real-sw
[[ -f $CANDIDATES ]] || { echo "no $CANDIDATES: run sw_session.py build first" >&2; exit 1; }
if systemctl --user is-active --quiet "$UNIT"; then
  echo "$UNIT is already running (systemctl --user status $UNIT)"; exit 0
fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "port $PORT is in use; choose another: PORT=... $0" >&2; exit 1
fi
mkdir -p "$(dirname "$LOG")"
systemd-run --user --unit="$UNIT" --working-directory="$UPGRADE" \
  -E PATH="$PATH" -E HOME="$HOME" \
  bash -c "unset VIRTUAL_ENV; $UPGRADE/scripts/guarded --name labeller-real-sw --mem 8G --vram 0 -- \
    $UPGRADE/.venv/bin/chirpdetector label-emitters --candidates $CANDIDATES --out $OUT \
    --blind --span-s 5 --span-hz 600 --keep-order --serve --port $PORT > $LOG 2>&1; \
    echo \"exit \$?\" >> $LOG"
echo "started $UNIT on 127.0.0.1:$PORT; log $LOG"
