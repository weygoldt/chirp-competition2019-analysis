#!/usr/bin/env bash
# Start the B2 fish-keypoint labelling session as a systemd user service (survives the shell).
#
#   experiments/video2022/serve_b2.sh [SESSION_DIR]     # default /mnt/data1/chirp_v2/video2022/labels/b2_v1
#
# Then open http://localhost:8766/ (from a laptop: ssh -N -L 8766:localhost:8766 polarbear).
# Every save is written at once; stop with: systemctl --user stop pose-labeller-b2
# Restarting resumes the same queue (labels.jsonl in the session directory).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
session=${1:-/mnt/data1/chirp_v2/video2022/labels/b2_v1}
py=${CHIRP_PY:-/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python}
[[ -f "$session/items.json" ]] || { echo "no items.json in $session" >&2; exit 1; }
systemd-run --user --unit=pose-labeller-b2 --working-directory="$here" \
  "$py" "$here/pose_labeller/server.py" "$session" --port 8766
echo "labeller: http://localhost:8766/  (laptop: ssh -N -L 8766:localhost:8766 polarbear)"
echo "stop:     systemctl --user stop pose-labeller-b2"
