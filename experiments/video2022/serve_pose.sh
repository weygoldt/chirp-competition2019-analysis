#!/usr/bin/env bash
# Start a fish-keypoint labelling session as a systemd user service (survives the shell).
#
#   experiments/video2022/serve_pose.sh SESSION_NAME [PORT]    # e.g. r1  -> /mnt/data1/chirp_v2/video2022/labels/r1
#
# Stops any other pose-labeller-* service on the same port first. Then open
# http://localhost:PORT/ (from a laptop: ssh -N -L PORT:localhost:PORT polarbear).
# Every save is written at once; stop with: systemctl --user stop pose-labeller-SESSION_NAME
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=${1:?session name, e.g. r1}
port=${2:-8766}
session=/mnt/data1/chirp_v2/video2022/labels/$name
py=${CHIRP_PY:-/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python}
[[ -f "$session/items.json" ]] || { echo "no items.json in $session" >&2; exit 1; }
for u in $(systemctl --user list-units --type=service --no-pager --plain 'pose-labeller-*' | awk '{print $1}' | grep -e '^pose-labeller-'); do
  if systemctl --user show "$u" -p ExecStart | grep -q -e "--port $port"; then systemctl --user stop "$u"; echo "stopped $u"; fi
done
systemd-run --user --unit="pose-labeller-$name" --working-directory="$here" \
  "$py" "$here/pose_labeller/server.py" "$session" --port "$port"
echo "labeller: http://localhost:$port/  (laptop: ssh -N -L $port:localhost:$port polarbear)"
echo "stop:     systemctl --user stop pose-labeller-$name"
