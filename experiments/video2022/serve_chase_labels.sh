#!/usr/bin/env bash
# Start a chase onset / offset labelling session as a systemd user service (survives the shell).
#
#   experiments/video2022/serve_chase_labels.sh SESSION_NAME [PORT]   # e.g. chase_v1 -> /mnt/data1/chirp_v2/video2022/chases/labels/chase_v1
#
# Then open http://localhost:PORT/ (from a laptop: ssh -N -L PORT:localhost:PORT polarbear).
# Every save is written at once; stop with: systemctl --user stop chase-labeller-SESSION_NAME
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=${1:?session name, e.g. chase_v1}
port=${2:-8768}
session=/mnt/data1/chirp_v2/video2022/chases/labels/$name
py=${CHIRP_PY:-/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python}
[[ -f "$session/clips.json" ]] || { echo "no clips.json in $session" >&2; exit 1; }
systemd-run --user --unit="chase-labeller-$name" --working-directory="$here" \
  "$py" "$here/chase_labeller/server.py" "$session" --port "$port"
echo "labeller: http://localhost:$port/  (laptop: ssh -N -L $port:localhost:$port polarbear)"
echo "stop:     systemctl --user stop chase-labeller-$name"
