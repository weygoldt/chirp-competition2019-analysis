#!/usr/bin/env bash
# Start a chase-review session as a systemd user service (survives the shell and Claude sessions).
#
#   paper/chase_review/serve.sh NAME [PORT]    # NAME -> /mnt/data1/chirp_v2/labels/chase_review_2019/NAME
#
# Build the session first (once): paper/chase_review/build_session.py NAME
# Then open http://localhost:PORT/ (from a laptop: ssh -N -L PORT:localhost:PORT polarbear).
# Every save is written at once; stop with: systemctl --user stop chase-review-NAME
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=${1:?session name, e.g. offsets_v1}
port=${2:-8769}
session=${CHASE_REVIEW_ROOT:-/mnt/data1/chirp_v2/labels/chase_review_2019}/$name
py=${CHIRP_PY:-$here/../.venv/bin/python}
[[ -x "$py" ]] || py=python3   # the server needs only the standard library (and ffmpeg)
[[ -f "$session/session.json" ]] || { echo "no session.json in $session; run build_session.py $name first" >&2; exit 1; }
systemd-run --user --unit="chase-review-$name" --working-directory="$here" \
  "$py" "$here/server.py" "$session" --port "$port"
echo "chase review: http://localhost:$port/  (laptop: ssh -N -L $port:localhost:$port polarbear)"
echo "stop:         systemctl --user stop chase-review-$name"
