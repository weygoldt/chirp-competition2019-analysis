#!/usr/bin/env bash
# Arm B continued to arm A's training budget (see prepare_longonly_cont.py):
# resume dualhead_r18_sim_v4_longonly_r1 after epoch 10, snapshot last.pt after
# every validated epoch (epochs/epoch_NN.pt), and stop the fit (SIGTERM to its
# guarded scope; the fit ships the best checkpoint <= that epoch) once epoch
# STOP_AFTER (default 22, where arm A stopped) is validated.
#
# Launch detached (survives the session; the unit must not start with chirpv2-):
#   systemd-run --user --unit=ablation-cont-$(date +%H%M%S) --working-directory="$PWD" \
#     -E PATH="$PATH" -E HOME="$HOME" "$PWD/experiments/ablation_scenarios/continue_longonly.sh"
# (systemd-run needs the absolute path; the script must be executable)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
config=$CHIRP_REPO/experiments/ablation_scenarios/dualhead_r18_sim_v4_longonly_cont.toml
dataset=$CHIRP_FAST/datasets/sim_v4_longonly
run=$CHIRP_DATA/runs/dualhead_r18_sim_v4_longonly_r1c
stop_after=${STOP_AFTER:-22}
logs=$CHIRP_DATA/runs/dualhead_logs
mkdir -p "$logs" "$run/epochs"
log="$logs/$(basename "$run")_$(date +%Y%m%d-%H%M%S).log"
snap="$run/epochs/snapshots.log"
echo "$(date -Is) start; log $log" >> "$snap"
cd "$CHIRP_REPO"
unset VIRTUAL_ENV
"$HOME/wrk/analyses/chirp_competition_upgrade/scripts/guarded" \
  --name ablation-cont --mem 24G --vram 0.4 -- \
  "${CHIRP_VENV:-$HOME/wrk/analyses/chirp_competition_upgrade/.venv}/bin/chirpdetector" \
  train detector "$config" --dataset "$dataset" --run-dir "$run" \
  > "$log" 2>&1 &
train=$!
curves=$run/final/curves.csv
seen=10 stopped=0
while kill -0 "$train" 2>/dev/null; do
  sleep 15
  [[ -f $curves ]] || continue
  last=$(tail -n 1 "$curves" | cut -d, -f1)
  [[ $last =~ ^[0-9]+$ ]] || continue
  if (( last > seen )); then
    cp "$run/final/dualhead/last.pt" "$run/epochs/epoch_$(printf %02d "$last").pt"
    seen=$last
    echo "$(date -Is) snapshot epoch $last" >> "$snap"
    if (( last >= stop_after && stopped == 0 )); then
      for unit in $(systemctl --user list-units --type=scope --plain --no-legend 'chirpv2-ablation-cont-*' | awk '{print $1}'); do
        echo "$(date -Is) SIGTERM $unit after epoch $last" >> "$snap"
        systemctl --user kill --signal=SIGTERM "$unit" || true
      done
      stopped=1
    fi
  fi
done
set +e
wait "$train"
status=$?
echo "$(date -Is) training exited $status" >> "$snap"
exit "$status"
