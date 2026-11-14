#!/usr/bin/env bash
# How the remaining steps were finally run (2026-11-14 00:38-01:48), as a systemd user
# service (systemd-run --user ... run_remaining.sh) with CPU boost disabled after three
# hard resets of the workstation during arm B's evaluation (see REPORT.md).
# Remaining steps of the front-end ablation, in order (2026-11-13, after two hard resets):
# arm B eval_postprocess (the step that was running at both resets), arm B on the 365
# real labels, arm B on the blind-benchmark scenes, the benchmark score, compare.py.
# Each step writes <name>_exit.log; the chain stops at the first failure.
set -uo pipefail
WT=/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.worktrees/ablation/root
source "$WT/experiments/env.sh"
R=$CHIRP_DATA/runs/ablation_frontend
L=$R/logs
cd "$WT"
unset VIRTUAL_ENV
step() {
  local name=$1; shift
  echo "[$(date -Is)] start $name" >> "$L/chain.log"; sync
  "$@"
  local rc=$?
  echo "$name exit $rc" > "$L/${name}_exit.log"
  echo "[$(date -Is)] $name exit $rc" >> "$L/chain.log"; sync
  return $rc
}
step evalB experiments/ablation/run_eval.sh B || exit 1
step real365_B bash -c "experiments/ablation/eval_real_labels_ablation.sh $R/real_labels_365 B=$R/B/eval_postprocess > $L/real365_B.log 2>&1" || exit 1
step benchB experiments/ablation/run_benchmark.sh B || exit 1
step bench_score experiments/ablation/run_benchmark.sh score || exit 1
step compare bash -c "uv run python experiments/ablation/compare.py $R --sim-a $CHIRP_FAST/datasets/sim_v4 --sim-b $CHIRP_FAST/datasets/sim_v4_longonly > $L/compare.log 2>&1" || exit 1
echo "[$(date -Is)] chain done" >> "$L/chain.log"
