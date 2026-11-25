#!/usr/bin/env bash
# Arm Bc (arm B continued, continue_longonly.sh) on the breakdown benchmark at
# one epoch: wait for the epoch's snapshot, build the bundle with the emitter
# unit repaired (make_bc_bundles.py), predict once (GPU), then the four
# evaluations (before/after post-processing x 100/25 ms matching) in parallel.
#   experiments/ablation_scenarios/bench_bc.sh EPOCH      # -> bench/Bc<EPOCH>
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../env.sh"
epoch=$1
venv=${CHIRP_VENV:-$HOME/wrk/analyses/chirp_competition_upgrade/.venv}
guard=$HOME/wrk/analyses/chirp_competition_upgrade/scripts/guarded
run=$CHIRP_DATA/runs/dualhead_r18_sim_v4_longonly_r1c
snap=$(printf "%s/epochs/epoch_%02d.pt" "$run" "$epoch")
bundle=$(printf "%s/bundles/epoch_%02d" "$run" "$epoch")
out=$CHIRP_DATA/runs/ablation_frontend/scenarios/bench/Bc$epoch
logs=$CHIRP_DATA/runs/ablation_frontend/scenarios/logs
ds=$CHIRP_DATA/datasets/ablation_bench_v1_B
mkdir -p "$logs" "$out"
unset VIRTUAL_ENV
until [[ -f $snap ]]; do sleep 60; done
sleep 30  # the watcher copies the snapshot; let the copy finish
"$venv/bin/python" "$here/make_bc_bundles.py" --epochs "$epoch"
"$guard" --name "bench-predict-Bc$epoch" --mem 16G --vram 0.3 -- \
  "$venv/bin/python" "$here/bench_predict.py" "$bundle" --dataset "$ds" --out "$out" --predict-only \
  > "$logs/predict_Bc${epoch}.log" 2>&1
for only in before_tol100 before_tol025 after_tol100 after_tol025; do
  "$guard" --name "bench-eval-Bc$epoch-$only" --mem 8G --vram 0 -- \
    "$venv/bin/python" "$here/bench_predict.py" "$bundle" --dataset "$ds" --out "$out" --reuse --only "$only" \
    > "$logs/predict_Bc${epoch}_$only.log" 2>&1 &
done
wait
grep -h "root: .* tol" "$logs"/predict_Bc"${epoch}"_*.log
