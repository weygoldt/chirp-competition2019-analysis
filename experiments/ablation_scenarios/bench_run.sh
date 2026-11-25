#!/usr/bin/env bash
# Breakdown benchmark after simulate.sh: build the two front-end datasets and
# run each arm's bundle on its own front end (A on ablation_bench_v1_A, B on
# ablation_bench_v1_B), through the guard. Optional arm "Bc" (arm B continued to
# 22 epochs, continue_longonly.sh): BC_BUNDLE=<bundle dir> bench_run.sh predict Bc
# REUSE=1 reuses an arm's cached raw predictions (bench/ARM/predictions_raw.parquet).
#   experiments/ablation_scenarios/bench_run.sh build|predict [ARM...]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../env.sh"
venv=${CHIRP_VENV:-$HOME/wrk/analyses/chirp_competition_upgrade/.venv}
guard=$HOME/wrk/analyses/chirp_competition_upgrade/scripts/guarded
out=$CHIRP_DATA/runs/ablation_frontend/scenarios
mkdir -p "$out/logs" "$out/bench"
unset VIRTUAL_ENV
step=$1; shift
arms=("$@"); [[ ${#arms[@]} -gt 0 ]] || arms=(A B)
for arm in "${arms[@]}"; do
  case $step in
    build)
      "$guard" --name "bench-build-$arm" --mem 24G --vram 0.2 -- \
        "$venv/bin/python" "$here/bench_build.py" "$arm" --workers 6 \
        > "$out/logs/build_$arm.log" 2>&1
      tail -n 1 "$out/logs/build_$arm.log" ;;
    predict)
      case $arm in
        A)  bundle=$CHIRP_DATA/runs/dualhead_r18_sim_v4_r1/final/bundle;          ds=$CHIRP_DATA/datasets/ablation_bench_v1_A ;;
        B)  bundle=$CHIRP_DATA/runs/dualhead_r18_sim_v4_longonly_r1/final/bundle; ds=$CHIRP_DATA/datasets/ablation_bench_v1_B ;;
        Bc) bundle=${BC_BUNDLE:?set BC_BUNDLE};                                    ds=$CHIRP_DATA/datasets/ablation_bench_v1_B ;;
        *) echo "unknown arm $arm" >&2; exit 2 ;;
      esac
      "$guard" --name "bench-predict-$arm" --mem 16G --vram 0.3 -- \
        "$venv/bin/python" "$here/bench_predict.py" "$bundle" --dataset "$ds" --out "$out/bench/$arm" ${REUSE:+--reuse} \
        > "$out/logs/predict_$arm.log" 2>&1
      grep -E "after tol|before tol" "$out/logs/predict_$arm.log" | tail -n 4 ;;
    *) echo "usage: $0 build|predict [ARM...]" >&2; exit 2 ;;
  esac
done
