#!/usr/bin/env bash
# Optional: both arms on the blind-benchmark scenes (the 789 scenes behind the
# 1,010 human-labelled items: scenes_{old,new,gap}.txt of experiments/benchmark),
# one GPU job per arm, then `label-benchmark score` with both arms as models
# (same sets and features as labels/benchmark/report_v2).
#   experiments/ablation/run_benchmark.sh A|B|score ...   (in this order)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
root=$CHIRP_DATA/runs/ablation_frontend/benchmark
L=$CHIRP_REPO/experiments/benchmark
scenes() { cat "$L"/scenes_old.txt "$L"/scenes_new.txt "$L"/scenes_gap.txt | sed "s|^|$CHIRP_DATA/|"; }
cd "$CHIRP_REPO"
unset VIRTUAL_ENV
G=$HOME/wrk/analyses/chirp_competition_upgrade/scripts/guarded
mkdir -p "$root"
for step in "$@"; do
  case $step in
    A) bundle=$CHIRP_DATA/runs/dualhead_r18_sim_v4_r1/final/bundle ;;
    B) bundle=$CHIRP_DATA/runs/dualhead_r18_sim_v4_longonly_r1/final/bundle ;;
    score)
      B=$CHIRP_DATA/labels/benchmark
      $G --name ablation-bench-score --mem 8G --vram 0 -- uv run chirpdetector label-benchmark score --out "$root/report" \
        --set v1_final=$B/human_benchmark_v1_truth.parquet:$CHIRP_DATA/labels/human_benchmark_v1_final \
        --set v2_extra=$B/human_benchmark_v2_extra_truth.parquet:$CHIRP_DATA/labels/human_benchmark_v2_extra \
        --features $B/features --model A="$root/models/A" --model B="$root/models/B" > "$root/score.log" 2>&1
      continue ;;
    *) echo "usage: $0 A|B|score ..." >&2; exit 2 ;;
  esac
  $G --name "ablation-bench-$step" --mem 16G --vram 0.4 -- uv run python experiments/ablation/run_benchmark_models.py \
    "$step" "$bundle" "$root/models/$step" $(scenes) > "$root/run_$step.log" 2>&1
done
