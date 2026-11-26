#!/usr/bin/env bash
# Scenes of train_v4r = train_v4 + rapid-chirping scenes (front-end ablation round 2).
#
# 1. $CHIRP_DATA/scenes/train_v4r/{train,val,test}: hard-linked copies of
#    train_v4/{train,val,test} (cp -al: same disk, no extra space; train_v4
#    itself is never written). The dataset builder assigns splits by the
#    resolved scene directory, so symbolic links would not do.
# 2. The rapid scenes (configs/train_v4_rapid_*.toml from make_configs.py)
#    simulated into train_v4r/train (seeds 40000:40300) and train_v4r/val
#    (40300:40340). The test split stays train_v4's (the rapid regime is
#    tested by the breakdown benchmark, ablation_bench_v1).
#
#   experiments/ablation_scenarios/rapid_training/prepare_scenes.sh [WORKERS]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../env.sh"
workers=${1:-6}
venv=${CHIRP_VENV:-$HOME/wrk/analyses/chirp_competition_upgrade/.venv}
src=$CHIRP_DATA/scenes/train_v4
dst=$CHIRP_DATA/scenes/train_v4r
unset VIRTUAL_ENV
for split in train val test; do
  if [[ ! -d $dst/$split ]]; then
    mkdir -p "$dst"
    cp -al "$src/$split" "$dst/$split"
    echo "linked $src/$split -> $dst/$split"
  fi
done
"$venv/bin/gridtools" sim run "$here/configs/train_v4_rapid_train.toml" --seeds 40000:40300 \
  --out "$dst/train" --workers "$workers" --no-plots
"$venv/bin/gridtools" sim run "$here/configs/train_v4_rapid_val.toml" --seeds 40300:40340 \
  --out "$dst/val" --workers "$workers" --no-plots
for split in train val test; do
  echo "$split: $(find "$dst/$split" -mindepth 2 -maxdepth 2 -name scene.json | wc -l) scenes"
done
