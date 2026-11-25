#!/usr/bin/env bash
# Simulate the front-end breakdown benchmark (ablation_bench_v1): every level
# of both sweeps (levels.json, configs/ from make_configs.py) into one flat
# directory $CHIRP_DATA/scenes/ablation_bench_v1/<seed> (seeds are unique per
# level; the level is the scene name in scene.json).
#   experiments/ablation_scenarios/simulate.sh [WORKERS]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../env.sh"
workers=${1:-6}
venv=${CHIRP_VENV:-$HOME/wrk/analyses/chirp_competition_upgrade/.venv}
out=$CHIRP_DATA/scenes/ablation_bench_v1
mkdir -p "$out"
unset VIRTUAL_ENV
"$venv/bin/python" - "$here/levels.json" <<'EOF' > "$out/.plan"
import json, sys
levels = json.load(open(sys.argv[1]))
for sweep in ("df", "rate"):
    for lv in levels[sweep]:
        print(lv["name"], f"{lv['seeds'][0]}:{lv['seeds'][1]}")
EOF
while read -r name seeds; do
  "$venv/bin/gridtools" sim run "$here/configs/$name.toml" --seeds "$seeds" \
    --out "$out" --workers "$workers" --no-plots
done < "$out/.plan"
echo "done: $(find "$out" -mindepth 2 -maxdepth 2 -name scene.json | wc -l) scenes in $out"
