#!/usr/bin/env bash
# Front-end ablation, round 2: train, evaluate and compare both front ends on
# training data that contains rapid chirp trains, across seeds.
#
# Both front ends (A: three windows, dataset sim_v4r; B: the 205-ms window in
# all three channels, sim_v4r_longonly; both = train_v4 + 300/40 rapid-chirping
# train/val scenes, prepare_scenes.sh + build_datasets.py) are trained with the
# released detector's config with early stopping off (dualhead_r18_v4r.toml:
# 30 epochs, best epoch by validation fitness), once per seed. Every model is
# then evaluated on
#   - the breakdown benchmark (ablation_bench_v1; ../bench_predict.py),
#   - the simulated test set and the 335 real labels
#     (experiments/evaluation/eval_postprocess.py), the 365 real labels
#     (experiments/ablation/eval_real_labels_ablation.sh),
#   - the real chirp train of Fig. S4C (real_train_counts.py),
# and compare_round2.py tabulates both arms across seeds
# ($CHIRP_DATA/runs/ablation_frontend/round2/compare/tables.md).
#
# Launch from the repository root as a user service (survives the session; the
# unit name must not start with chirpv2-, which the OOM watchdog stops). Turn
# CPU boost off first (single-core boost/idle transitions reset this machine;
# the setting is lost at every reboot):
#
#   echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
#   systemd-run --user --unit=ablation-r2-$(date +%m%d-%H%M) --working-directory="$PWD" \
#     -E PATH="$PATH" -E HOME="$HOME" \
#     "$PWD/experiments/ablation_scenarios/rapid_training/run_round2.sh"
#
#   tail -f /mnt/data1/chirp_v2/runs/ablation_frontend/round2/logs/round2.log
#   systemctl --user stop 'ablation-r2-*'     # stop (running fits ship their best epoch,
#                                             # but are not marked done: rerun retrains them)
#
# Stages: run_round2.sh [train] [evaluate] [compare]   (default: all three, in order)
# Settings (environment): SEEDS="1 2 3"  ARMS="A B"  PARALLEL_TRAIN=2 (fits at a
# time: ~3 GB VRAM and <= 20 GB RAM each)  PARALLEL_EVAL=2 (models evaluated at a time).
# Resumable: every finished step leaves a marker (runs/<name>/DONE,
# eval/<name>.DONE) and is skipped when the script runs again. A training run
# counts as finished only if its bundle records all 30 epochs; an unfinished
# one is moved aside (<run>.incomplete-<time>) and trained again from scratch,
# because [model] resume leaves the EMA's emitter unit at 1.0 (docs/status.md).
# Expected time: ~12-13 min per epoch (sim_v4 + ~11 % tiles), ~6.5 h per fit;
# 6 fits two at a time ~20 h; evaluation ~1.5 h per model, two at a time ~5 h.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../env.sh"
venv=${CHIRP_VENV:-$CHIRP_REPO/.venv}
guard=$CHIRP_REPO/scripts/guarded
R=$CHIRP_DATA/runs/ablation_frontend/round2
SEEDS=${SEEDS:-"1 2 3"}
ARMS=${ARMS:-"A B"}
PARALLEL_TRAIN=${PARALLEL_TRAIN:-2}
PARALLEL_EVAL=${PARALLEL_EVAL:-2}
EPOCHS=30
declare -A DATASET=([A]=$CHIRP_DATA/datasets/sim_v4r [B]=$CHIRP_DATA/datasets/sim_v4r_longonly)
declare -A BENCH=([A]=$CHIRP_DATA/datasets/ablation_bench_v1_A [B]=$CHIRP_DATA/datasets/ablation_bench_v1_B)
mkdir -p "$R"/{runs,configs,logs,bench,eval,s4c}
unset VIRTUAL_ENV
cd "$CHIRP_REPO"

log() { echo "[$(date -Is)] $*" | tee -a "$R/logs/round2.log"; }

# run "$@" in the background with at most $1 such jobs at a time
limited() {
  local n=$1; shift
  while (( $(jobs -rp | wc -l) >= n )); do wait -n || true; done
  "$@" &
}

names() { for seed in $SEEDS; do for arm in $ARMS; do echo "${arm}_s${seed}"; done; done; }

preflight() {
  for arm in $ARMS; do
    [[ -f ${DATASET[$arm]}/dataset.json ]] || { log "missing ${DATASET[$arm]}: run prepare_scenes.sh and build_datasets.py $arm"; exit 2; }
    [[ -f ${BENCH[$arm]}/dataset.json ]] || { log "missing ${BENCH[$arm]} (bench_run.sh build)"; exit 2; }
  done
  [[ $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 0) == 0 ]] \
    || log "WARNING: CPU boost is on (echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost)"
  local free_gb
  free_gb=$(df -BG --output=avail "$CHIRP_DATA" | tail -1 | tr -dc 0-9)
  (( free_gb > 100 )) || { log "only ${free_gb} GB free on $CHIRP_DATA"; exit 2; }
  log "round 2: seeds [$SEEDS], arms [$ARMS], $PARALLEL_TRAIN fits / $PARALLEL_EVAL evaluations at a time;" \
      "GPU in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo '?')"
}

train_one() {
  local name=$1 arm=${1%%_*} seed=${1#*_s}
  local run=$R/runs/$name config=$R/configs/$name.toml
  if [[ -f $run/DONE ]]; then log "train $name: done before, skipped"; return 0; fi
  if [[ -e $run ]]; then
    mv "$run" "$run.incomplete-$(date +%Y%m%d-%H%M%S)"
    log "train $name: unfinished run moved aside, training again from scratch"
  fi
  sed "s/^seed = 0$/seed = $seed/" "$here/dualhead_r18_v4r.toml" > "$config"
  grep -q "^seed = $seed$" "$config"
  log "train $name: start (seed $seed, ${DATASET[$arm]})"
  "$guard" --name "r2-train-$name" --mem 20G --vram 0.35 -- \
    "$venv/bin/chirpdetector" train detector "$config" --dataset "${DATASET[$arm]}" --run-dir "$run" \
    > "$R/logs/train_$name.log" 2>&1 || true
  local record=$run/final/bundle/bundle.json epochs best
  if [[ -f $record ]]; then
    epochs=$("$venv/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))['training']['epochs_run'])" "$record")
    best=$("$venv/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))['training']['best_epoch'])" "$record")
    if [[ $epochs == "$EPOCHS" ]]; then
      touch "$run/DONE"; log "train $name: done (best epoch $best of $epochs)"; return 0
    fi
    log "train $name: stopped after $epochs of $EPOCHS epochs; not marked done (rerun trains it again)"
  else
    log "train $name: FAILED, no bundle (log $R/logs/train_$name.log)"
  fi
  return 1
}

# one guarded step of an evaluation: step NAME LOG MEM VRAM CMD...; logs a failure
step() {
  local what=$1 logfile=$2 mem=$3 vram=$4; shift 4
  if "$guard" --name "r2-$what" --mem "$mem" --vram "$vram" -- "$@" > "$logfile" 2>&1; then
    return 0
  fi
  log "$what: FAILED (log $logfile)"
  return 1
}

eval_one() {
  local name=$1 arm=${1%%_*}
  local run=$R/runs/$name bundle=$R/runs/$name/final/bundle py=$venv/bin/python
  if [[ -f $R/eval/$name.DONE ]]; then log "eval $name: done before, skipped"; return 0; fi
  [[ -f $run/DONE ]] || { log "eval $name: training not finished, skipped"; return 0; }
  log "eval $name: breakdown benchmark"
  step "bench-$name" "$R/logs/bench_$name.log" 16G 0.25 \
    "$py" "$here/../bench_predict.py" "$bundle" --dataset "${BENCH[$arm]}" --out "$R/bench/$name" \
    --predict-only || return 1
  local pids=() failed=0
  for only in after_tol025 after_tol100; do
    step "bencheval-$name-$only" "$R/logs/bench_${name}_$only.log" 10G 0 \
      "$py" "$here/../bench_predict.py" "$bundle" --dataset "${BENCH[$arm]}" --out "$R/bench/$name" \
      --reuse --only "$only" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  (( failed )) && return 1
  log "eval $name: simulated test set and 335 real labels"
  step "pp-$name" "$R/logs/eval_postprocess_$name.log" 24G 0.3 \
    "$py" experiments/evaluation/eval_postprocess.py "$run" --sim "${DATASET[$arm]}" \
    --out "$R/eval/$name" || return 1
  log "eval $name: real chirp train of Fig. S4C"
  step "s4c-$name" "$R/logs/s4c_$name.log" 8G 0.2 \
    "$py" "$here/real_train_counts.py" "$bundle" --out "$R/s4c/$name.json" || return 1
  touch "$R/eval/$name.DONE"
  log "eval $name: done"
}

stage_train() {
  log "=== train"
  for name in $(names); do limited "$PARALLEL_TRAIN" train_one "$name"; done
  wait || true
}

stage_evaluate() {
  log "=== evaluate"
  for name in $(names); do limited "$PARALLEL_EVAL" eval_one "$name"; done
  wait || true
  local pairs=()
  for name in $(names); do [[ -f $R/eval/$name.DONE ]] && pairs+=("$name=$R/eval/$name"); done
  if (( ${#pairs[@]} )); then
    log "eval: 365 real labels for ${#pairs[@]} models"
    "$CHIRP_REPO/experiments/ablation/eval_real_labels_ablation.sh" "$R/real_labels_365" "${pairs[@]}" \
      > "$R/logs/real_labels_365.log" 2>&1 || log "eval: 365 real labels FAILED (log $R/logs/real_labels_365.log)"
  fi
}

stage_compare() {
  log "=== compare"
  "$venv/bin/python" "$here/compare_round2.py" --root "$R" > "$R/logs/compare.log" 2>&1 \
    && log "compare: $R/compare/tables.md" || log "compare FAILED (log $R/logs/compare.log)"
}

stages=("$@")
(( ${#stages[@]} )) || stages=(train evaluate compare)
preflight
for s in "${stages[@]}"; do
  case $s in
    train) stage_train ;;
    evaluate) stage_evaluate ;;
    compare) stage_compare ;;
    *) log "unknown stage $s (train, evaluate, compare)"; exit 2 ;;
  esac
done
missing=()
for name in $(names); do
  [[ " ${stages[*]} " == *" train "* && ! -f $R/runs/$name/DONE ]] && missing+=("train $name")
  [[ " ${stages[*]} " == *" evaluate "* && ! -f $R/eval/$name.DONE ]] && missing+=("eval $name")
done
if (( ${#missing[@]} )); then
  log "unfinished: ${missing[*]}; run the script again to retry them"
  exit 1
fi
log "stages done: ${stages[*]}"
[[ " ${stages[*]} " == *" compare "* ]] && log "round 2 tables: $R/compare/tables.md"
exit 0
