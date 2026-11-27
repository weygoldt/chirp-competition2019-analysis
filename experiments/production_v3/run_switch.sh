#!/usr/bin/env bash
# Production switch v3 (author's decision 2026-11-27): the single-window
# detector trained on train_v4r (front-end ablation round 2, seed 3) replaces
# the released three-window detector. Everything the paper needs from the GPU:
#
#   promote     copy the selected model (round2/runs/B_s3) to
#               $CHIRP_DATA/runs/dualhead_r18_single_v4r_s3 (checks weights 0ee36061)
#   decoder     train the chirp-parameter decoder on that detector's boxes,
#               scenes train_v4r (experiments/decoder/params_v1.toml), ~5 h
#   recal       refit its calibration without the octave-error template 1598
#               (as the v2 decoder's quick fix of 2026-11-27) -> v4r-single-s3-recal1
#   decodereval evaluate the decoder on the simulated test chirps (Fig. 10 I-K)
#   production  chirpdetector run configs/detect/competition2019_v3.toml over the
#               50 recordings of 2019/2020, ~2 h
#   gapfill     experiments/production/gapfill_reassign.py -> competition2019_v3_gapfill
#   thresholds  make_threshold_variant.py at 0.8 (primary) and 0.9 -> *_gapfill_t080/_t090
#   copy        copy_detections_to_trials.py into the recording directories, file
#               names rtdetr_sw{,_gapfill,_gapfill_t080,_gapfill_t090} (next to the
#               v2 copies, never over them): the paper's PAPER_DETECTIONS names
#   bench       the new detector on the blind-benchmark scenes, then
#               label-benchmark score with the v2 models -> labels/benchmark/report_v3
#
# Launch from the repository root as a user service (survives the session; the
# unit name must not start with chirpv2-). CPU boost off first:
#
#   echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
#   systemd-run --user --unit=switch-v3-$(date +%m%d-%H%M) --working-directory="$PWD" \
#     -E PATH="$PATH" -E HOME="$HOME" "$PWD/experiments/production_v3/run_switch.sh"
#   tail -F /mnt/data1/chirp_v2/logs/switch_v3/switch.log
#
# Stages: run_switch.sh [STAGE ...] (default: all, in the order above). Every
# stage checks whether its output is complete and skips itself then, so a rerun
# continues where it stopped (an unfinished decoder run is moved aside and
# trained again; the production run resumes per recording by itself).
# Expected time: ~8-9 h in all.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/../env.sh"
venv=${CHIRP_VENV:-$CHIRP_REPO/.venv}
py=$venv/bin/python
cd_bin=$venv/bin/chirpdetector
guard=$CHIRP_REPO/scripts/guarded
LOGS=$CHIRP_DATA/logs/switch_v3
SRC_MODEL=$CHIRP_DATA/runs/ablation_frontend/round2/runs/B_s3
DET_RUN=$CHIRP_DATA/runs/dualhead_r18_single_v4r_s3
DET=$DET_RUN/final/bundle
DET_SHA=0ee36061
PARAMS_RUN=$CHIRP_DATA/params/runs/v4r-single-s3
PARAMS=$CHIRP_DATA/params/runs/v4r-single-s3-recal1/bundle
OCTAVE_TEMPLATE=1598
CONFIG=$CHIRP_REPO/configs/detect/competition2019_v3.toml
RUN=$CHIRP_DATA/detections/competition2019_v3
BENCH=$CHIRP_DATA/labels/benchmark
mkdir -p "$LOGS"
unset VIRTUAL_ENV
cd "$CHIRP_REPO"

log() { echo "[$(date -Is)] $*" | tee -a "$LOGS/switch.log"; }
fail() { log "FAILED: $*"; exit 1; }
# guarded NAME MEM VRAM LOG CMD...
guarded() {
  local name=$1 mem=$2 vram=$3 logfile=$4; shift 4
  "$guard" --name "switch-$name" --mem "$mem" --vram "$vram" -- "$@" > "$logfile" 2>&1 \
    || fail "$name (log $logfile)"
}
n_complete() {  # recordings of a run whose meta.json says complete
  "$py" - "$1" <<'EOF'
import json, sys
from pathlib import Path
print(sum(json.loads(m.read_text()).get("status") == "complete" for m in Path(sys.argv[1]).glob("*/meta.json")))
EOF
}

stage_promote() {
  if [[ -f $DET/bundle.json ]]; then log "promote: $DET exists, skipped"; return; fi
  [[ -f $SRC_MODEL/DONE ]] || fail "promote: $SRC_MODEL is not a finished run"
  local sha
  sha=$("$py" -c "import json,sys; print(json.load(open(sys.argv[1]))['weights_sha256'])" "$SRC_MODEL/final/bundle/bundle.json")
  [[ $sha == "$DET_SHA"* ]] || fail "promote: weights $sha, expected $DET_SHA"
  mkdir -p "$DET_RUN/final"
  cp -a "$SRC_MODEL/config.toml" "$SRC_MODEL/config.json" "$SRC_MODEL/dataset.json" "$SRC_MODEL/run.json" \
        "$SRC_MODEL/provenance.json" "$DET_RUN/"
  cp -a "$SRC_MODEL/final/curves.csv" "$SRC_MODEL/final/fit.json" "$SRC_MODEL/final/provenance.json" "$DET_RUN/final/"
  cp -a "$SRC_MODEL/final/bundle" "$DET_RUN/final/bundle"
  printf '{"source": "%s", "weights_sha256": "%s", "promoted_at": "%s", "reason": "%s"}\n' \
    "$SRC_MODEL" "$sha" "$(date -Is)" \
    "production switch v3: single-window detector, train_v4r, seed 3 (best validation fitness of 3 seeds)" \
    > "$DET_RUN/PROMOTED.json"
  log "promote: $SRC_MODEL -> $DET_RUN (weights $sha)"
}

stage_decoder() {
  if [[ -f $PARAMS_RUN/bundle/bundle.json ]]; then log "decoder: $PARAMS_RUN done, skipped"; return; fi
  if [[ -e $PARAMS_RUN ]]; then
    mv "$PARAMS_RUN" "$PARAMS_RUN.incomplete-$(date +%Y%m%d-%H%M%S)"
    log "decoder: unfinished run moved aside"
  fi
  log "decoder: training on train_v4r with the boxes of $DET (~5 h)"
  guarded decoder 24G 0.25 "$LOGS/decoder.log" \
    "$cd_bin" train params experiments/decoder/params_v1.toml \
    --scenes "$CHIRP_DATA/scenes/train_v4r" --detector "$DET" --run-dir "$PARAMS_RUN"
  [[ -f $PARAMS_RUN/bundle/bundle.json ]] || fail "decoder: no bundle in $PARAMS_RUN"
  log "decoder: done"
}

stage_recal() {
  if [[ -f $PARAMS/bundle.json ]]; then log "recal: $PARAMS exists, skipped"; return; fi
  local n
  n=$("$py" - "$PARAMS_RUN" "$OCTAVE_TEMPLATE" <<'EOF'
import json, sys
from pathlib import Path
import pandas as pd
run, template = Path(sys.argv[1]), int(sys.argv[2])
samples = Path(json.loads((run / "bundle/bundle.json").read_text())["training"]["samples_path"])
val = pd.read_parquet(run / "val_predictions.parquet", columns=["row"])
t = pd.read_parquet(samples / "samples.parquet", columns=["row", "template_id"])
print(int((val.merge(t, on="row")["template_id"] == template).sum()))
EOF
)
  mkdir -p "$(dirname "$PARAMS")"
  if (( n > 0 )); then
    log "recal: $n validation boxes of template $OCTAVE_TEMPLATE; refitting the calibration without them"
    guarded recal 8G 0 "$LOGS/recal.log" \
      "$cd_bin" train params-recalibrate "$PARAMS_RUN" --out "$PARAMS" \
      --exclude-template "$OCTAVE_TEMPLATE" \
      --reason "octave-error chirp-chamber template (zero-crossing trace at 2x the baseline EODf); as the v2 decoder's recal1, 2026-11-27"
  else
    log "recal: no validation box of template $OCTAVE_TEMPLATE; calibration unaffected, bundle copied"
    cp -a "$PARAMS_RUN/bundle" "$PARAMS"
  fi
  [[ -f $PARAMS/bundle.json ]] || fail "recal: no bundle at $PARAMS"
}

stage_decodereval() {
  if [[ -f $PARAMS/eval/predictions.parquet ]]; then log "decodereval: done before, skipped"; return; fi
  log "decodereval: simulated test chirps of unseen fish"
  guarded decodereval 24G 0.25 "$LOGS/decodereval.log" \
    "$cd_bin" evaluate params "$PARAMS" --scenes "$CHIRP_DATA/scenes/train_v4r"
  [[ -f $PARAMS/eval/predictions.parquet ]] || fail "decodereval: no $PARAMS/eval/predictions.parquet"
  log "decodereval: done ($PARAMS/eval)"
}

stage_production() {
  [[ -f $PARAMS/bundle.json ]] || fail "production: decoder bundle $PARAMS missing"
  local done_
  done_=$( [[ -d $RUN ]] && n_complete "$RUN" || echo 0 )
  if (( done_ == 50 )); then log "production: 50 recordings complete, skipped"; return; fi
  log "production: $CONFIG ($done_ of 50 recordings complete so far)"
  guarded production 24G 0.4 "$LOGS/production.log" \
    "$cd_bin" run "$CONFIG" --dataset "$CHIRP_REC_2019"
  done_=$(n_complete "$RUN")
  (( done_ == 50 )) || fail "production: only $done_ of 50 recordings complete"
  log "production: done ($RUN)"
}

stage_gapfill() {
  if [[ -d ${RUN}_gapfill ]] && (( $(n_complete "${RUN}_gapfill") == 50 )); then
    log "gapfill: done before, skipped"; return
  fi
  log "gapfill: ${RUN} -> ${RUN}_gapfill"
  guarded gapfill 16G 0 "$LOGS/gapfill.log" \
    "$py" experiments/production/gapfill_reassign.py --source "$RUN" --out "${RUN}_gapfill" --overwrite
  (( $(n_complete "${RUN}_gapfill") == 50 )) || fail "gapfill: ${RUN}_gapfill incomplete"
  log "gapfill: done"
}

stage_thresholds() {
  for thr in 0.8 0.9; do
    local tag out
    tag=$(printf '_t%03d' "$("$py" -c "print(round($thr * 100))")")
    out=${RUN}_gapfill$tag
    if [[ -d $out ]] && (( $(n_complete "$out") == 50 )); then log "thresholds: $out done before, skipped"; continue; fi
    log "thresholds: $thr -> $out"
    guarded "threshold$tag" 8G 0 "$LOGS/threshold$tag.log" \
      "$py" experiments/production/make_threshold_variant.py --run "${RUN}_gapfill" \
      --source-suffix _gapfill --threshold "$thr" --verify
    (( $(n_complete "$out") == 50 )) || fail "thresholds: $out incomplete"
  done
}

stage_copy() {
  local spec run suffix dest
  for spec in ":_sw" "_gapfill:_sw_gapfill" "_gapfill_t080:_sw_gapfill_t080" "_gapfill_t090:_sw_gapfill_t090"; do
    suffix=${spec%%:*} dest=${spec#*:}
    run=$RUN$suffix   # the run directory carries the files' suffix
    log "copy: $run (files rtdetr$suffix) -> recording directories as rtdetr$dest"
    guarded "copy$dest" 4G 0 "$LOGS/copy$dest.log" \
      "$py" experiments/production/copy_detections_to_trials.py --run "$run" --suffix "$suffix" --dest-suffix "$dest"
  done
}

stage_bench() {
  local models=$BENCH/models/single_v3 report=$BENCH/report_v3
  local L=$CHIRP_REPO/experiments/benchmark
  mapfile -t scenes < <(cat "$L"/scenes_old.txt "$L"/scenes_new.txt "$L"/scenes_gap.txt | sed "s|^|$CHIRP_DATA/|")
  log "bench: new detector on ${#scenes[@]} blind-benchmark scenes"
  guarded bench-models 16G 0.3 "$LOGS/bench_models.log" \
    "$py" experiments/ablation/run_benchmark_models.py single_v3 "$DET" "$models" "${scenes[@]}"
  if [[ -f $report/metrics.json ]]; then log "bench: $report exists, scoring skipped"; return; fi
  log "bench: scoring -> $report"
  guarded bench-score 8G 0 "$LOGS/bench_score.log" \
    "$cd_bin" label-benchmark score --out "$report" \
    --set "v1_final=$BENCH/human_benchmark_v1_truth.parquet:$CHIRP_DATA/labels/human_benchmark_v1_final" \
    --set "v2_extra=$BENCH/human_benchmark_v2_extra_truth.parquet:$CHIRP_DATA/labels/human_benchmark_v2_extra" \
    --features "$BENCH/features" \
    --model "single_v3=$models" --model "dualhead=$BENCH/models/dualhead" \
    --model "yolo26s=$BENCH/models/yolo26s" --model "v1=$BENCH/models/v1:v1"
  log "bench: done ($report)"
}

stages=("$@")
(( ${#stages[@]} )) || stages=(promote decoder recal decodereval production gapfill thresholds copy bench)
[[ $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 0) == 0 ]] \
  || log "WARNING: CPU boost is on (echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost)"
log "=== switch v3: ${stages[*]}"
for s in "${stages[@]}"; do
  declare -F "stage_$s" > /dev/null || fail "unknown stage $s"
  "stage_$s"
done
log "=== switch v3 done: ${stages[*]}"
