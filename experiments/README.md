# Experiments

The scripts and run configs that produced the results of the chirp pipeline
v2, copied from the data directory where they were run
(`/mnt/data1/chirp_v2/{runs/scripts,runs/configs,params/configs,labels/benchmark/scripts,labels/candidates,runs/reattribution_report/scripts,templates,detections}`).
They are kept as they were run, except that machine-specific paths now come
from one place:

- Python scripts import `paths.py` (`DATA`, `FAST`, `REC_2019`, `REC_2022`,
  `V1_BENCHMARK`, `REPO`);
- shell scripts source `env.sh` (the same as `CHIRP_*` variables, plus
  `CHIRP_REPO`).

| variable | default (the machine the results were made on) | contents |
|---|---|---|
| `CHIRP_DATA` | `/mnt/data1/chirp_v2` | libraries, scenes, datasets, runs, bundles, labels, detections |
| `CHIRP_FAST` | `~/chirp_v2_fast` | fast-disk copy of the detector datasets (`datasets/sim_v4`) |
| `CHIRP_REC_2019` | `/mnt/data2/2019_tube_competition` | 2019/2020 tube competition recordings and tracks |
| `CHIRP_REC_2022` | `/mnt/data2/2022_tube_competition` | 2022 recordings (`raw/`) and tracks (`interrim/`) |
| `CHIRP_V1_BENCHMARK` | `~/wrk/analyses/chirp_competition/detector_benchmark` | frozen v1 detector benchmark (v1 boxes on the labelled regions) |

Run everything from the repository root with the workspace environment
(`uv run ...`), heavy jobs through `scripts/guarded` (RAM cap in a systemd
scope, `CHIRP_MAX_VRAM_FRACTION`). The scripts are not linted (`ruff`
excludes `experiments/`). Paths inside the run configs (`configs/`,
`experiments/training/*.toml`) are still absolute; they name the files
exactly as they were used.

What was not copied: the background screens' run scripts beyond
`scripts/v1_screen.py` / `scripts/v2_screen.py` (in `scripts/`), transient
helpers under `/tmp`, a superseded variant of `train_dualhead.sh` that ran
from a worktree, `train_when_ready.sh` (the sim_v1 YOLO baseline, superseded
by decision 20), and a cancelled config (`dualhead_r18_sim_v3.toml.cancelled`).

## Map: script -> output -> result

### Template library (`templates/`)

| script | output (`$CHIRP_DATA/...`) | result |
|---|---|---|
| `run_harvest.py OUT` | `templates/long_chirps/` (`candidates.csv`, `recordings.csv`, `harvest/`) | 533 long chirps harvested from the 2019/2020 recordings |
| `make_sheets.py` | `templates/long_chirps/contact_sheets/`, `rejections.png` | review sheets; vetoes go to `review.csv` (none vetoed) |
| `build_v3.py`, `report_v3.py` | `templates/v3/` | v2 + harvested long chirps, 1.2-s templates |
| `build_v4.py` | `templates/v4/` (`report.md`) | final library: 13,548 templates, 12,692 pass the default QC exclusions |
| `build_splits_v1.py [LIB]` | `templates/v4/splits_v1.{json,md}` | fish-disjoint splits: train 625 units / 11,230 templates, val 78 / 1,124, test 78 / 1,194 |

The base library v2 has no script: `gridtools.templates.build_library(H5,
$CHIRP_DATA/templates/v2, BuildConfig())` on the MSc thesis' extraction
`extracted_chirps.h5` (sha256 `d2e13b9e8ff7…`, in
`~/wrk/analyses/mscthesis_chirpdetection/data/interrim/new_chirpextractions/`);
`templates/v2/manifest.json` records the config.

### Training (`training/`)

| file | run (`$CHIRP_DATA/runs/...`) | role |
|---|---|---|
| `dualhead_r18_sim_v4.toml` | `dualhead_r18_sim_v4_r1` | **the released detector** (best epoch 16 of 22) |
| `yolo26s_sim_v4.toml` | `yolo26s_sim_v4_r1` | box-only reference |
| `train_dualhead.sh CONFIG DATASET RUN [VRAM]` | logs in `runs/dualhead_logs/` | how the dual-head runs were started |
| `dualhead_r18_sim_v1*.toml`, `yolo26s_sim_v{1,2,3}.toml` | `dualhead_r18_sim_v1_r{1,3}`, `yolo26s_sim_v{1,2,3}_r*` | earlier training sets (synthetic noise, v1 backgrounds); kept for reference only (decision 20) |

`yolo26s_sim_v2_r2` (stopped run, bundled by `scripts/v2_screen.py
bundle`) is the second chirp screen of background library v2.

### Evaluation (`evaluation/`)

| script | output | result |
|---|---|---|
| `evaluate_dualhead.py BUNDLE --sim DATASET --out RUN/eval` | `runs/dualhead_r18_sim_v4_r1/eval/` | assignment on sim_v4 test: 0.914 correct vs 0.648 for the box-edge baseline; emitter error per ΔF bin |
| `evaluate_bundle.py BUNDLE --sim DATASET --out RUN/eval` | `runs/yolo26s_sim_v4_r1/eval/` | YOLO26s on sim_v4 test and the real labels |
| `eval_postprocess.py RUN --sim DATASET` | `runs/*/eval_postprocess/` | post-processing before/after on sim_v4 test (box-midpoint matching) and the 335 real labels; the oracle check; flagged predictions reused below |
| `eval_reattribute.py RUN` | `runs/*/eval_postprocess/reattribute/` | re-attribution vs flag-only per threshold |
| `make_real_unmatched_v4.py` | `labels/candidates/real_unmatched_v4.parquet` | the 58 unmatched boxes the author labelled (`labels/emitters`) |
| `eval_real_labels.sh [OUT]` | `eval/real_labels_365/` | detection on the 365 real labels (335 + 30 confirmed) |
| `make_report.py RUN --eval EVAL`, `plot_dualhead_curves.py` | `RUN/report.md`, figures | training reports |
| `postprocess/explore_sim.py`, `postprocess/oracle_only.py` | exploration tables, `oracle_true_boxes.json` | the measurements behind the post-processing defaults |

`eval/` of the dual-head run predates the box-midpoint matching fix; its
detection counts differ from `eval_postprocess/`, which the detection
results cite.

### Decoder (`decoder/`)

| file | run (`$CHIRP_DATA/params/runs/...`) | role |
|---|---|---|
| `params_v1.toml` | `v4-dualhead-sim-v4-r2` | **the released decoder** (on the dual-head's boxes; also `v4-dualhead-sim-v4`, before the matching fix, and `v4-yolo26s-sim-v4`) |
| `params_footprint.toml` | `v2hybrid-footprint` | reference decoder on footprint boxes (sim_v2) |

### Human benchmark (`benchmark/`)

| script | output (`$CHIRP_DATA/labels/benchmark/...`) | role |
|---|---|---|
| `run_dualhead_sim_v4_test.py` | `model_dualhead_sim_v4_test/` | dual-head detections on the 300 sim_v4 test scenes (input of `label-benchmark build`, v1 set) |
| `run_models_v2.py MODEL SCENE...` | `models/{dualhead,yolo26s}/` | v2 models on the benchmark scenes, `keep_dropped=True` |
| `run_v1_scenes.py` | `models/v1/` | the frozen v1 detector on scenes (v1 environment) |
| `chain_v2_models.sh`, `chain_gaps.sh` | logs in `$CHIRP_DATA/logs/` | the order the model runs were chained in |
| `scenes_{old,new,gap}.txt` | | the scene lists those chains used (relative to `$CHIRP_DATA`) |

### Re-attribution report (`reattribution/`)

`analyse.py` (tables, `summary.json`), `gallery.py` (real panels,
`gallery_contact_sheet.png`), `upper_bound_cases.py`: everything in
`$CHIRP_DATA/runs/reattribution_report/` (`report.md`).

### Production (`production/`)

`launch_competition2019_v2.sh`: the full run over the 2019/2020
competition recordings with `configs/detect/competition2019_v2.toml`
(outputs `$CHIRP_DATA/detections/competition2019_v2_20261113/`).

`copy_detections_to_trials.py`: copies each recording's
`chirps_rtdetr.parquet`, `chirp_{times,ids}_rtdetr.npy` and `meta.json` (as
`chirps_rtdetr_meta.json`) from that run into `$CHIRP_REC_2019/<recording>/`
(done 2026-11-13 for all 50 recordings). Never overwrites: identical files
are skipped, different ones stop the script. Verifies sha256 and the counts
in `meta.json`; manifest `<run>/trial_copies_manifest.csv`. `--run` and
`--suffix` copy another run's files, e.g. the gap-filled patch below with
`--run $CHIRP_DATA/detections/competition2019_v2_20261113_gapfill --suffix
_gapfill` (`chirps_rtdetr_gapfill.parquet`, `chirp_{times,ids}_rtdetr_gapfill.npy`,
`chirps_rtdetr_gapfill_meta.json`).

`gapfill_reassign.py`: dataset patch of that run (2026-11-14; chirpdetector
unchanged). Fills the interior gaps (10-300 s, EODf jump <= 15 Hz) of the two
trial fish's wavetracker tracks by linear interpolation on the tracker's
time grid and reruns chirpdetector's post-processing and assignment on the
stored detections (no re-detection). An equivalence test first reproduces
every stored table and legacy array from the unpatched tracks. Outputs
`$CHIRP_DATA/detections/competition2019_v2_20261113_gapfill/` (`README.md`,
`equivalence.csv`, `gaps_gapfill.csv`, `transitions.csv`,
`overview_trials.csv`; per recording `chirps_rtdetr_gapfill.parquet` with
`gapfill_changed`, legacy arrays, `meta.json`, `tracks_gapfill.npz`).

`make_threshold_variant.py`: the gap-filled run at a stricter operating
threshold, for the robustness check of the manuscript analyses (2026-11-14,
author's decision: primary threshold stays 0.5). Only `chirp` changes
(`score >= threshold & kept`); post-processing and assignment are reused;
`--verify` first reproduces the source's legacy arrays at its own threshold.
At 0.9: `$CHIRP_DATA/detections/competition2019_v2_20261113_gapfill_t090/`
(120,808 -> 86,990 legacy chirps in the 50 recordings), copied with
`--suffix _gapfill_t090`.

### README figures (`readme_figures/`)

The overview figures of the two package READMEs, each in a light and a
dark variant (`style.py`: Inter, crest colours, transparent background).
`chirpdetector_figure.py` reads one window of the full 2019 run and its
raw recording; `gridtools_figure.py` reads a scene simulated from
`readme_scene.toml` (command in its docstring). Outputs go to
`$CHIRP_DATA/readme_figures/out/` and are copied by hand to
`packages/*/assets/overview-{light,dark}.png`; `preview.py` puts them on
GitHub's page backgrounds for checking.

## Reproducing everything

In order; every step names the result it produced. Commands for steps
without a script are given as they were run (from logs and manifests).

```sh
source experiments/env.sh

# 1. Calibration of the simulator (calibration/v1)
uv run python -m gridtools.calibrate $CHIRP_REC_2019 $CHIRP_DATA/calibration/v1 --workers 8 --seed 0

# 2. Template library
uv run python -c "from gridtools.templates import BuildConfig, build_library; \
  build_library('extracted_chirps.h5', '$CHIRP_DATA/templates/v2', BuildConfig())"
uv run python experiments/templates/run_harvest.py $CHIRP_DATA/templates/long_chirps
uv run python experiments/templates/make_sheets.py      # then review.csv by hand
uv run python experiments/templates/build_v3.py
uv run python experiments/templates/build_v4.py
uv run python experiments/templates/build_splits_v1.py  # writes splits_v1.json once

# 3. Background libraries
uv run python -m gridtools.background build $CHIRP_DATA/background/v1 \
    --datasets competition2019 competition2022 field_iriri2026 field_flona2025 field_flona2023 \
    --root-2019 $CHIRP_REC_2019 --root-2022 $CHIRP_REC_2022 --root-iriri2026 $CHIRP_FIELD_IRIRI2026 \
    --root-flona2025 $CHIRP_FIELD_FLONA2025 --root-flona2023 $CHIRP_FIELD_FLONA2023 --workers 6 --seed 0
#    chirp screens: scripts/v1_screen.py (frozen v1 environment, see its docstring)
#    -> $CHIRP_DATA/background/screens/v1; scripts/v2_screen.py bundle + screen -> screens/v2_yolo26s
uv run python -m gridtools.background carry-over $CHIRP_DATA/background/v1 $CHIRP_DATA/background/v2 \
    --datasets field_flona2025 field_iriri2026
uv run python -m gridtools.background build $CHIRP_DATA/background/v2 \
    --config configs/background/v2.toml --datasets competition2019 competition2022 \
    --root-2019 $CHIRP_REC_2019 --root-2022 $CHIRP_REC_2022

# 4. Training scenes train_v4 (3,000 one-minute hybrid scenes)
uv run gridtools sim run configs/scenes/train_v4_train.toml --seeds 5000:7400 --out $CHIRP_DATA/scenes/train_v4/train --workers 12
uv run gridtools sim run configs/scenes/train_v4_val.toml   --seeds 7400:7700 --out $CHIRP_DATA/scenes/train_v4/val --workers 12
uv run gridtools sim run configs/scenes/train_v4_test.toml  --seeds 7700:8000 --out $CHIRP_DATA/scenes/train_v4/test --workers 12

# 5. Detector datasets (sim_v4: 81,000 tiles; real_test_v1: the labelled regions)
uv run chirpdetector data build-scenes --out $CHIRP_FAST/datasets/sim_v4 \
    --split-dir train=$CHIRP_DATA/scenes/train_v4/train --split-dir val=$CHIRP_DATA/scenes/train_v4/val \
    --split-dir test=$CHIRP_DATA/scenes/train_v4/test --workers 8 --threads 2 --device cuda
uv run chirpdetector data build-real --out $CHIRP_DATA/datasets/real_test_v1 \
    --recording $CHIRP_REC_2019/2019-12-09-10_00 --recording $CHIRP_REC_2019/2020-06-03-10_10 \
    --recording $CHIRP_REC_2022/raw/2022-05-12-10_00 --recording $CHIRP_REC_2022/raw/2022-06-01-10_00

# 6. Detectors
experiments/training/train_dualhead.sh experiments/training/dualhead_r18_sim_v4.toml \
    $CHIRP_FAST/datasets/sim_v4 $CHIRP_DATA/runs/dualhead_r18_sim_v4_r1
scripts/guarded --name train-yolo26s-sim-v4 --mem 24G --vram 0.3 -- uv run chirpdetector train detector \
    experiments/training/yolo26s_sim_v4.toml --dataset $CHIRP_FAST/datasets/sim_v4 --run-dir $CHIRP_DATA/runs/yolo26s_sim_v4_r1

# 7. Evaluation
for run in dualhead_r18_sim_v4_r1 yolo26s_sim_v4_r1; do
  scripts/guarded --name pp-eval --mem 24G --vram 0.3 -- uv run python experiments/evaluation/eval_postprocess.py \
      $CHIRP_DATA/runs/$run --sim $CHIRP_FAST/datasets/sim_v4
  scripts/guarded --name pp-reattr --mem 24G --vram 0 -- uv run python experiments/evaluation/eval_reattribute.py $CHIRP_DATA/runs/$run
done
scripts/guarded --name eval-dh --mem 24G --vram 0.3 -- uv run python experiments/evaluation/evaluate_dualhead.py \
    $CHIRP_DATA/runs/dualhead_r18_sim_v4_r1/final/bundle --sim $CHIRP_FAST/datasets/sim_v4 --out $CHIRP_DATA/runs/dualhead_r18_sim_v4_r1/eval
uv run python experiments/evaluation/make_real_unmatched_v4.py   # then label-emitters --candidates ... --out $CHIRP_DATA/labels/emitters
experiments/evaluation/eval_real_labels.sh

# 8. Decoder
scripts/guarded --name params --mem 12G --vram 0.1 -- uv run chirpdetector train params experiments/decoder/params_v1.toml \
    --scenes $CHIRP_DATA/scenes/train_v4 --detector $CHIRP_DATA/runs/dualhead_r18_sim_v4_r1/final/bundle \
    --run-dir $CHIRP_DATA/params/runs/v4-dualhead-sim-v4-r2
uv run chirpdetector evaluate params $CHIRP_DATA/params/runs/v4-dualhead-sim-v4-r2/bundle --scenes $CHIRP_DATA/scenes/train_v4

# 9. Human benchmark (labelling steps are the author's)
scripts/guarded --name hb-model --mem 16G --vram 0.3 -- uv run python experiments/benchmark/run_dualhead_sim_v4_test.py
uv run chirpdetector label-benchmark build --scenes $CHIRP_DATA/scenes/train_v4/test \
    --detections $CHIRP_DATA/labels/benchmark/model_dualhead_sim_v4_test \
    --candidates $CHIRP_DATA/labels/candidates/human_benchmark_v1.parquet \
    --truth $CHIRP_DATA/labels/benchmark/human_benchmark_v1_truth.parquet
uv run gridtools sim run configs/scenes/train_v4_test.toml --seeds 8000:8600 --out $CHIRP_DATA/scenes/benchmark_v2
#    features, gaps, model runs (run_models_v2.py, run_v1_scenes.py, the chains), build-v2, labelling, and
#    label-benchmark score with --set v1_final=...:labels/human_benchmark_v1_final
#    --set v2_extra=...:labels/human_benchmark_v2_extra --model dualhead/yolo26s/v1 -> labels/benchmark/report_v2

# 10. Re-attribution report
uv run python experiments/reattribution/analyse.py
uv run python experiments/reattribution/gallery.py
uv run python experiments/reattribution/upper_bound_cases.py

# 11. The full 2019/2020 run (resumable; dry runs: configs/detect/dryrun.toml)
experiments/production/launch_competition2019_v2.sh
```

The worker counts of steps 4 and the exact arguments of the benchmark v2
steps were not recorded; they do not change the results (scenes and
samples are seeded per scene). The labelling steps (template review, the 58
real candidates, the benchmark sessions) are human work and are not
reproducible by script; their outputs are in `$CHIRP_DATA/labels/` and
`templates/long_chirps/review.csv`.
