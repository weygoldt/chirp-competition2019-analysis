# chirp_competition_upgrade

Version 2 of the chirp pipeline of the chirp competition paper: simulate
grid recordings of wave-type electric fish with exact ground truth,
train and evaluate a chirp detector that also predicts each chirp's
emitter, measure every chirp's height and duration, and re-detect the
chirps of the staged competitions. The pipeline is finished; see
[docs/status.md](docs/status.md) for what is final and what is open.

## The parts

```text
packages/gridtools        simulator, chirp template and background libraries, recording I/O
  |  scenes: real chirp templates on simulated fish, on real background recordings,
  |  with footprint boxes and the per-EOD-period ground truth
  v
packages/chirpdetector    datasets, detector (dual-head RT-DETRv2: box + emitter EODf),
  |                       chirp-parameter decoder, evaluation, labeller, production pipeline
  v
chirpdetector run         the competition recordings -> one chirp table per recording,
                          legacy chirp_times / chirp_ids arrays for the paper's analysis code
```

| path | contents |
|---|---|
| `packages/gridtools` | [gridtools](https://github.com/weygoldt/gridtools) (public), git submodule on `main` |
| `packages/chirpdetector` | [chirpdetector](https://github.com/weygoldt/chirpdetector) (public), git submodule on `main` |
| `configs/` | the configs of the final data and runs: `scenes/` (training scenes), `background/` (background library v2), `detect/` (production and dry runs), `showcase/` |
| `experiments/` | the scripts and run configs behind every result, with a map from script to result ([experiments/README.md](experiments/README.md)) |
| `scripts/` | `guarded` (RAM/VRAM-capped jobs), `oom_watchdog.py`, `v1_screen.py` and `v2_screen.py` (chirp screens of the background library) |
| `docs/` | `status.md`, `plan.md` (decisions 1-24), `audit-2026-11.md` (what was wrong with v1) |

## Documentation

- [chirpdetector documentation](https://github.com/weygoldt/chirpdetector/blob/main/docs/README.md)
  (`packages/chirpdetector/docs/` in this checkout)
- [gridtools documentation](https://github.com/weygoldt/gridtools/blob/main/docs/README.md)
  (`packages/gridtools/docs/` in this checkout)
- this project: [docs/status.md](docs/status.md) (what is final, decisions,
  open items), [experiments/README.md](experiments/README.md) (every script
  and how to reproduce each result), [docs/plan.md](docs/plan.md) and
  [docs/audit-2026-11.md](docs/audit-2026-11.md)

## Install

```sh
git clone --recurse-submodules git@github.com:weygoldt/chirp_competition_upgrade.git
cd chirp_competition_upgrade
uv sync              # Python 3.14; both packages editable, with chirpdetector's dualhead and label extras
make check           # ruff, ruff format --check, pyright and pytest of both packages
```

Both submodules track `main`; `git submodule update --remote` moves them to
the latest `main`. To work in a submodule, check out a branch there first
(`git -C packages/chirpdetector switch -c my-change`).

Bulk data (libraries, scenes, datasets, runs, bundles, labels, detections)
is not in git. It lives under `/mnt/data1/chirp_v2/` (`$CHIRP_DATA`), the
recordings under `/mnt/data2/` (read-only). `experiments/paths.py` and
`experiments/env.sh` hold these locations in one place; set `CHIRP_DATA`,
`CHIRP_FAST`, `CHIRP_REC_2019`, `CHIRP_REC_2022` and `CHIRP_V1_BENCHMARK`
to use other ones.

| `$CHIRP_DATA/...` | contents |
|---|---|
| `calibration/v1` | measurements of the 33 trials behind the simulator's defaults |
| `templates/v4` | chirp template library (13,548 templates; `splits_v1.json`) |
| `background/v2` | background library (1,342 one-minute segments), `background/screens/` |
| `scenes/train_v4/{train,val,test}` | the 3,000 training scenes; `scenes/benchmark_v2*` |
| `datasets/real_test_v1`, `~/chirp_v2_fast/datasets/sim_v4` | detector datasets |
| `runs/dualhead_r18_sim_v4_r1`, `runs/yolo26s_sim_v4_r1` | the final detector runs (bundle in `final/bundle`) |
| `params/runs/v4-dualhead-sim-v4-r2` | the final decoder run (bundle in `bundle/`) |
| `labels/` | the author's labels: `emitters/` (real), `human_benchmark_*`, `benchmark/` (truth, model outputs, reports) |
| `eval/real_labels_365` | detection on the 365 real labels |
| `detections/` | production and dry runs of `chirpdetector run` |
| `release/models-v1` | the released bundles (GitHub release `models-v1` of chirpdetector) |

## Reproduce

[experiments/README.md](experiments/README.md) lists every step, from the
calibration and the libraries to the scenes, datasets, trainings,
evaluations, the human benchmark, the re-attribution report and the full
2019/2020 run, with the script or command and the output it produced.
GPU jobs ran on one RTX 4080 shared with other work, through
`scripts/guarded`.

## Results

All numbers are from the reports named next to them (paths relative to
`$CHIRP_DATA`, i.e. `/mnt/data1/chirp_v2/`).

### Detection on real recordings

Four hand-labelled recordings (2019, 2020, 2022 tube competitions; 6
exhaustively labelled regions), threshold 0.5, v1 benchmark matching
(centres within 100 ms, overlapping frequency ranges, one-to-one).

| labels | detector | precision | recall | F1 | report |
|---|---|---|---|---|---|
| 365 (335 hand + 30 confirmed) | dual-head (released) | 0.938 | 0.992 | 0.964 | `eval/real_labels_365/dualhead_r18_sim_v4_r1/summary.csv` |
| 365 | YOLO26s | 0.947 | 0.984 | 0.965 | `eval/real_labels_365/yolo26s_sim_v4_r1/summary.csv` |
| 365 | v1 (paper) | 1.000 | 0.701 | 0.824 | `eval/real_labels_365/v1/summary.csv` |
| 335 (original) | dual-head | 0.868 | 1.000 | 0.929 | `runs/dualhead_r18_sim_v4_r1/eval_postprocess/summary.json` |
| 335 | YOLO26s | 0.884 | 1.000 | 0.938 | `runs/yolo26s_sim_v4_r1/eval_postprocess/summary.json` |
| 335 | v1 (paper) | 1.000 | 0.764 | 0.866 | `~/wrk/analyses/chirp_competition/detector_benchmark/results/` |

The 30 extra chirps are the author's confirmations among the final
detectors' unmatched boxes (`labels/emitters`, made by
`experiments/evaluation/make_real_unmatched_v4.py`; 35 boxes called chirps,
5 of them on already-labelled chirps). The 95 % CI of the dual-head
detector's recall on the 365 labels is 0.976-0.998.

### Detection and assignment on simulated test scenes

sim_v4 test: 300 scenes, 8,100 tiles, 9,401 chirp labels; background
recordings and chirp-template fish disjoint from training. Box-midpoint
matching, after post-processing, threshold 0.5:

| detector | TP | FP | FN | precision | recall | F1 | report |
|---|---|---|---|---|---|---|---|
| dual-head | 8,999 | 225 | 402 | 0.976 | 0.957 | 0.966 | `runs/dualhead_r18_sim_v4_r1/eval_postprocess/summary.json` |
| YOLO26s | 8,948 | 101 | 453 | 0.989 | 0.952 | 0.970 | `runs/yolo26s_sim_v4_r1/eval_postprocess/summary.json` |

Emitter assignment (nearest live track to the predicted EODf) on the
dual-head's 8,802 true positives: 0.914 correct, against 0.648 for the
nearest track to the box's lower edge plus the median offset (41 Hz); with
the two closest fish less than 10 Hz apart, 0.875 vs 0.577
(`runs/dualhead_r18_sim_v4_r1/eval/summary.json`, `sim_v4_test/emitter_by_df_bin.csv`).

### Chirp parameters

Decoder on the dual-head's boxes, 7,678 simulated test chirps of unseen fish
(score >= 0.5), calibrated estimates, median absolute error: height 9.54 Hz
(6.7 %), FWHM 1.16 ms (9.8 %), 10 % width 1.58 ms (7.0 %), undershoot
1.52 Hz (36.0 %); shrinkage slope of the height 0.933 -> 0.976 after
calibration. On 513 harvested real long chirps, median absolute difference
to an independent trace estimate: height 17.7 Hz, FWHM 24.5 ms
(`params/runs/v4-dualhead-sim-v4-r2/bundle/eval/report.md`).

### Human against model

1,010 blind items on simulated scenes, one expert labeller, models at 0.5
(`labels/benchmark/report_v2/report.md`):

| | human | dual-head | YOLO26s | v1 |
|---|---|---|---|---|
| recall on true chirps | 0.924 | 0.963 | 0.974 | 0.438 |
| false-alarm rate | 0.083 | 0.090 | 0.049 | 0.015 |
| chirp found with the right emitter | 0.812 | 0.868 | | |

The dual-head model was better than the human below -20 dB SNR (0.811 vs
0.609) and at ΔF below 10 Hz, and worse on other models' false alarms
(0.160 vs 0.073 false-alarm rate).

### Harmonic re-attribution

Re-attribution of harmonic copies whose fundamental was missed essentially
never triggers on real data (0 times in about 12 h of dry runs and the 4
labelled recordings; once per about 1,400 copies in simulation); dropping
the copies removes false positives at no recall cost
(`runs/reattribution_report/report.md`).

### Throughput

The production pipeline ran at 93-213x real time per 6-h recording on the
RTX 4080 with a 40 % memory cap (dry runs, `detections/dryrun_*/run.log`,
"done in" lines). The full 2019/2020 run
(`detections/competition2019_v2_20261113/`) finished 50 recordings with no
failure (`run.log`).
