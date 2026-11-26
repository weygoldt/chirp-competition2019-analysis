# Front-end ablation, revisited: where should the short windows help?

2026-11-25. Follow-up of the front-end ablation (`experiments/ablation/`,
report `$CHIRP_DATA/runs/ablation_frontend/report.md`), which found the two
short STFT windows (0.0512 s, 0.0128 s in image channels 1-2) of the released
dual-head detector (arm A) "comparable" to the 0.2048-s window in all three
channels (arm B). The author's hypotheses: the multi-window front end should
win (1) at emitter assignment when two fish come very close in EODf and
(2) when chirps follow each other faster than in the training data (minimum
0.2 s), where the long window smears neighbouring chirps together.

Outputs: `$CHIRP_DATA/runs/ablation_frontend/scenarios/` (report `REPORT.md`).

## What was done

1. **Re-stratification of the stored sim_v4 test results** (`s1_restratify.py`,
   no re-prediction). The ablation's ΔF strata were the closest pair of any
   two live fish in the tile; here every label is binned by the *emitter's*
   ΔF to its nearest other fish (`min_df_live_hz`).
2. **A dedicated breakdown benchmark** (`ablation_bench_v1`, 450 one-minute
   hybrid scenes from the sim_v4 test config: test-split templates of unseen
   fish, test-split real backgrounds), two sweeps (`make_configs.py`,
   `simulate.sh`):
   - (a) ΔEODf ramp-down: two simulated fish as a close pair at 0.5-20 Hz
     (9 levels x 30 scenes), no crossings or ramps, the calibrated drift kept,
     chirps as in training at 0.25 /s per fish; scored by each chirp's actual
     emitter ΔF.
   - (b) chirp-rate ramp-up: two fish 40-80 Hz apart, small chirps in
     near-periodic trains at 1000 down to 75 ms (9 levels x 20 scenes), no
     minimum interval and no responses; scored by each chirp's interval to
     the nearest chirp of the same fish. The simulator never lets one fish's
     chirp supports (~70 ms for small templates) overlap, which sets the
     fastest level (realised median 78 ms, ~13 chirps/s).
3. **Both front ends** (`bench_build.py`: arm A = the sim_v4 front end,
   arm B = the sim_v4_longonly one, checked against the training datasets'
   manifests), **both detectors on their own front end** (`bench_predict.py`:
   the simulated-test path of `experiments/evaluation/eval_postprocess.py`,
   matching within 100 ms and, for the rate sweep, within 25 ms),
   **scoring** (`bench_analyse.py`, paired scene bootstrap of B - A) and the
   **figure** (`bench_figure.py`).
4. **Training-budget control** (`prepare_longonly_cont.py`,
   `continue_longonly.sh`; started 2026-11-25 15:45 as
   `$CHIRP_DATA/runs/dualhead_r18_sim_v4_longonly_r1c`, resumed after epoch 10
   at step 40500): arm B early-stopped after 10 epochs
   (best epoch 4), arm A ran 22 (best 16). A's validation emitter error only
   fell below B's in epochs 11-16, in the low-learning-rate part of the cosine
   schedule B never reached, so any A > B difference in assignment is
   confounded with training length. The two scripts resume B from its epoch-10
   state (a copy with patience 100, nothing else changed) through epoch 22 and
   snapshot every epoch, for a budget-matched arm `Bc`.

## Round 2: trained on rapid chirping, several seeds (`rapid_training/`, 2026-11-26)

Round 1 compared detectors trained on train_v4, whose chirps of one fish are
always >= 200 ms apart, on rapid trains they never saw, with one training run
per front end and the single-window run resumed. The author's design for a fair
comparison (ablation only, not a production model): both front ends trained on
data that contains the tested regime, several seeds, no early stopping.

| file | output (`$CHIRP_DATA/...`) | role |
|---|---|---|
| `make_configs.py` | `configs/train_v4_rapid_{train,val}.toml` | train_v4's train/val configs with rapid chirp timing (0.3-1 /s per fish, bursts of ~5 at a median 110 ms, no minimum interval; the simulator still keeps one fish's chirps ~70 ms apart) |
| `prepare_scenes.sh` | `scenes/train_v4r/{train,val,test}` | train_v4 hard-linked (train_v4 is never written) + 300 train / 40 val rapid scenes (seeds 40000:40340); test = train_v4 test |
| `build_datasets.py A\|B` | `datasets/sim_v4r`, `datasets/sim_v4r_longonly` | both front ends (as sim_v4 / sim_v4_longonly, checked) |
| `dualhead_r18_v4r.toml` | | the released training config with `patience = 0` (full 30 epochs, best epoch by validation) |
| `run_round2.sh [train] [evaluate] [compare]` | `runs/ablation_frontend/round2/` | orchestration: 2 front ends x seeds 1-3, two fits at a time; each model on the breakdown benchmark, the sim test set + 335 and 365 real labels, the Fig. S4C train; resumable (DONE markers; unfinished fits retrained from scratch, not resumed) |
| `real_train_counts.py BUNDLE` | `round2/s4c/<model>.json` | chirps of the Fig. S4C train a detector finds (checks itself against Fig. S7's 109 / 62 first) |
| `compare_round2.py` | `round2/compare/{tables.md,summary.csv,per_model.csv}` | per seed, mean ± SD per front end, single minus three windows with a Welch 95 % CI over seeds; round 1's models as reference rows |

Launch (from the repository root, after both datasets exist):

```sh
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
systemd-run --user --unit=ablation-r2-$(date +%m%d-%H%M) --working-directory="$PWD" \
  -E PATH="$PATH" -E HOME="$HOME" "$PWD/experiments/ablation_scenarios/rapid_training/run_round2.sh"
tail -f /mnt/data1/chirp_v2/runs/ablation_frontend/round2/logs/round2.log
```

## In the manuscript (2026-11-26)

Supplementary Fig. S7 (`fig:frontend_ablation`, Appendix 7), the end of the
Methods paragraph "Spectrograms" and one sentence in the Fig. S4 caption, all
`#toreview` (manuscript TODO item F). The text is
`manuscript_frontend_ablation.patch` (a commit of the manuscript repository on
top of `82a1954`); the figure and numbers come from the paper pipeline:

- `export_for_paper.py`: the benchmark's per-label table for the paper module
  (`$CHIRP_DATA/runs/ablation_frontend/scenarios/paper/bench_labels.parquet`,
  sha256 in the `.json` beside it and checked by the module);
- `paper/scripts/v2_fig_extract.py frontends` (root env): the dense chirp train
  of Fig. S4C detected by the released and the single-window detector (epoch 16
  of arm Bc), on the production tile grid;
- `paper/scripts/plot_v2_frontend_ablation.py` (paper env, chain `v2figs`):
  figure and `#stat` namespace `v2_frontend_ablation`, at the operating
  threshold (true positives at 0.5 with score >= 0.8, as Fig. 10).

Integration, from the repository root (paper outputs are git-ignored, so the
two steps run where the primary run's outputs live):

```sh
cd paper
PAPER_OUT=$PWD/output/rtdetr_gapfill_t080 PAPER_DETECTIONS=rtdetr_gapfill_t080 \
    ../.venv/bin/python scripts/v2_fig_extract.py frontends
PAPER_DETECTIONS=rtdetr_gapfill_t080 ./run.sh plot_v2_frontend_ablation.py
cd ..
python3 paper/scripts/manuscript_stats.py collect rtdetr_gapfill_t080
paper/export_figures.sh
cd manuscript && git apply ../experiments/ablation_scenarios/manuscript_frontend_ablation.patch && make
```

## Reproduce

```sh
source experiments/env.sh
uv run python experiments/ablation_scenarios/s1_restratify.py
uv run python experiments/ablation_scenarios/make_configs.py
scripts/guarded --name bench-sim --mem 12G --vram 0 -- experiments/ablation_scenarios/simulate.sh 6
experiments/ablation_scenarios/bench_run.sh build A B
experiments/ablation_scenarios/bench_run.sh predict A B
uv run python experiments/ablation_scenarios/bench_analyse.py
uv run python experiments/ablation_scenarios/bench_figure.py
# budget-matched arm B (GPU, ~2.2 h), then add it to the comparison:
uv run python experiments/ablation_scenarios/prepare_longonly_cont.py
systemd-run --user --unit=ablation-cont-$(date +%H%M%S) --working-directory="$PWD" \
    -E PATH="$PATH" -E HOME="$HOME" "$PWD/experiments/ablation_scenarios/continue_longonly.sh"
# (systemd-run needs the script's absolute path)
```
