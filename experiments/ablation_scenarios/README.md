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
