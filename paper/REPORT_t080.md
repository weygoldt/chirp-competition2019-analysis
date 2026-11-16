# Primary run: gap-filled v2 detections at the operating threshold 0.8

2026-11-16. Full tables: `paper/output/compare/comparison.md`
(`paper/scripts/compare_runs.py`); figures of this run are in
`paper/output/rtdetr_gapfill_t080/figures/` and, exported by
`paper/export_figures.sh`, in `manuscript/figures/` (manuscript commit
`fe1747c`). Earlier comparison of 0.5 vs 0.9: `REPORT_gapfill_threshold.md`.

| run | detections | output root |
|---|---|---|
| v2 gap-filled, 0.5 (sensitivity) | `rtdetr_gapfill` | `paper/output/rtdetr_gapfill/` |
| v2 gap-filled, 0.8 (**primary**) | `rtdetr_gapfill_t080` | `paper/output/rtdetr_gapfill_t080/` |
| v2 gap-filled, 0.9 (sensitivity) | `rtdetr_gapfill_t090` | `paper/output/rtdetr_gapfill_t090/` |

## Why 0.8

Set from agreement with one expert's labels on real recordings
(`$CHIRP_DATA/labels/real_v1/report/report.md`; 2019 recordings, weighted
estimates with 95% bootstrap intervals). These labels are not ground
truth; absolute performance comes from the simulator benchmark.

| threshold | detections the expert judged chirps | expert-marked chirps the model found |
|---|---|---|
| 0.5 | 0.834 [0.729, 0.902] | 0.935 [0.852, 0.996] |
| 0.8 | 0.899 [0.799, 0.961] | 0.863 [0.739, 0.963] |
| 0.9 | 0.942 [0.828, 0.997] | 0.728 [0.598, 0.847] |

## Chirps in the 33 analysed trials, first 3 h (total; per-trial median)

| | v1 | gap-filled, 0.5 | gap-filled, 0.8 | gap-filled, 0.9 |
|---|---|---|---|---|
| winner | 5,409 (20) | 6,971 (52) | 6,237 (28) | 5,627 (9) |
| loser | 15,225 (316) | 33,426 (856) | 30,420 (776) | 21,388 (608) |
| winners without chirps | 5 | 0 | 0 | 4 |

## 1. Results that hold at every threshold

| result | 0.5 | 0.8 | 0.9 |
|---|---|---|---|
| losers chirp more than winners (count multiplier) | 7.80 [3.14, 14.27] | 10.06 [3.27, 20.55] | 10.13 [2.13, 28.01] |
| coupling: × per doubling of previous inter-chase interval | 0.941 [0.913, 0.970] | 0.940 [0.910, 0.972] | 0.924 [0.891, 0.955] |
| coupling: × per doubling of chase duration | 1.272 [1.201, 1.342] | 1.288 [1.212, 1.360] | 1.301 [1.220, 1.386] |
| coupling joint contrast | 2.73 [2.16, 3.36] | 2.84 [2.20, 3.51] | 3.18 [2.40, 4.08] |
| receiver effect, primary M4 | 1.168 [1.037, 1.297] | 1.200 [1.066, 1.337] | 1.167 [1.035, 1.299] |
| chirp → longer chase (M5) | 1.331 [1.249, 1.418] | 1.356 [1.271, 1.444] | 1.363 [1.278, 1.449] |
| hurdle presence, loser / winner (× shuffle null) | 2.03 / 0.27 | 2.25 / 0.17 | 2.14 / 0.10 |
| P(loser chirp in −3…13 s around a chase offset) | 0.618 [0.463, 0.758] | 0.581 [0.415, 0.728] | 0.452 [0.253, 0.661] |
| echo responses above chance (both directions) | p = 0.001 | p = 0.001 | p = 0.001 |

## 2. Results that depend on the threshold, with the published v1 values

| result | v1 (published) | 0.5 | 0.8 | 0.9 |
|---|---|---|---|---|
| P(loser chirps before rather than after the chase offset) | 0.458 [0.275, 0.617] | 0.711 [0.605, 0.808] | **0.673 [0.554, 0.772]** | 0.550 [0.400, 0.699] |
| echo GLMM, loser after winner | 0.370 [0.184, 0.550] | 0.309 [0.196, 0.422] | 0.259 [0.150, 0.385] | 0.158 [0.062, 0.264] |
| echo GLMM, winner after loser | 0.040 [0.014, 0.078] | 0.049 [0.028, 0.078] | 0.026 [0.012, 0.045] | 0.008 [0.002, 0.020] |
| echo observed / chance, losers | 30.3% / 19.7% | 30.6% / 22.6% | 26.0% / 18.5% | 19.9% / 13.4% |
| hurdle intensity contrast, winner/loser | 0.91 [0.65, 1.22] | 0.70 [0.55, 0.84] | **0.76 [0.58, 0.97]** | 0.92 [0.59, 1.33] |
| Mtail (chirp in the chase tail → next interval) | 0.93 [0.80, 1.07] | 0.92 [0.81, 1.03] | 0.89 [0.79, 1.01] | 0.87 [0.76, 0.99] |
| Mpost unclipped | 0.88 [0.78, 0.98] | 0.88 [0.79, 0.98] | 0.92 [0.82, 1.02] | 0.93 [0.82, 1.03] |
| loser chirp peak around contacts | +2.8 s | −0.4 s | **+2.7 s** | +2.7 s |
| loser elevation around chase offsets | [−3.1, +13.2] s | [−8.0, +13.3] s | [−5.8, +13.3] s | [−5.0, +11.9] s |
| loser chirps, first vs last hour (normalised rate) | 0.43 vs 2.16 | 1.13 vs 2.12 | **1.07 vs 2.12** | 0.76 vs 2.12 |
| winner chirps, first vs last hour (normalised rate) | | 1.15 vs 0.78 | 0.96 vs 0.44 | 0.00 vs 0.03 |
| sex multiplier (male/female chirp count) | 2.36 [0.82, 4.97] | 1.75 [0.80, 3.05] | 2.13 [0.79, 4.02] | 3.34 [0.89, 7.57] |

Bold marks where the primary run's conclusion differs from the published
v1 text or from one of its neighbours:

- **Before vs after the chase offset:** v1 found no preference (HDI
  includes 0.5). At 0.8, losers chirp more before than after the offset
  (HDI excludes 0.5), as at 0.5. At 0.9 the HDI includes 0.5 again.
- **Hurdle intensity:** at 0.8 the winner/loser contrast excludes 1
  (winners chirp less intensely once they chirp). v1 and 0.9 include 1.
- **Contact peak:** at 0.8 the loser peak is back after contact (+2.7 s,
  v1 +2.8 s). At 0.5 it sat just before contact.
- **Late-trial bias of losers:** smaller than in v1 (the first-hour rate
  is 1.07, v1 0.43). The last hour is unchanged.

## Fit diagnostics (0.8)

Six of the 61 Gamma GLMMs (`stats_chirp_affects_chasing.log`) reported 1–2
divergences after tuning. All six are crossed-random-effect sensitivity fits
(`sensitivity_loserid_winnerid_recording`: M1 full/untruncated, M5 and M5b
untruncated/matched); the primary models had none. The 0.5 run showed the same
pattern (5 warnings).

## Run notes

- Chains counts … echo ran 09:25–13:24 from the `paper-rerun` worktree.
  `descriptive` then failed: its Fig. 1 inputs had just moved into the
  repository (`paper/inputs/`), and the worktree's older code still read
  them from the data drive. `descriptive`, `allocation` and `posteriors`
  were rerun from `main` (13:25–13:40, exit 0). The analysis code is the
  same in both; `main` differs only in its input and output paths. The
  log of the first part is `logs/run_all_part1.log`.
- The three v2 pipeline figures of this run come from the 09:42 `v2figs`
  run at 0.8.
