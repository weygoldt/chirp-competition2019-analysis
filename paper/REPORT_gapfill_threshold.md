# Gap-filled tracks and the 0.9 threshold: what changes in the manuscript's numbers

2026-11-14. Full tables: `$CHIRP_DATA/paper_rerun/compare/comparison.md`
(`paper/scripts/compare_runs.py`). Runs (all on branch `paper-rerun`):

| run | detections | output root |
|---|---|---|
| v2, 0.5 | `rtdetr` (production run) | `$CHIRP_DATA/paper_rerun/rtdetr/` |
| v2 gap-filled, 0.5 (**primary**) | `rtdetr_gapfill` (trial fish's track gaps filled, assignment rerun) | `…/rtdetr_gapfill/` |
| v2 gap-filled, 0.9 (robustness) | `rtdetr_gapfill_t090` (same, operating threshold 0.9) | `…/rtdetr_gapfill_t090/` |

On the 365 real hand labels the production model gives precision 0.938 /
recall 0.992 at 0.5 and 0.981 / 0.827 at 0.9.

## Chirps in the 33 analysed trials, first 3 h (total; per-trial median)

| | v1 | v2, 0.5 | gap-filled, 0.5 | gap-filled, 0.9 |
|---|---|---|---|---|
| winner | 5,409 (20) | 6,900 (49) | 6,971 (52) | 5,627 (9) |
| loser | 15,225 (316) | 33,011 (847) | 33,426 (856) | 21,388 (608) |
| winners without chirps | 5 | 0 | 0 | 4 |

At 0.9 the winner totals are back at v1's level; most of v2's additional
winner chirps are detections scored 0.5–0.9.

## 1. Gap filling changes nothing material

Every reported posterior moves by ≤ 0.02 on its own scale (e.g. status
multiplier 7.90 → 7.80, primary receiver-effect M4 1.154 → 1.168, coupling
joint contrast 2.74 → 2.73); hurdle, echo and allocation printouts are
equal to the second decimal.

## 2. Robust to the 0.9 threshold (same conclusion)

| result | gap-filled, 0.5 | gap-filled, 0.9 |
|---|---|---|
| losers chirp more than winners (count multiplier) | 7.80 [3.14, 14.27] | 10.13 [2.13, 28.01] (wider) |
| coupling: × per doubling of previous inter-chase interval | 0.941 [0.913, 0.970] | 0.924 [0.891, 0.955] |
| coupling: × per doubling of chase duration | 1.272 [1.201, 1.342] | 1.301 [1.220, 1.386] |
| coupling joint contrast | 2.73 [2.16, 3.36] | 3.18 [2.40, 4.08] |
| receiver effect, primary M4 | 1.168 [1.037, 1.297] | 1.167 [1.035, 1.299] |
| chirp → longer chase (M5) | 1.331 [1.249, 1.418] | 1.363 [1.278, 1.449] |
| hurdle presence, loser / winner (× shuffle null) | 2.03 / 0.27 | 2.14 / 0.10 |
| P(loser chirp in −3…13 s around a chase offset) | 0.618 [0.463, 0.758] | 0.452 [0.253, 0.661] |
| echo responses above chance (both directions) | p = 0.001 | p = 0.001 |

## 3. Sensitive to the threshold (conclusion depends on the 0.5–0.9 detections)

| result | gap-filled, 0.5 | gap-filled, 0.9 | v1 (published) |
|---|---|---|---|
| P(loser chirps before rather than after the chase offset) | 0.711 [0.605, 0.808] | **0.550 [0.400, 0.699]** | 0.458 [0.275, 0.617] |
| echo GLMM, loser after winner | 0.309 [0.196, 0.422] | **0.158 [0.062, 0.264]** | 0.370 [0.184, 0.550] |
| echo GLMM, winner after loser | 0.049 [0.028, 0.078] | 0.008 [0.002, 0.020] | 0.040 [0.014, 0.078] |
| echo observed / chance, losers | 30.6 % / 22.6 % | 19.9 % / 13.4 % | 30.3 % / 19.7 % |
| hurdle intensity contrast, winner/loser | 0.70 [0.55, 0.84] | **0.92 [0.59, 1.33]** | 0.91 [0.65, 1.22] |
| Mtail (chirp in the chase tail → next interval) | 0.92 [0.81, 1.03] | 0.87 [0.76, 0.99] | 0.93 [0.80, 1.07] |
| Mpost unclipped | 0.88 [0.79, 0.98] | 0.93 [0.82, 1.03] | 0.88 [0.78, 0.98] |
| loser chirp peak around contacts | −0.4 s | **+2.7 s** | +2.8 s |
| loser elevation around chase offsets | [−8.0, +13.3] s | [−5.0, +11.9] s | [−3.1, +13.2] s |
| loser chirps, first vs last hour (normalised rate) | 1.13 vs 2.12 | 0.76 vs 2.12 | 0.43 vs 2.16 |
| sex multiplier (male/female chirp count) | 1.75 [0.80, 3.05] | 3.34 [0.89, 7.57] | 2.36 [0.82, 4.97] |

At 0.9 several of the changes that v2 introduced relative to the published
v1 results move back toward v1 (before/after offset, the contact peak's
timing, the hurdle intensity contrast, the late-trial bias of losers).
Winner chirps at 0.9 are too sparse for the temporal-allocation analysis
(per-trial median normalised rate 0 in the first hour).

## What decides it

Whether winner- and loser-assigned detections scored 0.5–0.9 are real
chirps. The real-data labelling session `labels/real_v1` (built
2026-11-14, branch `v2-real-labels`) samples ≥ 60 winner-assigned and 30
loser-assigned candidates per bin in 0.5–0.7 and 0.7–0.9 (2019) for exactly
this; its precision per score bin and status should set the operating
threshold for the manuscript, or justify reporting both.
