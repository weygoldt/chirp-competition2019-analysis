# Manuscript rerun on the v2 detections: old vs new

2026-11-13. Branch `paper-rerun` of `chirp_competition_upgrade`, folder `paper/`.
Outputs: `paper/output/rtdetr/` (`figures/`, `data/processed/`,
`logs/`, `report/`). "Old" = the published analysis on the v1 detections
(`chirp_times_rcnn.npy`), read from the old repository's `data/processed/` and
the manuscript; "new" = the same code on the v2 detections (`chirp_times_rtdetr.npy`,
run `competition2019_v2_20261113`). Posteriors are medians with 95% HDIs
(`reported_posteriors_median_hdi.csv`, old and new) unless stated otherwise.
`manuscript.typ` was not edited.

## 1. What was done

- **Detections into the data directories.** `experiments/production/copy_detections_to_trials.py`
  copied `chirps_rtdetr.parquet`, `chirp_{times,ids}_rtdetr.npy` and
  `chirps_rtdetr_meta.json` (= the run's `meta.json`) into all 50 recording
  directories of `/mnt/data2/2019_tube_competition` (200 files, 96 MB;
  exclusive create, sha256-verified, counts checked against `meta.json`;
  manifest `…/competition2019_v2_20261113/trial_copies_manifest.csv`). Nothing in
  the recording directories was overwritten, moved or deleted.
- **Port.** All 13 figure chains, the posterior summary, and the scripts
  behind the numbers quoted in the text: verbatim copies first, adaptations
  in separate commits (see `README.md`). `run_all.sh` regenerates everything
  (~4 h).
- **All 15 figures** of the manuscript are rendered on v2 in
  `figures/` (PDF; most also PNG/SVG). The three pipeline figures are new v2
  figures (`plot_v2_pipeline_explainer`, `plot_v2_performance`,
  `plot_v2_failure_modes`; see `v2_figures.md`).

## 2. Chirp counts (33 analysed trials)

| | v1 | v2 | v2/v1 |
|---|---|---|---|
| winner chirps, whole recording | 6,795 | 8,645 | 1.27 |
| loser chirps, whole recording | 18,689 | 39,189 | 2.10 |
| winner chirps, first 3 h | 5,409 | 6,900 | 1.28 |
| loser chirps, first 3 h | 15,225 | 33,011 | 2.17 |
| per-trial median, winners / losers (whole) | 35 / 369 | 70 / 960 | |
| trials with losers > winners (first 3 h) | 28 | 28 | |
| winners without chirps in the first 3 h | 5 | 0 | |

Per-trial ratios vary widely (winners 10–90 %: 0.75–12.0×; losers 1.34–5.30×).
Per trial: `report/detection_counts.csv`. The data root's `trial_summary.csv`
(v1 counts) was not changed; the analysis reads a copy with recounted chirps
(`data/processed/trial_summary.csv`).

## 3. Figures and quoted numbers

### fig:chirp_demo (`plot_what_is_a_chirp_and_why_use_spetrograms`)
No competition detections; same templates (`chirp_overview_plot_data.npz` equal
to the old file). Only the synthetic noise realisation differs (now seeded).

### fig:trial_overview (`plot_competition_trial_overview`, 2020-06-09-10_00)
Winner / loser chirps shown: published figure 18 / 632, which are the **2023
`_cnn` detections**, not the v1 production detections (v1: 2 / 624); v2: 46 / 937.
Spectrogram, tracks, 271 chases, 11 contacts, rises unchanged.

### fig:chirp_space (`plot_chirp_distributions_manuscript`)
- n = 25,484 → **47,834**.
- **The width axis does not measure chirp duration.** v2 box widths are almost
  constant (median 244 ms, IQR 241–247 ms): the box spans the long analysis
  window's footprint, not the chirp. Panel A collapses to a vertical stripe;
  markers 1 and 3 label empty space. The manuscript's "durations ∼70 to ∼570 ms
  (median 145 ms)" were v1 box widths, which had the same problem to a lesser
  degree. The v2 decoder gives median FWHM 10.3 ms (winners) / 8.1 ms (losers).
- Heights (box top − assigned track baseline), medians W / L: 165.6 / 114.6 →
  252.5 / 114.8 Hz; troughs −66.1 / −29.9 → −61.6 / −39.4 Hz.
- Axes extended to 900 ms / 900 Hz and hexbin vmax 3000 → 7000 (74 points
  and 4 hexagons would be clipped otherwise).
- **Suggestion:** replace with the decoder's chirp-space figure
  (`/mnt/data1/chirp_v2/detections/competition2019_v2_20261113/figures/chirp_space_v2_*`).

### fig:trial_progressions (+ fig:temporal_allocation_supp)
- Winner-chirp trials 30 → **33**; supplementary panel D "n = 28 winner trials in the active phase" → **33**.
- Loser chirps, first vs last hour: 0.43 vs 2.16 → **1.13 vs 2.10** /min.
  The late bias remains (envelopes separate at 97% of time points) but is
  weaker (largest deviation from the null −0.25 → −0.16).
- Inter-chirp intervals: winners 1.0 s (0.6–3.0) → 1.0 s (0.6–6.4); losers
  1.9 s (0.8–8.8) → **1.3 s (0.6–4.5)**.
- Behaviour numbers unchanged (chasing 0.71/min, chase 5.5 s, inter-chase 30.3 s,
  contacts 0.22/min, etc.).

### fig:temporal_allocation_15_supp
- Winner trials "n = 12, 10 in the active phase" → **15, all 15**.
- **The loser late-trial bias disappears**: largest deviation −0.16 → −0.05,
  envelopes separate at 23% → 0% of time points; first vs last hour
  0.72 vs 1.54 → 1.48 vs 1.67 /min. "trended toward a late-trial bias" no longer holds.

### fig:chirp_status (`plot_chirp_winner_loser_trial_summary`)
| | old | new |
|---|---|---|
| loser/winner count multiplier | 4.54 [1.37, 10.31] | **7.90 [3.21, 14.22]** |
| male/female | 2.36 [0.82, 4.97] | 1.73 [0.78, 3.03] |
| body size per SD | 0.81 [0.47, 1.25] | 0.89 [0.59, 1.29] |
| median chirps winner / loser | 35 / 369 | 70 / 960 |

Manuscript l. ~414 ("4.5× … HDI [1.4, 10.3]") and l. ~1189 ("4.5×").

### fig:event_correlations (`plot_event_time_rates_winner_loser_chirp`)
- Winner-chirp-centred panel (A1): 28 → **33** trials (v2 finds winner chirps in
  the active phase of 2020-05-12, -05-14, -05-18, -05-19, -05-27); the caption's
  "only 28 trials versus 33" no longer holds.
- Rate axes extended (loser 10 → 15 /min, winner 4 → 6 /min).

| loser chirps around | old | new |
|---|---|---|
| chase onset, trough | −1.3 s, 1.30/min, ×0.55 | −1.5 s, 2.79/min, ×0.69 |
| chase onset, peak | +16.4 s, 4.35/min, ×1.85 | +5.2 s, 8.07/min, ×1.99 |
| chase offset, peak | +0.5 s, 6.62/min, ×2.81 | +0.6 s, 10.56/min, ×2.60 |
| chase offset, elevated region | [−3.1, 13.2] s | **[−8.0, 13.3] s** |
| contact, strongest peak | +2.8 s, 3.54/min, ×2.48 | **−0.4 s**, 6.59/min, ×2.25 |
| winner chirps around contact | −1.6 s, 3.28/min, ×2.60 | −0.8 s, 4.10/min, ×2.32 |

The offset region motivated the fixed −3…13 s window used downstream (kept).

### fig:chirp_coupling
| | old | new |
|---|---|---|
| per doubling of prev. ICI (C2) | 0.906 [0.868, 0.943] | 0.939 [0.911, 0.969] |
| per doubling of chase duration (C2) | 1.300 [1.203, 1.393] | 1.269 [1.201, 1.340] |
| prev. ICI p10 → p90 (caption "0.61") | 0.61 [0.49, 0.74] | **0.73 [0.63, 0.85]** |
| duration p10 → p90 (caption "2.14") | 2.14 [1.71, 2.62] | 2.00 [1.68, 2.32] |
| joint contrast (caption "∼3.5×") | 3.50 [2.51, 4.63] | **2.74 [2.14, 3.35]** |
| LOO C2 − C1 | 24.5 (SE 6.8) | 35.1 (SE 8.2) |
| LOO C1 − C0 | 9.7 (SE 4.9) | 7.3 (SE not printed by the script) |
| full-width subset, per doubling of prev. ICI | 0.937 [0.888, 0.984] | **0.967 [0.931, 1.003]** |
| full-width subset, per doubling of duration | 1.336 [1.193, 1.488] | 1.292 [1.193, 1.400] |

l. ~2288 ("the prev ICI multiplier … remained below 1"): the HDI now reaches 1.

### fig:winner_behavior_predicts_chirp (`plot_obliterate_receiver_effect`)
| inter-chase-interval multiplier | old | new |
|---|---|---|
| M1 causal | 1.09 [0.96, 1.21] | 1.04 [0.93, 1.15] |
| M2 acausal | 0.70 [0.62, 0.78] | 0.71 [0.64, 0.79] |
| M3 + prev. ICI | 1.16 [1.03, 1.29] | **1.10 [0.98, 1.23]** |
| M4 + prev. ICI + duration (primary) | 1.19 [1.06, 1.33] | 1.15 [1.03, 1.28] |
| M5 chirp → duration | 1.26 [1.18, 1.35] | 1.32 [1.24, 1.41] |
| M5b (adjusted) | 1.25 [1.17, 1.33] | 1.31 [1.22, 1.39] |
| Mtail | 0.93 [0.80, 1.07] | 0.91 [0.82, 1.03] |
| Mpost clipped | 1.20 [1.07, 1.35] | 1.22 [1.08, 1.35] |
| Mpost unclipped | 0.88 [0.78, 0.98] | 0.86 [0.76, 0.96] |

l. ~1114 ("a credible positive chirp slope (M3 …)"): **M3's HDI now includes 1**;
the primary M4 stays credible. The schematic's "contamination zone" label
was hard-coded (6.1% = 119/1,958 rows on v1); now computed: **7.4%** (144/1,958).

### fig:re_analysis
Crossed M4 1.18 [1.05, 1.33] → 1.15 [1.02, 1.29]; σ_recording 0.48 → 0.46,
σ_loser 0.49 → 0.48, σ_winner 0.19 → 0.18; primary σ_recording 0.63 → 0.61.
l. ~2002 "medians 1.19 and 1.18" → 1.15 and 1.15.

### fig:hurdle_model (`plot_hurdle_decomposition`)
| multiplier vs shuffle null | old | new |
|---|---|---|
| presence, loser | 1.51 [0.78, 2.42] | **2.03 [1.32, 2.80]** (HDI excludes 1) |
| presence, winner | 0.19 [0.07, 0.37] | 0.26 [0.15, 0.41] |
| intensity, loser | 1.60 [1.33, 1.86] | 1.76 [1.44, 2.08] |
| intensity, winner | 1.45 [1.00, 1.98] | 1.23 [0.91, 1.58] |
| combined, loser | 1.90 [1.40, 2.38] | 2.14 [1.74, 2.58] |
| combined, winner | 0.68 [0.38, 1.03] | **0.66 [0.46, 0.88]** (HDI no longer reaches 1) |
| presence contrast winner/loser | 0.13 [0.06, 0.21] | 0.13 [0.08, 0.19] |
| intensity contrast winner/loser | 0.91 [0.65, 1.22] | **0.70 [0.56, 0.85]** (P(winner > loser) 0.27 → 0.0003) |

Intensity cells 17 (13 L / 4 W) → 25 (15 / 10). Methods: "10% of loser chirps
fall into two or more windows" → 8.6%; "18% of consecutive offsets < 16 s" unchanged.

### Numbers in the text without a figure
| | old | new |
|---|---|---|
| echo: loser chirp ≤ 10 s after a winner chirp, observed / chance | 30.3% / 19.7% (of 5,409) | 30.0% / 22.4% (of 6,900) |
| echo GLMM, losers | 37.0% [18.4, 55.0] | **30.4% [19.8, 42.3]** |
| echo: winner after loser, observed / chance | 10.9% / 7.0% (of 15,225) | 9.2% / 8.0% (of 33,011) |
| echo GLMM, winners | 4.0% [1.4, 7.8] | 4.8% [2.8, 7.8] |
| observed / chance, losers; winners | 1.54×; 1.55× | **1.34×; 1.15×** (both p = 0.001) |
| loser/winner chirp ratio in the echo data ("2.8×") | 2.81 | 4.78 |
| P(loser chirp in −3…13 s around an offset) | 30.3% [14.2, 50.6] | **61.2% [45.7, 74.5]** |
| P(winner chirp in −3…13 s) | 1.7% [0.2, 5.3] | 4.7% [1.9, 9.2] |
| P(loser chirp before vs after offset), 10-s windows | 45.8% [27.5, 61.7] | **71.3% [61.1, 81.2]** (P > 0.5 = 0.999, n = 1,302) |

The last row uses `make_chirps_around_chasing_obs.py`, a reconstruction of the
input whose writer was never committed (audit O5); on v1 it reproduces the
old file's window columns exactly.

### New v2 pipeline figures
See `v2_figures.md` (sources, caption drafts). Two points for the text:
(1) the README's assignment accuracy (0.914, box-edge 0.648) comes from an
older evaluation (before post-processing, centre matching); the figure uses
the post-processed true positives (0.921; < 10 Hz: 0.881; box edge 0.651) —
quote one consistently. (2) On real long chirps the decoder's FWHM is a median
24.5 ms longer than the trace reference (simulation predicts 13.6 ms) and its
heights sit above the trace peaks in the explainer's panel E; unresolved.
The multi-window front end is shown as a design. Ablation (2026-11-14, branch
`v2-ablation-frontend`, `$CHIRP_DATA/runs/ablation_frontend/report.md`): against the
long window alone it is ~1 pp better in precision/F1/assignment on simulated test
scenes (F1 96.6 vs 95.8%) but has the lower AP, and is indistinguishable on the 365
real labels (F1 96.4 vs 96.5%) and the blind benchmark; one run per arm. Describe it
as comparable, not as an improvement.

## 4. Where the port differs from the old code

- Detections (by design). Environment: Python 3.14, pymc 5.28, bambi 0.17,
  arviz 0.23 (old: 3.12, 5.22, 0.15, 0.21); `fork` start method restored.
- Seeds added where the old code had none: status GLMM fit and predictions,
  status-plot jitter, event-rate analyzer, warped-chase analyzer, the
  synthetic noise of fig:chirp_demo.
- Axis limits extended only where v2 data would be clipped (fig:chirp_space,
  fig:event_correlations); one hard-coded detection-derived label computed
  (receiver-effect schematic).
- New writers for inputs whose writers were never committed, verified against
  the old files: `make_competition_overview.py` (byte-identical on the 2023 cnn
  detections), `make_chirps_around_chasing_obs.py` (identical window columns on v1).
  `make_trial_summary.py` recounts the summary's chirp columns.
- Temporal-allocation bootstrap/null bands differ from a v1 rerun by ≤ 0.023
  because the old environment consumed 7 extra `random.random()` draws and
  the chirp analyses draw detection-dependent numbers of samples first.
- Open audit items (O2–O15) not fixed.

## 5. Sampler diagnostics (new fits)

- Status, coupling (5), hurdle (2), echo (2), offset (2), before/after: 0
  divergences. Max r̂ ≤ 1.010 except the hurdle presence model (1.018, on
  `1|interaction_id_sigma`; status ≤ 1.006) and the echo winner-after-loser
  model (1.013, Intercept). Min bulk ESS ≥ 294 (echo), ≥ 573 (hurdle
  presence), ≥ 803 (coupling), 4,775 (status).
- Receiver effect (59 Gamma fits, `slope_summary.csv`): max r̂ ≤ 1.01, min
  bulk ESS ≥ 757 in all. Primary (recording) and fallback fits: 0 divergences.
  Seven fits of the crossed loser × winner × recording sensitivity structure
  have 1–8 divergences (M5 full 8, M2 untruncated 3, the others 1); the old run
  had 1–3 in five fits of the same structure.

## 6. Operational notes

- Two runs (`analyze_event_time_rates_trial.py --ag-events` and
  `stats_chirp_affects_chasing.py`) ended with a bash "unexpected EOF" in
  `run.sh` after Python had finished, because `run.sh` was edited while they
  ran; all outputs were written and checked. Don't edit `run.sh` during runs.
- `paper/run_all.sh` has not been run end to end in one go; every chain was
  run step by step with the same commands.
