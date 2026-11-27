# Manuscript analysis (chirp competition), rerun on the v2 detections

The analyses and figures of the manuscript on chirping in staged shelter
competitions of *Apteronotus leptorhynchus*
(`manuscript/manuscript.typ`, the manuscript repository as a submodule), ported from the
old analysis repository (`~/wrk/analyses/chirp_competition`, commit
`d040123`) so that they run front to back on the chirp detections of
pipeline v2.

- **What changed, by design:** only the chirp detections. Every ported
  script was first committed verbatim, then adapted in a separate commit
  (loader, data root, output paths, seeds where the old code had none, the
  minimum for current library versions). `git log -- paper/scripts/<name>`
  shows both; each ported file starts with a `# Ported from … / # Changes:`
  header. Models, priors, windows, kernels, trial selection, numbers of
  permutations/resamples and MCMC settings are unchanged (except the window
  around chase offsets, which since 2026-11-19 is derived from the data:
  below); the open items of the old repository's `docs/known_issues.md`
  (O2–O15) are not fixed.
- **Not changed:** the old repository and the manuscript's `figures/`
  (`paper_config.py` refuses to write there).
- **Environment:** own uv project, Python 3.14 (`pyproject.toml`,
  `uv.lock`): pymc 5.28, bambi 0.17, arviz 0.23, pandas 2.3, matplotlib
  3.10. The old analysis ran on Python 3.12 with pymc 5.22, bambi 0.15,
  arviz 0.21, so posteriors differ by sampler version as well as by
  detections. `paper_config.py` restores the `fork` start method (Python
  3.14 defaults to `forkserver`, which re-runs module-level scripts in
  pymc's workers). Plots use `text.usetex` (`style.py`), so LaTeX is needed.

## Running

```sh
paper/run_all.sh                     # everything, ~4 h (sequential)
paper/run_all.sh status coupling     # selected chains
paper/run.sh SCRIPT.py [ARGS]        # one step
make paper-test                      # loader tests (from the repository root)
```

| variable | default | meaning |
|---|---|---|
| `CHIRP_REC_2019` | `/mnt/data2/2019_tube_competition` | recordings, tracks, BORIS, LED files, `trial_summary.csv`, detections |
| `PAPER_DETECTIONS` | `rtdetr` | which `chirp_{times,ids}_<x>.npy` the analysis reads (`rtdetr` = v2, `rcnn` = v1) |
| `PAPER_OUT` | `paper/output/$PAPER_DETECTIONS` | intermediates (`data/processed/`), figures (`figures/`), logs (`logs/`), `report/` |
| `CHIRP_ROOT_PY` | `<repo>/.venv/bin/python` | root environment (chirpdetector, gridtools) for `v2_fig_extract.py` and `v2_assign_params.py` |
| `PAPER_PRIMARY` | `rtdetr_gapfill_t080` | the primary run: its figures go into the manuscript, its loser chirp-rate peak sets the window around chase offsets for every run |
| `PAPER_OFFSET_WINDOW` | unset | **testing only**: e.g. `-3,13` replaces the window around chase offsets (s); stats written with it are refused by `collect` |

`run.sh` runs a step in the paper environment with the output root as
working directory and `MPLBACKEND=Agg` (the old scripts' `plt.show()` calls
return at once), and logs to `$PAPER_OUT/logs/`. Fits and long resampling
steps run through `scripts/guarded` (RAM cap). Inputs that cannot be
regenerated (the chirp templates of the first figure, from the MSc thesis
extraction) are copied to `paper/inputs/` with their
sha256.

## Figure → scripts → intermediates

Intermediates are under `$PAPER_OUT/data/processed/`. Chains are the
`run_all.sh` names.

| manuscript label (PDF) | chain | scripts, in order | intermediates |
|---|---|---|---|
| `fig:chirp_demo` (`plot_what_is_a_chirp_and_why_use_spetrograms`) | descriptive | `collect_chirp_overview_plot_data`, plot (uses `legacy_spectrogram`) | `chirp_overview_plot_data.npz`; no competition detections |
| `fig:trial_overview` (`plot_competition_trial_overview`) | descriptive | `make_competition_overview` (new: the old writer was never committed), plot | `competition_overview_2020-06-09-10_00.json` |
| `fig:chirp_space` (`plot_chirp_distributions_manuscript`) | descriptive | plot (reads the decoder's calibrated FWHM, height and undershoot from `chirps_<detections>.parquet`; until 2026-11-17 the boxes) | — |
| `fig:trial_progressions` (`plot_event_trial_progression_correlations_all_fish`, + `_supplement` = `fig:temporal_allocation_supp`) | allocation | `analyze_event_trial_progression_trial`, `summarize_all_timing_distributions`, plot | `trial_event_progressions_all_fish/`, `agonistic_behavior_timing*.json` |
| `fig:temporal_allocation_15_supp` (`plot_event_trial_progression_correlations_15_fish`) | allocation | `analyze_event_trial_progression_trial --annotated-only`, plot | `trial_event_progressions_annotated_subset/` |
| `fig:chirp_status` (`plot_chirp_winner_loser_trial_summary`) | status | `make_trial_summary` (new), `tool_pivot_trial_summary_to_long_format`, `stats_bayesian_glmm_chirps_status_v3`, plot | `trial_summary.csv`, `trial_summary_long.csv`, `stats_bayesian_glmm_chirps_status_v3/` |
| chirps within agonistic windows (no figure; `res:chirps_status`) | agonistic | `print_chirps_in_agonistic_windows` (full-trial and ±600 s shifts) | `chirps_in_agonistic_windows{,_maxshift_600s}.csv` |
| `fig:event_correlations` (`plot_event_time_rates_winner_loser_chirp`) | rates | `analyze_event_time_rates_trial` (default and `--ag-events`), plot (`_src_paper` variant prints the extrema the text quotes), `offset_window` (the window around chase offsets, from the loser peak) | `event_time_rates/*.json`, `stats/offset_window.json` |
| `fig:chirp_coupling` (`plot_chirp_coupling`) | interchase, coupling | `summarize_submissive_chirp_effect_on_dominant`, `stats_chirp_coupling`, plot | `chirp_effect_on_dominant_adaptive_window/data/`, `chirp_coupling/negbin/` |
| `fig:receiver_before_after` (`plot_receiver_before_after`, main-text Fig. 8; new 2026-11-26, label proposed; panels replaced the same day: raster, matched comparison, previous intervals) | interchase, reattack (window from rates) | `stats_reattack_hazard` (Bayesian re-attack hazard: piecewise-exponential Poisson GLMMs on the person-period table of `reattack_hazard.py`, 9 fits, 30-45 min on 4 cores (0.5: 28, 0.8: 32, 0.9: 46 with three noncentred refits); `--report-only` rebuilds the stats from the saved posteriors), `stats_reattack_matched` (the matched chirp-triggered comparison of `reattack_matched.py`, below; ~3 min), plot (A: 44 intervals sampled at random (seed 42) from the primary model's intervals, with the chase's last 5 s, loser chirps, waiting time before / after the first counted chirp and the next onset; B: fraction not yet re-attacked after the loser's first chirp after the offset vs matched moments, weighted Kaplan-Meier, bootstrap bands, the matched curve to the power 0.5 (HR 0.5) and the model's HR; C: obs.csv, the previous inter-chase interval of all chases with a previous one, with and without a loser chirp in the offset window (M2's predictor), curves averaged over competitions with weight n_chirp n_none / (n_chirp + n_none) in both groups) | `reattack_hazard/`, `reattack_matched/`, `chirp_effect_on_dominant_adaptive_window/data/` |
| `fig:receiver_matched_checks` (`plot_receiver_matched_checks`, supplement; new 2026-11-26, label proposed) | reattack | `stats_reattack_matched`, plot (A: the kernel-smoothed re-attack rate after the first chirp vs matched moments; B, C: rate and not yet re-attacked with every loser chirp between chases as a trigger; D: rate ratios in 0-30 s: the model's HR, the data, the trap (matched intervals required to stay silent) and the surrogates (shuffled chirps, simulated HR 1, 0.5, 2)) | `reattack_matched/` |
| (dropped 2026-11-26: `plot_obliterate_receiver_effect`, the windowed models; the manuscript keeps one sentence with M1/M4) | interchase, gamma, receiver (window from rates) | `stats_chirp_affects_chasing` (59 fits), plot | `…/gamma_glmm_binary/` |
| (dropped 2026-11-26: `plot_re_analysis`, random-effects check of M4) | gamma, receiver | plot | `…/gamma_glmm_binary/` |
| `fig:hurdle_model` (`plot_hurdle_decomposition`) | hurdle | `summarize_chirp_event_correlation_around_chasing`, `stats_bayesian_hurdle_glmm_chirprate_deviation_during_chasing_offsets`, plot | `chirp_event_correlations_offset_window/` |
| `fig:pipeline` (`plot_v2_pipeline_explainer`; replaced v1's `fig:chirp_assignment`) | v2figs | `v2_fig_extract snippets traces` (root env), plot | `v2_figures/` |
| `fig:performance` (`plot_v2_performance`; replaced `fig:performances_supp_full`) | v2figs | `v2_fig_extract assign-baseline`, `v2_assign_params` (simulator truth of the simulated test chirps: the true positives, and every chirp label with whether it was detected at the operating threshold; both root env), plot | `v2_figures/`, evaluation outputs and simulated scenes under `$CHIRP_DATA` |
| `fig:failure_modes` (`plot_v2_failure_modes`, Fig. S4; replaced `fig:performances_supp`) | v2figs | plot | `v2_figures/` |
| `fig:frontend_ablation` (`plot_v2_frontend_ablation`, supplement; new 2026-11-26) | v2figs | `v2_fig_extract frontends` (root env: failure-mode C's chirp train detected by the released detector and by the single-window ablation detector at the same training budget), plot (reads the benchmark's per-label table `$CHIRP_DATA/runs/ablation_frontend/scenarios/paper/bench_labels.parquet`, sha256 checked; made by `experiments/ablation_scenarios/export_for_paper.py`, report `$CHIRP_DATA/runs/ablation_frontend/scenarios/REPORT.md`) | `v2_figures/f3_dense_frontends.parquet` |
| `fig:chase_validation` (`plot_chase_validation`, supplement; new 2026-11-24) | validation | `chase_validation` (the author's re-labelled chases vs BORIS), `stats_bayesian_logreg_loser_chirping_starts_before_chasing_offset`, `chase_validation_motion` (reads the videos), `chase_validation_rates` (Fig. 6's loser rates, `chase_event_rates`, for the median-shifted and the re-labelled chases; needs chain rates), plot | `chase_validation/`, `offset_chirp_start/` |
| pipeline numbers of the Methods (no figure) | v2figs | `summarize_pipeline` (templates, scenes, expert benchmark, real-data session, production run, from their result files under `$CHIRP_DATA`) | — |
| reported posterior numbers (no figure) | echo, hurdle, posteriors | `summarize_chirp_event_correlation_around_{winner,loser}_chirps`, `stats_bayesian_logreg_echo_responses_new`, `stats_bayesian_logreg_loser_chirps_around_chasing_offset`, `make_chirps_around_chasing_obs` (reconstructed writer, audit O5; its only reader, the 10-s before/after regression, was replaced on 2026-11-24 by the chirping-starts-before-offset test of chain validation), `summarize_reported_posteriors` | `reported_posteriors_median_hdi.csv` |

The sources and every plotted number of the three new v2 figures, with
caption drafts, are in `v2_figures.md`. `compare_detection_counts.py`
writes the chirp counts per trial and status, v1 against v2 (the run's
`PAPER_DETECTIONS`), to `$PAPER_OUT/report/`. `summarize_dataset.py` (chain
`counts`) describes the dataset: trials per year and sex combination, the
recordings' durations, the annotated trials and their chases and contacts.

**Chases and contacts → recording time.** Every script that uses BORIS
events reads them through `analyze_event_time_rates_trial.load_and_convert_boris_events`.
Since 2026-11-19 it maps video frames to recording time with
`led_alignment.fit_led_alignment` (the LED map of the hazard analysis, now
shared): video and recording LED flashes are paired by the pattern of the
10 s pauses, and the map is split where video was lost; an implausible map
raises `LedAlignmentError`. Before, a line through the first and the last LED
shifted 2020-06-09 by 5-10 s and 2020-05-13 by 0-2.3 s in the first 3 h (all
other trials by at most 24 ms). `plot_led_alignment.py` writes the diagnostic
plots and a per-trial table to `paper/output/led_alignment/` (`README.txt` there).

**Reviewing the chase annotations.** `chase_review/` is a browser tool that shows
each BORIS chase of the Fig. 6 trials in the video (10 s before the onset to 10 s
after the offset, frame-exact), with the winner's and loser's chirps on a timeline,
and saves your own onset and offset next to the annotator's (never over them), to
find out how the annotator defined onset and offset. See
[`chase_review/README.md`](chase_review/README.md).

**Timing validation of the chases (chain `validation`, 2026-11-24).** The BORIS onsets and offsets
were set by another annotator; on video, the BORIS onsets fall on the peak of the fishes' motion and the
offsets on its return to rest. The author re-labelled a random sample in the review tool
(`paper/chase_review`, blind to the BORIS marks and the chirps) with a dominant-centred criterion: onset =
the dominant accelerates towards the subordinate, offset = the dominant decelerates and/or changes its
trajectory. `chase_validation.py` reads a frozen copy of those labels (`paper/inputs/chase_validation/`,
sha256 in `paper/inputs/README.md`; `CHASE_VALIDATION` overrides the place), checks the BORIS times of the
review session against `load_and_convert_boris_events`, matches the author's chases to BORIS by overlap
and reports the shifts. `stats_bayesian_logreg_loser_chirping_starts_before_chasing_offset.py` replaces the
10-s before/after regression: it asks whether loser chirping starts before the offset more or less often
than symmetric timing predicts (1 - 0.5^k for k chirps in the window, a GLMM offset term), with the BORIS,
the author's and the median-shifted BORIS offsets. `chase_validation_rates.py` computes the loser chirp rate of
Fig. 6 (B1, C1) for every chase shifted by the median shift and for the author's onsets / offsets, with the same
functions (`chase_event_rates.py`: trials, kernel, weighting, leave-out resamples; it first checks that it
reproduces Fig. 6's JSONs number for number); the validation figure shows them over Fig. 6's own curves.

**The window around chase offsets.** Several analyses count chirps from
shortly before to some seconds after each chase offset (the "offset window";
in names `offset_window`). Until 2026-11-19 every script had the constants
-3 and +13 s (`m3p13`, `FIXED_LEFT/RIGHT`), read off an earlier loser
chirp-rate curve. Now the window is one parameter, derived by a rule
(`scripts/offset_window.py`):

- *Rule:* the stretch around chase offsets in which the losers' chirp rate is
  above the null's 95 % band, i.e. the region of the strongest loser peak
  around chase offsets (`event_rates.loser.chase_offset.peak.region_s`,
  written by `plot_event_time_rates_winner_loser_chirp_src_paper`), both ends
  rounded to the nearest whole second (half-way rounds outwards). The window
  must contain the offset. The same rule on the older curve ([-3.1, +13.2] s)
  gave the old -3 to 13 s; on the primary run of 2026-11-19 ([-5.3, +10.6] s)
  it gives -5 to 11 s.
- *Source:* derived once, from the primary run (`PAPER_PRIMARY`); the
  sensitivity runs (0.5, 0.9) use the same window, so differences between
  thresholds reflect the detections, not the windows. `offset_window.py`
  (last step of chain `rates`) writes the namespace `offset_window` (window,
  width, unrounded region, the rule's numbers): in the primary run from its
  own event rates, in a sensitivity run as a copy of the primary's (it fails
  if that is missing). Every analysis reads the window with
  `offset_window.offset_window()`, which reads
  `paper/output/$PAPER_PRIMARY/stats/offset_window.json` and raises if it is
  missing or no longer matches the primary's event rates.
- *Chains that depend on it* (`WINDOW_CHAINS` in `run_all.sh`, which checks
  the window before each of them): `interchase` (obs.csv: counts in the
  window, its chase-tail and post-offset parts), `coupling` (counts and
  exposure, full-width subset), `gamma` (predictor `loser_chirped_offset_window`,
  untruncated subset, decomposition), `receiver` (window
  schematic, forest, RE figure), `reattack` (the hazard's `chirped` counts from the window's
  left edge; its placebo and Fig. 8 A use `loser_chirped_offset_window`), `hurdle` (counts in the unclipped window, the offset logistic
  regressions; the before/after regression uses 10-s windows but its input
  copies the window columns), `posteriors`. So in the primary run `rates`
  comes first (as in `ALL`); after the window changes, rerun these chains in
  every run.
- *Testing only:* `PAPER_OFFSET_WINDOW=-3,13` replaces the window without
  reading any file (to compare against the old outputs in a scratch
  `PAPER_OUT`); `write_stats` marks such stats and `collect` refuses them.
  With it, the refactored scripts reproduced the old outputs value for value
  (2026-11-19).
- *Names* (were window-specific): columns `*_fixed_m3p13` → `*_offset_window`,
  `fixed_window_{start,end,duration}` → `offset_window_*`,
  `fixed_m3p13_window_width` → `offset_window_width`; Gamma predictor
  `loser_chirped_offset_window`; directory `chirp_event_correlations_m3p13/` →
  `chirp_event_correlations_offset_window/` (and the hurdle files in it);
  stats `hurdle.n_chirps_fixed_m3p13.*` → `hurdle.n_chirps_offset_window.*`.

**The re-attack hazard (Fig. 8, chain `reattack`, 2026-11-26).** Does a loser chirp change the
hazard of the winner's next chase onset after a chase offset? The windowed Gamma GLMMs
(`stats_chirp_affects_chasing.py`) relate the next inter-chase interval to a loser chirp in the offset window,
which is clipped at the next onset (the predictor depends on the response); their complete-window check keeps
only chases whose window is complete (762 of 1957 at 0.8) and so conditions on the outcome.
`reattack_hazard.py` builds every interval from a chase offset to the next onset with the windowed models'
own loaders and filters (and checks them against obs.csv number for number); the last interval of a trial is
censored at 3 h, or ends at the onset of a chase that starts before 3 h and ends after it (2 trials; the
windowed filter drops such chases). The intervals are split into person-period rows at the baseline pieces
of time since the offset (0-2, 2-5, 5-10, 10-20, 20-50, 50-100, 100-200, 200-500, 500+ s; no interval is
shorter than 1.66 s) and wherever the chirp covariate switches: `chirped` is 1 from the loser's first chirp at
or after max(offset - 5 s, onset) on (the windowed predictor's left part, without the right-side clip).
`stats_reattack_hazard.py` fits Poisson GLMMs with `offset(log exposure)` (bambi; random intercept and
`chirped` slope per competition; M1/M3/M4-like adjustments for the log previous interval and the log chase
duration; recency (1 s, 5 s) and lag-guarded (0.5-1.5 s) sensitivity covariates; a placebo, the next chase's
windowed predictor, measured after the event: the hazard analogue of M2). The earlier frequentist
discrete-time hazard analysis (`scripts/hazard/`) shares the data layer's choices except two: it keeps an
unpaired BORIS onset as an interval of unknown chase state (2020-05-13, 59 s) where the paper's loader merges
it into the next chase, and it merges same-sender detections closer than 50 ms (one loser chirp).

**The matched chirp-triggered comparison (Fig. 8 B, supplement `fig:receiver_matched_checks`, 2026-11-26).** The
hazard model's comparison drawn as curves against time since the chirp (`reattack_matched.py`,
`stats_reattack_matched.py`). *Triggers:* the loser's first chirp after a chase offset in an interval in which the loser
had not chirped since max(offset - 5 s, onset), i.e. where `chirped` switches after the offset (at most one per
interval). *Matched moments:* the same time since the offset in every other interval of the same competition in which,
at that moment, the winner had not yet attacked again and the loser had not yet chirped; only what is known at that
moment decides (the matched interval may chirp later or end at once). A matched moment stops counting at its own loser's
first chirp (censored, as the hazard model moves that time to `chirped` = 1). Every eligible moment is used, weight
1 / n per trigger; a trigger without any is dropped and counted. *Estimators:* weighted Kaplan-Meier, the
kernel-smoothed rate (Gaussian, SD 1 s, reflected at 0; Fig. 6's 0.3 s is noise at ~280 re-attacks), events per minute
at risk in 0-1, 0-10, 0-30 s and their ratio; 1000 bootstrap resamples of the competitions. Not adjusted for chasing
density or chase duration (like the unadjusted hazard fit). *Checks:* every loser chirp as a trigger; the trap
(matched intervals required to stay silent until the re-attack: conditions on the future and keeps short intervals);
surrogates (100 each): loser chirp trains shuffled among the intervals of a competition keeping their times since
the offset, and re-attacks simulated from the pooled piecewise baseline with a planted HR of 1, 0.5, 2. Outputs:
`reattack_matched/` (`curves.npz`, `summary.json`, `triggers_*.csv` with each trigger's number of matched moments,
`surrogates.csv`, `loser_chirps.csv`); stats only with the default settings.

Reports: `REPORT_t080.md` (primary run at the operating threshold 0.8, with the 0.5 and 0.9 sensitivity runs and the published v1 values), `REPORT_gapfill_threshold.md` (0.5 vs 0.9, gap filling), `REPORT_rtdetr.md` (first v2 rerun), `REPORT_hazard_v2.md` (the event-level hazard analysis of loser chirps vs chasing, `scripts/hazard/`, v1 vs v2, and the BORIS/LED converter check; chain `hazard`, not in the default `run_all.sh`).

## Figures into the manuscript

`paper/export_figures.sh [--dry-run]` copies the figure PDFs of the primary
run (`PAPER_PRIMARY`, default `rtdetr_gapfill_t080`: gap-filled detections at
the operating threshold 0.8) into `manuscript/figures/`. It is the only way
figures enter the manuscript: the analysis scripts cannot write there
(`paper_config.py` refuses), and the sensitivity runs (`rtdetr_gapfill` at
0.5, `rtdetr_gapfill_t090`) never do. Identical files are skipped; files in
`manuscript/figures/` that are not in its list are left alone. Commit the
result in the manuscript submodule, then the submodule pointer here.

## Numbers into the manuscript

The text never copies a number. Every script that computes a number the
manuscript cites writes it with `manuscript_stats.write_stats(namespace,
values)` to `$PAPER_OUT/stats/<namespace>.json` (nested numbers; the file also
names the script, the detections and the commit of `paper/scripts`). One
script owns each namespace:

| namespace | script | contents |
|---|---|---|
| `dataset` | `summarize_dataset` | trials, draws, exclusions, years, sexes, recording hours, annotated trials, chases and contacts |
| `counts` | `compare_detection_counts` | chirps per status, v1 and the run, whole recording and first 3 h |
| `agonistic_windows`, `…_maxshift_600s` | `print_chirps_in_agonistic_windows` | coverage, shares of chirps in windows, circular-shift tests |
| `progression`, `progression_annotated_subset` | `analyze_event_trial_progression_trial` | trials with events, trials with fewer than two events |
| `allocation`, `allocation_15_fish` | `plot_event_trial_progression_correlations_{all,15}_fish` | rates, first/last-hour rates, durations and intervals |
| `status_model`, `status_figure` | `stats_bayesian_glmm_chirps_status_v3`, `plot_chirp_winner_loser_trial_summary` | model size; per-trial medians |
| `chirp_distributions` | `plot_chirp_distributions_manuscript` | the plotted decoder FWHMs, heights, undershoots; chirps outside the decoder's validated domain |
| `event_rates` | `plot_event_time_rates_winner_loser_chirp_src_paper` | peaks and troughs of the event-triggered rates (a stretch outside the null counts for the side of its extremum); `split_at_event`: the same with each stretch cut at the event time, so each side has its own extremum |
| `offset_window` | `offset_window` | the window around chase offsets (the primary's, in every run), its width, the unrounded region, the rule |
| `echo` | `stats_bayesian_logreg_echo_responses_new` | observed, chance and model echo fractions, diagnostics |
| `chase_windows` | `summarize_chirp_event_correlation_around_chasing` | chases, merged chases, overlap of the offset windows |
| `offset_logreg` | the offset logistic regression | events, observed fractions, model estimates |
| `chase_validation` | `chase_validation` | the author's re-labelling of BORIS chases: reviews, matched chases, onset / offset shifts (median, IQR, share earlier, dependence on duration and recording), segmentation |
| `offset_chirp_start` | `stats_bayesian_logreg_loser_chirping_starts_before_chasing_offset` | loser chirping starts before chasing offset vs symmetric timing: BORIS, author and shifted offsets, windows 1-10 s (replaced `before_offset_logreg`) |
| `chase_validation_figure` | `plot_chase_validation` | the numbers of the validation figure's panels |
| `hurdle`, `hurdle_decomposition` | hurdle stats script, `plot_hurdle_decomposition` | presence and intensity multipliers, contrasts; combined multipliers; rows of the positive-count part and those of them whose window no surrogate hit (null mean count set to 1) |
| `coupling` | `stats_chirp_coupling` | data anchors, overdispersion, LOO, Pareto k |
| `receiver_effect_fits`, `receiver_effect_figure` | `stats_chirp_affects_chasing`, `plot_obliterate_receiver_effect` | n and divergences per fit; window disagreement |
| `reattack_hazard` | `stats_reattack_hazard` | intervals, events, censored intervals, competitions, person-period rows, events and exposure per baseline piece and per covariate state; per fit the hazard ratio (median, 95 % HDI, P(HR < 1)), covariate slopes, group SDs, divergences, R-hat, ESS, run time |
| `reattack_matched` | `stats_reattack_matched` | settings (kernel SD, resamples, surrogates, seed); intervals, chirp states at the offset; per design (first chirps, all chirps, the trap) candidates, triggers used and dropped (no matched moment; earliest dropped time), matched moments and intervals, matched moments per trigger (median, IQR, range), trigger times, matched moments stopped by their own chirp, rate ratios and rates in 0-1, 0-10 and 0-30 s with bootstrap intervals, not yet re-attacked at 10 and 30 s; the person-period Mantel-Haenszel ratio; per surrogate scenario the median and 2.5-97.5 % of the design's and the trap's ratios and of the Mantel-Haenszel ratio |
| `receiver_before_after` | `plot_receiver_before_after` | C: chases with a previous interval with and without a loser chirp, competitions, medians of the previous interval (competition-averaged as plotted, pooled, relative to the competition median), per competition the median ratios below and above 1; A: the raster's rows by kind; the HR shown in B |
| `re_analysis` | `plot_re_analysis` | replication, grouping levels, HDI overlap |
| `posteriors` | `summarize_reported_posteriors` | median and 95 % HDI of every reported posterior |
| `v2_performance`, `v2_failure_modes`, `v2_pipeline_explainer` | the three v2 figures | the numbers of their panels and captions |
| `pipeline` | `summarize_pipeline` | templates, scenes, expert benchmark, real-data session, production run (from the pipeline's result files) |
| `chase_hazard` | `hazard/chirp_chase_hazard` | event-level hazard of chase offsets / onsets after loser chirps: GLMM, FE and RE ratios, per-dyad O/E, sensitivity, jitter, simulations, acausal control (`REPORT_hazard_v2.md`) |
| `boris_led_check` | `hazard/hz_led_check` | per trial, the paper's BORIS/LED conversion minus the repaired map of the hazard data layer |

```sh
paper/run_all.sh                                          # writes $PAPER_OUT/stats/
python3 paper/scripts/manuscript_stats.py collect rtdetr_gapfill_t080   # -> manuscript/stats.json
python3 paper/scripts/manuscript_stats.py check                         # what changed since
# the sensitivity runs of Table S1 (detection thresholds 0.5 and 0.9)
python3 paper/scripts/manuscript_stats.py collect rtdetr_gapfill --out manuscript/stats_t050.json
python3 paper/scripts/manuscript_stats.py collect rtdetr_gapfill_t090 --out manuscript/stats_t090.json
```

`collect` refuses stats written by uncommitted code in `paper/scripts`;
`stats.json` records per input file its sha256, script and commit. The
manuscript reads it with `#stat("namespace.key…", digits: …)` (`stats.typ`),
and its `make` runs `check` first and stops on any difference, so the text
cannot build on numbers that no longer match the outputs; `#stat(…, run:
"0.5")` reads a sensitivity run's file instead (Table S1). Commit the stats
files in the manuscript submodule, then the submodule pointer here. `compare_runs.py` tabulates the same stats files across the
runs (0.5, 0.8, 0.9) in `paper/output/compare/comparison.md`.


## Derived-data package

`python3 paper/scripts/build_data_package.py OUT [--dry-run]` collects every
input of the analyses except the raw recordings, the videos and the caches
computed from them (about 1.7 TB) into one folder of about 4.5 GB: per
recording the wavetracker tracks (without `sign_v`), the detections of the
three threshold runs (from the `*_recal1` runs; the trial-folder copies must be
identical or the build stops), rises, grid configuration, BORIS annotations,
LED synchronization and video frame times; the trial and fish metadata; the
re-labeled chases and the Fig. 1 extractions (`inputs/`); and the data of the
detection pipeline at their `$CHIRP_DATA` paths (`pipeline/`: chirp templates,
detection metadata, gap-filled tracks, evaluation outputs, simulated test
scenes and training chirps, dataset manifests, decoder evaluation, expert
labels). It writes `README.md` (layout and file descriptions), `MANIFEST.tsv`
(sha256 of every file), `package.json` (sources, commits, counts) and
`recordings/recordings.csv` (durations, from the detection metadata). Run it
from the main checkout (it reads the git-ignored `paper/inputs/`; elsewhere set
`PAPER_INPUTS`). It is safe to rerun when the data change: only changed files
are copied, `--prune` deletes files that are no longer part of the package,
`--verify` re-hashes the package against its manifest.

To run the analyses on the package instead of the original data:
`CHIRP_REC_2019=<package>/recordings CHIRP_DATA=<package>/pipeline
PAPER_INPUTS=<package>/inputs`. Steps that read raw recordings, videos or
spectrogram caches (example chirps of Fig. 3, the overview spectrogram, the
video motion of the chase-timing figure, the raw windows of the v2 pipeline
figures) cannot run on it; the package README lists them.
