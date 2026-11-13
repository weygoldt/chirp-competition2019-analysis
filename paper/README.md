# Manuscript analysis (chirp competition), rerun on the v2 detections

The analyses and figures of the manuscript on chirping in staged shelter
competitions of *Apteronotus leptorhynchus*
(`~/wrk/analyses/chirp_competition_paper/manuscript.typ`), ported from the
old analysis repository (`~/wrk/analyses/chirp_competition`, commit
`d040123`) so that they run front to back on the chirp detections of
pipeline v2.

- **What changed, by design:** only the chirp detections. Every ported
  script was first committed verbatim, then adapted in a separate commit
  (loader, data root, output paths, seeds where the old code had none, the
  minimum for current library versions). `git log -- paper/scripts/<name>`
  shows both; each ported file starts with a `# Ported from … / # Changes:`
  header. Models, priors, windows, kernels, trial selection, numbers of
  permutations/resamples and MCMC settings are unchanged; the open items of
  the old repository's `docs/known_issues.md` (O2–O15) are not fixed.
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
| `PAPER_OUT` | `$CHIRP_DATA/paper_rerun/$PAPER_DETECTIONS` | intermediates (`data/processed/`), figures (`figures/`), logs (`logs/`), `report/` |
| `CHIRP_ROOT_PY` | `<repo>/.venv/bin/python` | root environment (chirpdetector, gridtools) for `v2_fig_extract.py` |

`run.sh` runs a step in the paper environment with the output root as
working directory and `MPLBACKEND=Agg` (the old scripts' `plt.show()` calls
return at once), and logs to `$PAPER_OUT/logs/`. Fits and long resampling
steps run through `scripts/guarded` (RAM cap). Inputs that cannot be
regenerated (the chirp templates of the first figure, from the MSc thesis
extraction) are copied to `$CHIRP_DATA/paper_rerun/inputs/` with their
sha256.

## Figure → scripts → intermediates

Intermediates are under `$PAPER_OUT/data/processed/`. Chains are the
`run_all.sh` names.

| manuscript label (PDF) | chain | scripts, in order | intermediates |
|---|---|---|---|
| `fig:chirp_demo` (`plot_what_is_a_chirp_and_why_use_spetrograms`) | descriptive | `collect_chirp_overview_plot_data`, plot (uses `legacy_spectrogram`) | `chirp_overview_plot_data.npz`; no competition detections |
| `fig:trial_overview` (`plot_competition_trial_overview`) | descriptive | `make_competition_overview` (new: the old writer was never committed), plot | `competition_overview_2020-06-09-10_00.json` |
| `fig:chirp_space` (`plot_chirp_distributions_manuscript`) | descriptive | plot (reads `chirps_rtdetr.parquet` boxes) | — |
| `fig:trial_progressions` (`plot_event_trial_progression_correlations_all_fish`, + `_supplement` = `fig:temporal_allocation_supp`) | allocation | `analyze_event_trial_progression_trial`, `summarize_all_timing_distributions`, plot | `trial_event_progressions_all_fish/`, `agonistic_behavior_timing*.json` |
| `fig:temporal_allocation_15_supp` (`plot_event_trial_progression_correlations_15_fish`) | allocation | `analyze_event_trial_progression_trial --annotated-only`, plot | `trial_event_progressions_annotated_subset/` |
| `fig:chirp_status` (`plot_chirp_winner_loser_trial_summary`) | status | `make_trial_summary` (new), `tool_pivot_trial_summary_to_long_format`, `stats_bayesian_glmm_chirps_status_v3`, plot | `trial_summary.csv`, `trial_summary_long.csv`, `stats_bayesian_glmm_chirps_status_v3/` |
| `fig:event_correlations` (`plot_event_time_rates_winner_loser_chirp`) | rates | `analyze_event_time_rates_trial` (default and `--ag-events`), plot (`_src_paper` variant prints the extrema the text quotes) | `event_time_rates/*.json` |
| `fig:chirp_coupling` (`plot_chirp_coupling`) | interchase, coupling | `summarize_submissive_chirp_effect_on_dominant`, `stats_chirp_coupling`, plot | `chirp_effect_on_dominant_adaptive_window/data/`, `chirp_coupling/negbin/` |
| `fig:winner_behavior_predicts_chirp` (`plot_obliterate_receiver_effect`) | interchase, gamma, rates, receiver | `stats_chirp_affects_chasing` (59 fits), `analyze_warped_chase_chirp_rates_trial`, plot | `…/gamma_glmm_binary/`, `event_time_rates/lchirps@{agon,agoff,warped_chase}.json` |
| `fig:re_analysis` (`plot_re_analysis`) | gamma, receiver | plot | `…/gamma_glmm_binary/` |
| `fig:hurdle_model` (`plot_hurdle_decomposition`) | hurdle | `summarize_chirp_event_correlation_around_chasing`, `stats_bayesian_hurdle_glmm_chirprate_deviation_during_chasing_offsets`, plot | `chirp_event_correlations_m3p13/` |
| `fig:chirp_assignment` → **new** `plot_v2_pipeline_explainer` | v2figs | `v2_fig_extract snippets traces` (root env), plot | `v2_figures/` |
| `fig:performances_supp_full` → **new** `plot_v2_performance` | v2figs | `v2_fig_extract assign-baseline` (root env), plot | `v2_figures/`, evaluation outputs under `$CHIRP_DATA` |
| `fig:performances_supp` → **new** `plot_v2_failure_modes` | v2figs | plot | `v2_figures/` |
| reported posterior numbers (no figure) | echo, hurdle, posteriors | `summarize_chirp_event_correlation_around_{winner,loser}_chirps`, `stats_bayesian_logreg_echo_responses_new`, `stats_bayesian_logreg_loser_chirps_around_chasing_offset`, `make_chirps_around_chasing_obs` (reconstructed writer, audit O5), `stats_bayesian_logreg_loser_chirps_before_vs_after_chasing_offset`, `summarize_reported_posteriors` | `reported_posteriors_median_hdi.csv` |

The sources and every plotted number of the three new v2 figures, with
caption drafts, are in `v2_figures.md`. `compare_detection_counts.py`
writes the chirp counts per trial and status, v1 against v2, to
`$PAPER_OUT/report/`.
