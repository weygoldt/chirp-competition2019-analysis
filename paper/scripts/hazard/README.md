# Does subordinate chirping change the dominant's chasing?

> **Port.** Copied from `src_hazard/` of the old analysis repository
> (`chirp_competition`, branch `chirp-chase-hazard`, b65108e) and adapted only
> where it reads data and writes outputs: data root and output directory from
> `paper_config`, chirps from the v2 detections named by `PAPER_DETECTIONS`
> (`chirps_<det>.parquet`, rows with `keep`, box centre `t_center_s`, sender
> `track_id`; `rcnn` still reads the v1 npy files), the cited numbers to
> `write_stats("chase_hazard")` (`hz_stats.py`). `hz_led_check.py` is new: it
> compares the paper's BORIS/LED converter with the repaired map of `hz_data`.
> **Every number quoted in this README is from the v1 run (RCNN detections);**
> the v2 results and the comparison are in `paper/REPORT_hazard_v2.md` and in
> each run's `summary.txt`.

Event-level analysis of the 2019/2020 tube-competition trials (*Apteronotus
leptorhynchus*, dyads with one winner = dominant and one loser = subordinate).

- **(a)** Does a subordinate chirp increase the hazard that an ongoing chase ends?
- **(b)** Does a subordinate chirp decrease the hazard that a new chase starts?

Session-level chirp and chase counts are confounded by dyad (size, EODf, sex,
motivation), so everything here is conditional on chase timing: discrete-time
hazard models of chase offsets (inside chases) and onsets (outside chases) with
elapsed chase or inter-chase time as the baseline, and within-trial permutation
nulls. The result is an **association conditional on chase timing**, not a
causal effect (see Open issues 1). The answers, with all numbers, are in
`<out>/summary.txt`; this README describes the method, assumptions and data
problems.

Units are seconds and Hz throughout.

## How to run

In the paper environment (`paper/pyproject.toml`; see `paper/README.md`), with
the detections in `PAPER_DETECTIONS` (the data root is the `CHIRP_REC_2019`
default, `/mnt/data2/2019_tube_competition`, and may be given as the
positional argument). From the repository root:

```sh
export PAPER_DETECTIONS=rtdetr_gapfill_t080

# full run, default settings (v1 took ~2.5 h: steps 1-5 ~50 min with 8 workers,
# step 4b ~20 min, step 5b ~80 min with 4 workers; see timings.csv)
paper/run.sh hazard/chirp_chase_hazard.py

# smoke test: 20 jitters per null, 8 simulation replicates, 4 acausal-simulation
# datasets per account; not for inference (writes no stats)
paper/run.sh hazard/chirp_chase_hazard.py --quick --out /tmp/quick_out

# acausal control alone, into an existing output directory, then rebuild the report
OUT=paper/output/$PAPER_DETECTIONS/data/processed/chirp_chase_hazard
paper/run.sh hazard/hz_acausal.py --out "$PWD/$OUT"
paper/run.sh hazard/chirp_chase_hazard.py --report-only

# the BORIS/LED converter check (paper converter vs hz_data's map)
paper/run.sh hazard/hz_led_check.py
```

The data root is only read, never written. The default output directory is
`$PAPER_OUT/data/processed/chirp_chase_hazard` (`paper_config.PROCESSED`).
A full run (not `--quick`, also `--report-only`) writes the cited numbers to
`$PAPER_OUT/stats/chase_hazard.json`; `hz_led_check.py` writes
`boris_led_check.json`.

| flag | default | meaning |
|---|---|---|
| `data_root` (positional, optional) | `CHIRP_REC_2019` | dataset root with `trial_summary.csv` and one folder per recording |
| `--out` | `$PAPER_OUT/data/processed/chirp_chase_hazard` | output directory |
| `--bin-width` | 0.05 | hazard bin width (s) |
| `--hist-max` | 2.0 | extent of the raised-cosine chirp-history basis (s) |
| `--n-basis` | 5 | number of raised-cosine bumps |
| `--n-jitter` | 200 | null replicates for the descriptives and the jitter tests |
| `--n-sims` | 50 | replicates of the main simulation scenarios (others scale with it) |
| `--max-time` | 10800 | analysis window end (s); `0` = whole recording |
| `--use-rises` | off | add dominant and subordinate rise-history covariates |
| `--include-bad` | off | keep the four "EODfs too close" recordings |
| `--include-draws` | off | keep draws |
| `--seed` | 42 | master seed (all resampling and simulation is deterministic) |
| `--workers` | 8 | forked workers for jitter tests and simulation |
| `--nulls` | `shift,matched` | jitter-test nulls (`''` = none; the verdict then says "jitter test not run") |
| `--no-vb` | off | skip the mean-field VB cross-check |
| `--skip-descriptives`, `--skip-sim` | off | skip those steps (`--skip-sim` also skips step 5b) |
| `--skip-acausal` | off | skip the acausal control (step 4b, `hz_acausal.py`) and its simulation calibration (step 5b, `hz_acausal_sim.py`) |
| `--n-boot` | 500 | dyad bootstrap resamples of the acausal mirror contrast |
| `--quick` | off | caps: 50 descriptive nulls, 20 jitters per null (min p 0.095), 8 simulation replicates, 4 acausal-simulation datasets per account, no VB; `config.json` records the effective values under `effective` |
| `--report-only` | off | rebuild `effects.csv` and `summary.txt` from the tables in `--out`, including `acausal/` and `simulation/acausal_sim_*` when present (uses its `config.json` and its start time; no refitting; floats are rewritten bit-identically) |

Each module also runs on its own (`paper/run.sh hazard/hz_data.py`,
`hz_desc.py`, `hz_hazard.py`, `hz_sim.py`, `hz_acausal.py DATA --out DIR
[--quick] [--n-boot N]`, `hz_acausal.py --out DIR --report-only` to redraw its
figures and `acausal_summary.txt` from the saved tables, `hz_acausal_sim.py DATA
--out DIR [--quick] [--n-reps N]` and `hz_acausal_sim.py --out DIR
--report-only`; see their `--help`).

Tests (in the paper environment: `cd paper && uv run pytest scripts/hazard`;
not part of `make paper-test`. The files' fallback runner below, for an
environment without pytest, cannot run the parametrized tests):

```sh
$PY src_hazard/test_hz_hazard.py   # 23 tests: no leakage, risk sets, guard window, solvers,
                                   # --use-rises column selection, per-dyad O/E, RE meta
$PY src_hazard/test_hz_sim.py      # 12 tests: simulator validity, recovery, heterogeneous
                                   # effects, annotation lag, trigger share, calibration
$PY src_hazard/test_hz_acausal.py  # 13 tests: future window excludes the outcome bin, past
                                   # columns unchanged, window edges, recording-end drop,
                                   # mirror / restricted / shift trains, restricted real on
                                   # the mirror's support, shift comparator windows, onset
                                   # counts of the state-rate table, contrast bookkeeping
$PY src_hazard/test_hz_acausal_sim.py  # 16 tests: offset-locked chirps off by default and on
                                   # their own random stream, response latencies and rate,
                                   # cluster symmetry, bout dependence, window counts,
                                   # neighbouring-offset and 2-3 s co-occurrence pairings,
                                   # neighbour bout ratio, bursts across chase boundaries,
                                   # slow rate modulation, exceedance fractions and the
                                   # status bands (incl. borderline), verdicts and their
                                   # borderline / criterion dependence, chase-start counts
                                   # and calibration, calibration update, statistics on
                                   # synthetic trials
```

## Outputs

```
<out>/
  summary.txt            answer to (a) and (b) with caveats (also printed)
  effects.csv            every fitted effect with CI, one tidy table (columns below)
  timings.csv            wall time per pipeline step
  config.json, run.log   configuration used (incl. effective quick-mode values) and console log
  validation/            data_report.md, flags.csv (every contradiction or repair),
                         trial_summary.csv (all 43 trials, inclusion + ~100 QC columns),
                         events.csv (tidy event table: recording, group, event, sender, t, in_chase)
  descriptives/          peth_pooled, peth_per_trial, tests, tests_per_trial, state_rates, params
  hazard/                hazard_effects, hazard_dyad_oe (per-dyad observed/expected),
                         hazard_jitter_null, hazard_jitter_summary, hazard_leave_one_dyad_out,
                         hazard_sensitivity, hazard_design_summary, hazard_timings (.csv),
                         hazard_summary.txt
  acausal/               acausal control (uses chirps AFTER the outcome bin): acausal_effects,
                         acausal_dyad_oe, acausal_leave_one_dyad_out, acausal_chirp_counts
                         (per dyad and arm), acausal_latencies (per chirp), acausal_bootstrap
                         (per resample), acausal_state_rates (chirp rates by state and
                         around chase boundaries), acausal_design_summary, acausal_timings
                         (.csv), acausal_config.json, acausal_summary.txt (both acausal
                         blocks with their simulation blocks)
  simulation/            sim_summary.txt and sim_*.csv (estimates, summaries, jitter FPR,
                         kernels, fitted and calibrated parameters, calibration log,
                         data-vs-simulation match, trigger bookkeeping, timings);
                         acausal_sim_summary.txt and acausal_sim_*.csv (estimates per
                         replicate, summary per scenario, observed values with their
                         SEs, per-dyad counts, latency profiles (default and drift
                         background), drift calibration, scenarios, calibrated
                         parameters, calibration log, timings; onset_estimates,
                         onset_summary, onset_scenarios for the onset accounts) of
                         step 5b; the summary tables carry, per account and
                         statistic, the observed value's exceedance (n_below_obs,
                         n_above_obs, exceed_side, exceed_n, exceed_frac), z and p,
                         the statistic's criterion and the status by exceedance, by z
                         and by the criterion
  figures/               desc_* (alignments per null, forest plots, null comparison,
                         chirp rate by state, desc_per_trial/), hazard_kernels,
                         hazard_tau_scan, hazard_dyad_oe (per-dyad O/E forest plot),
                         hazard_jitter_null, sim_recovery, sim_fpr,
                         sim_confounds, acausal_two_sided_kernel,
                         acausal_tau_past_vs_future, acausal_mirror,
                         acausal_sim_calibration (PNG + PDF)
```

`effects.csv` columns: `analysis` (`hazard`, `sensitivity`,
`leave_one_dyad_out`, `jitter_test`, `descriptive`, `acausal`), `kind` (`offset` = chase
ends, inside chases; `onset` = chase starts, outside chases), `sender`,
`model` (`kernel`, `tau_<ms>` or `guard_500ms`), `backend` (`mixed_laplace`,
`glm_fe`, `re_meta`, `mixed_vb`), `link`, `scale`, `term`, `term_type`, `lag`
(kernel curve rows), `subset` (sensitivity set), `left_out`, `null`,
`window_lo`/`window_hi` (descriptives), `estimate`/`se`/`ci_low`/`ci_high`
(log scale), `p`, `ratio`/`ratio_low`/`ratio_high` (exp of the former),
`ci_method`, `n_bins`, `n_events`, `n_recordings`, jitter-null columns,
`converged`, `note` (e.g. how many dyads and events carry a chirp term, or
that a term is not interpretable), `variant` (acausal rows only: `two_sided`,
`real_full`, `restricted_real`, `mirror`, `shift_<ms>`,
`restricted_real-mirror`; their `subset` holds the window W as `w_<ms>`;
the restricted-real and mirror rows with `w_500ms` / `w_1000ms` are the FE-only
comparators of the shift arms; for the bootstrap rows `n_jitter` holds the
number of resamples). For the
acausal refits, `effects.csv` keeps the chirp, kernel, contrast, test and
random-effects rows; their control, spline and intercept rows are only in
`acausal/acausal_effects.csv`.

Key terms: `sub_any_<tau>ms` (>= 1 subordinate chirp in the preceding tau),
`sub_grd_500ms` (>= 1 subordinate chirp 0.5-1.5 s before the bin),
`kernel_sub` (per-chirp kernel at `lag`), `kernel_sub_mean` (kernel averaged
over 0-`hist_max`), `kernel_sub_mean_late` (over 0.5-`hist_max`),
`kernel_sub_joint` (Wald test of all basis coefficients),
`dyad_sd_sub_effect` and `heterogeneity_G` (`re_meta`: between-dyad SD of the
chirp effect and the homogeneity test), and the same with `dom_` for
dominant chirps (comparison).

**Rows not to interpret** (kept for completeness, flagged in `note`): all
dominant-chirp terms (few dyads and events), the `glm_fe` kernel joint tests
of dominant chirps (degenerate cluster-robust F), and the GLMM rows
`Intercept`, `dsize`, `deodf`, `abs_deodf`, `deodf_missing`, `sd_recording` (dyad-level terms
that trade off against the random intercepts; the GLMM posterior has several
modes along that ridge).

## What the pipeline does

1. **Load and validate** (`hz_data.py`). Chirps from `chirp_times_rcnn.npy` /
   `chirp_ids_rcnn.npy` (sender = track id), checked against the detector's
   own `chirpdetector_bboxes.csv` (Data contradictions 10, 11); for v2
   detections from `chirps_<det>.parquet` (rows with `keep`, time
   `t_center_s`, sender `track_id`), checked against the parquet's chirp boxes
   (unassigned = `chirp` without `track_id`, not dropped by post-processing); EODf tracks
   from `fund_v.npy`, `ident_v.npy`, `idx_v.npy`, `times.npy` (baseline =
   trailing 60 s rolling median, step-held on a 1 s grid so each grid point
   uses only raw samples at or before it, read 1 s before each bin); BORIS
   chase onsets/offsets and contacts converted to grid time with the LED sync
   pulses (logic of `load_boris`, `converte_video_frames_to_grid_idx`,
   `load_and_convert_boris_events` in `src/analyze_event_time_rates_trial.py`,
   plus an anchor check, see Data contradictions 4; the map,
   `fit_led_alignment`, lives in `paper/scripts/led_alignment.py` since
   2026-11-19 and the paper's BORIS loader uses it too). Every contradiction or
   repair goes to `validation/flags.csv`.
2. **Descriptives** (`hz_desc.py`). Chase offsets/onsets aligned to
   subordinate chirps, subordinate chirps aligned to chase offsets/onsets, the
   post-chirp hazard of the current chase or gap ending, and the same for
   dominant chirps. Nulls: the specified within-trial, within-state circular
   shift (keeps inter-chirp intervals), plus an elapsed-time-matched swap,
   because the shift null is biased when the chirp rate depends on time since
   the chase or gap began (simulation: >90% false positives; the real data have
   this structure). The matched null moves each chirp independently, so it
   does **not** keep inter-chirp intervals (bursts are broken up; its null SD
   is ~10% narrower, mildly anti-conservative). Dyad-bootstrap CIs; per-dyad
   tests in `tests_per_trial.csv`.
3. **Hazard models** (`hz_hazard.py`). One row per 50 ms bin in the risk set.
   Offset: bins inside a chase, event = chase ends. Onset: bins outside chases,
   event = next chase starts. Controls: natural spline of log elapsed chase /
   inter-chase time (essential: chirps accumulate in long chases), first-gap
   flag, dominant chirp history, time since last contact, size difference,
   signed and absolute dEODf (step-held trailing median read 1 s before the
   bin; before a fish's first tracked sample, 50-390 s into each trial, no past
   EODf exists, so dEODf is mean-filled and flagged by `deodf_missing`), spline
   of time in trial, dyad; rises optional.
   Chirp history: 5 log-time raised cosines over 0.05-2 s, ">= 1 chirp in the
   preceding tau" for tau in 0.25-5 s, and the **lag-guarded indicator**
   (0.5-1.5 s before the bin; leaves out chirps within the ~0.5 s precision of
   the chase boundaries). History covariates use only events in earlier bins
   (tests check that a chirp in the outcome bin is excluded).
   **Between-dyad heterogeneity**: for the 1 s and the guarded indicator,
   each dyad's observed events in indicator-on bins are compared with the
   number expected under no chirp effect (`hazard_dyad_oe.csv`), and pooled in
   a Poisson-normal random-effects meta-analysis (mean HR across dyads,
   between-dyad SD, G test of homogeneity). Jitter tests refit under both
   nulls; leave-one-dyad-out refits.
4. **Sensitivity** (main script). tau = 1 s models (GLMM, FE, RE) refitted
   without flagged dyads (close EODfs; the two LED-repaired dyads; dyads where
   most chirp boxes contain both EODfs), with only unambiguous chirps, and with
   every annotated offset moved by -0.75, -0.5, -0.25, +0.5 s and every onset by
   -0.5, +0.5 s (a uniform annotation lag).
4b. **Acausal control** (`hz_acausal.py`; see its section below). Past vs
   future chirp windows around the outcome bin, a two-sided kernel, and for
   offsets a mirror surrogate (post-offset chirps reflected into the chase)
   and a constant backward shift, each shift with the arms of its own window.
   Uses chirps after the outcome bin on purpose; never part of the primary
   models. No statistic has a null of 1; step 5b supplies the expectations.
5. **Simulation** (`hz_sim.py`). A generative model matched to each real trial
   (chase/gap durations, chirp rates by chase state and elapsed time, burst
   structure, dominant chirps, contacts), with known chirp effects on the
   offset/onset hazards; per-trial rates are recalibrated under the effect so
   effect scenarios reproduce each trial's chase and chirp counts. Reports
   recovery; false-positive rates of the FE Wald and cluster-robust tests, the
   GLMM, the random-effects meta-analysis and the jitter tests, both with one
   common effect and with dyad-specific effects (SD fitted to the data); a
   hidden common trigger that ends a chase and elicits a chirp; systematic late
   annotation of offsets (0.25-0.75 s) with no chirp effect; slow (30 s) and
   short (3 s) latent states for onsets; chase-boundary error (0.5 s steps) and
   imperfect chirp detection.
5b. **Simulation calibration of the acausal control** (`hz_acausal_sim.py`;
   see "Acausal control: simulation calibration" below). What each account of
   the offset association (causal effect with 1 s and 1.5 s lags, common or
   dyad-specific; response burst; both, also on a background with slow
   co-variation of chirping; common trigger; symmetric cluster) and of the
   onset association (no effect, dyad-specific effects, a short latent state,
   a causal effect) produces for the step-4b statistics.

## Mixed-model backend: choice and why

**Primary: logit GLMM with a random intercept per dyad** (statsmodels
`BinomialBayesMixedGLM`), fitted by a **Laplace approximation** (posterior
mode + inverse Hessian, using the class's own log posterior with an analytic
Hessian and a trust-region Newton solver), started from two initial dyad SDs
(0.5 and 3); the highest log posterior is kept and the mode dependence is
written to the `note` column. Weak priors: N(0, 5^2) on standardized fixed
effects, N(0, 2^2) on the log dyad SD.

- **R lme4 / glmmTMB are not available** (R has neither package, and nothing
  may be installed).
- **bambi / PyMC (MCMC) are not installed and would be too slow.** Designs have
  2.7e5 (offset) to 2.9e6 (onset) bins; the pipeline fits 8 models x 2 kinds x
  2 backends, and the jitter tests and the simulation refit hundreds of times.
- **Mean-field VB (statsmodels `fit_vb`) is not used as primary.** It returns
  only marginal SDs, which it tends to understate, and no covariance between
  the overlapping raised-cosine coefficients, which the kernel confidence band
  and the joint test need. It is kept as a cross-check for the offset
  tau = 1 s model.
- **No random slope.** statsmodels fits the joint mode of (effects, random
  effects, log SDs); with a random chirp slope that joint mode collapses the
  slope SD toward zero whenever few dyads carry information, so it cannot
  measure heterogeneity. The between-dyad variation of the chirp effect is
  therefore estimated by the two-stage random-effects meta-analysis
  (`re_meta`), whose calibration is checked in simulation.

The GLMM interval assumes **one chirp effect shared by all dyads**. The data
reject that assumption for both questions (heterogeneity tests in
`summary.txt`), so its interval (and its very small p) describes a pooled
within-dyad association, not an effect that holds across dyads.

**Cross-check 1: cloglog GLM with dyad fixed intercepts and cluster-robust
(dyad) SEs**, t(G-1) intervals (own IRLS, matches statsmodels GLM to 1e-6).
Cloglog makes it the grouped-time analogue of a Cox model, so it reports
hazard ratios. Per-bin event probabilities are < 1%, so OR and HR agree.
Its sandwich SE reflects between-dyad differences in the chirp effect; with
15 clusters it is still somewhat too narrow in simulation.

**Cross-check 2: random-effects meta-analysis of per-dyad observed/expected
events** (`re_meta`). Allows the effect to differ between dyads; reports the
mean HR over dyads (profile-likelihood CI with a t(k-1)^2 cut-off), the
between-dyad SD, a prediction range for a single dyad, and which informative
dyads (>= 2 expected events) lie significantly above or below 1.

**Cross-check 3: jitter (permutation) tests.** The FE model is refitted under
200 null subordinate chirp trains per null (circular shift within chase state,
and elapsed-time-matched swap). This is a within-dyad test of the pooled
association; it does not test whether dyads agree. Its calibration is judged
by the ratio of the null SD to the true sampling SD in simulation (reported in
`sim_summary.txt`); the rejection count over ~10 simulated datasets cannot
tell a 5% from a 15% false-positive rate. Smallest attainable p-value:
2 / (n_jitter + 1) = 0.01.

**Not modelled:** crossed fish random effects (10 fish recur across 15 dyads).

## Acausal control (`hz_acausal.py`, step 4b)

Around chase offsets, subordinate chirps are nearly symmetric in raw counts
(within 0.5 / 1 / 2 s: 121 / 238 / 387 in-chase chirps before the offset, 131 /
257 / 432 out-of-chase chirps after it; 0-3 h, 15 dyads). The offset
association could then be one chirp burst straddling an imprecisely annotated
chase end (a chirp response to the chase end, a common trigger, or annotation
lag) rather than something specific to chirps that precede the offset. The
acausal control asks whether the association is **direction-specific**. It
deliberately uses chirps from after the outcome bin, so it is labelled
`analysis = acausal` everywhere and never enters the primary models.

**A. Past vs future window (offset and onset).** The standard design (same
risk sets and controls, incl. the dominant past indicator) plus
future-window columns mirrored around the outcome bin, which is excluded on
both sides, as the past indicator excludes it:

| past (standard) | future (acausal) |
|---|---|
| `sub_any_<tau>`: >= 1 chirp in `[t_k - tau, t_k)` (bins k-n..k-1) | `sub_fut_<tau>`: `[t_k + bw, t_k + bw + tau)` (bins k+1..k+n) |
| `sub_grd_500ms`: `[t_k - 1.5, t_k - 0.5)` | `sub_fgd_500ms`: `[t_k + bw + 0.5, t_k + bw + 1.5)` |
| `sub_h1..5` (lag j bins before) | `sub_fh1..5` (j bins after), same raised-cosine basis |

and `dom_fut_*`, `dom_fgd_*`, `dom_fh*` as controls. Models per window: past
only (`past_<ms>`), future only (`fut_<ms>`), joint (`pf_<ms>`); the same for
the guard (`*_guard_500ms`); and a joint two-sided kernel
(`kernel_two_sided`). Backends: `glm_fe` (primary for the contrast), GLMM
(reported next to FE in the summaries; it assumes one common effect, which
the heterogeneity tests reject, caveat 4), and per-term per-dyad O/E with the
random-effects meta-analysis (1 s and guarded windows; `hz_hazard.dyad_oe` /
`heterogeneity` take a `term` and explicit `cols`; their SD / G rows are named
`dyad_sd_sub_effect:<term>`, `heterogeneity_G:<term>`). Reported: HR_past and
HR_future of the joint model (each adjusted for the other), the contrast
`log HR_past - log HR_future` (`term = sub_any_<ms>-sub_fut_<ms>`,
`term_type = contrast`, cluster-robust t(G-1) CI and p; it uses the full
covariance of the two terms), the leave-one-dyad-out range of the contrast
(`acausal_leave_one_dyad_out.csv`; 1 s, guard, kernel), and for the kernel
the mean HR per chirp before vs after the bin, their ratio, and a joint Wald
test of equal kernel shape (`kernel_sub_shape_diff_joint`). Kernel rows
`kernel_sub_fut` carry a negative signed lag. The future window uses the
full chirp train (chirps after 3 h or inside unknown-state intervals are
used). Bins whose longest future window (5 s) runs past the recording end
are dropped from the acausal designs only (`acausal_design_summary.csv`;
none in these data, since every recording runs ~6 h and the window ends at
3 h). Two dyads (05-13, 06-04) have no assigned sub chirp after 10,776 s and
10,770 s although the detector ran on; this touches only bins in the last
seconds of the window.

The tau scan of the past / future ratio (0.25-5 s) is **not calibrated**:
the simulation covers the 1 s and guarded windows only. Sub chirps stay
frequent 2-5 s after an offset (~4.0 per min, close to the in-chase mean of
4.1 and far above the 1.2 outside chases; `acausal_state_rates.csv`), so
longer after-windows do not reach a low-rate gap; risk-set bins within tau
before an offset carry the post-offset burst in their own after-window, and
how that moves the ratio under each account is unknown.

**Onsets (b): the control does not test (b).** The after-window of an onset
bin lies inside the chase that the outcome starts (first 1 s of a chase:
1.8 sub chirps per min, ~1.5x the out-of-chase rate), so the after-window
indicator is itself raised by the outcome. HR_future > 1 and a past / future
contrast below 1 then follow without any chirp effect, and HR_past of the
joint model, adjusted for that after-window, is biased toward < 1 (it
conditions on a consequence of the outcome, a collider), the direction of
(b). Neither the contrast nor the joint HR_past is used as evidence for or
against (b); the (b) estimate stays the standard past-only HR. The onset
simulation (below) quantifies the shifted null.

**B. Mirror surrogate (offset; the literal request).** For each chase i, the
sub chirps after its offset, `t` in `(off_i, min(off_i + W, next onset, next
unknown-state interval))` with latency `L = t - off_i`, are placed at
`off_i - L` and kept if `off_i - L > on_i` (inside chase i; dropped ones are
counted). Mirror train = real out-of-chase sub chirps (unchanged) + mirrored
chirps; all real in-chase sub chirps removed. Its fair counterpart
`restricted_real` = real out-of-chase sub chirps + real in-chase sub chirps
less than W before their own chase's offset, kept only where the mirror could
hold a chirp at the same latency (`off_i + L` before the next onset and the
next unknown-state interval, outside unknown-state intervals), so both arms
have the same support. W = 2 s (reported first; = `hist_max`) and 5 s. At
W = 2 s no real chirp loses its counterpart; at W = 5 s 14 do. (An
earlier version truncated only the mirror arm; that biased the W = 5 s
contrast toward 1, so the truncation cannot explain the W = 5 s guarded
contrast.) The standard offset models (tau scan, guard, kernel; FE, GLMM, RE)
are refitted on each train via `build_design(..., chirp_override=...)`, plus
`real_full` (unchanged train = the primary model) as reference. Paired
contrast `log HR(restricted_real) - log HR(mirror)` for tau = 1 s and the
guarded indicator: dyad bootstrap, 500 resamples (dyads drawn with
replacement; a dyad drawn m times enters with frequency weight m, which
equals m copies with their own intercepts; both arms refitted with FE on the
same resample; percentile CI; seed fixed).

**C. Constant backward shift (offset; secondary).** The post-offset chirps
(W = 2 s) moved to `t - Delta`, Delta = 0.5, 1, 2 s, kept if inside chase i,
replacing the real in-chase chirps as in B: what the post-offset chirps alone
would look like if every offset were annotated Delta late. A shifted train
keeps only chirps with `L < Delta` and *translates* them (to
`off_i - Delta + L`; the mirror reflects them to `off_i - L`), so each shift
arm is compared with the restricted-real and mirror arms built with
`W = Delta` (FE only, no bootstrap; `subset = w_500ms`, `w_1000ms`; the
W = 2 s arms serve Delta = 2 s). Against the W = 2 s arms, the shorter shifts
looked 2-4x "better" than real chirps purely through selection.

**Selection.** The B and C trains put every in-chase chirp into the last W
seconds of a chase, which by itself raises the offset HR. Compare arms with
the same W, never with the primary HR.

**Reading rule.** No acausal statistic has a null of 1 (the offset risk set
holds in-chase bins only, so the after-window of an offset bin lies outside
the chase and that of a bin just before an offset holds the post-offset
burst). Each statistic is read against its simulated distribution under
each account (step 5b), not against 1. A value compatible with causal and
non-causal accounts alike does not discriminate; in particular, with the
observed post-offset chirp response a causal effect also gives past ~
future, so past ~ future is not evidence against a causal effect. A value
inconsistent with every non-causal account (fewer than 2.5% of its simulated
datasets at or beyond the observed value, or p < 0.05 by z where that is the
criterion) indicates direction-specific timing those accounts do not
produce, not causality; verdicts that rest on borderline accounts (2.5-10%,
or p 0.05-0.2) are flagged, and accounts that were not simulated remain
possible. For onsets, a contrast beyond a no-effect account shows an
asymmetry beyond chase-start chirping only if that account also reproduces
the observed chase-start chirping (HR_future, chirps in the first 1 s of
chases). The `Reading` line of the acausal block in `summary.txt`
only says whether each CI includes 1; the simulation block below it says
what the values mean. `acausal/acausal_summary.txt` repeats both blocks
(rebuilt by `hz_acausal_sim.py` and by the main script's `--report-only`).

**API for simulations.** `hz_acausal.acausal_stats(trials, cfg,
kinds=("offset",), parts=("two_sided", "mirror"), backends=("glm_fe",),
taus=(1.0,), windows=(2.0,), deltas=(), n_boot=0, n_workers=1)` runs the same
statistics on any list of `Trial`s (real or simulated) without writing files
(~11 s for the real offset data) and returns effect rows with `variant`.
Key rows: `two_sided` / `pf_1000ms` / `sub_any_1000ms-sub_fut_1000ms`,
`pf_guard_500ms` / `sub_grd_500ms-sub_fgd_500ms`, `kernel_two_sided` /
`kernel_sub_mean-kernel_sub_fut_mean`, and `restricted_real-mirror` /
`sub_any_1000ms` (paired difference; point estimate, bootstrap CI only with
`n_boot > 0`). Simulated trials need `t_end` beyond `obs_end` for the future
window, else the last 5 s of bins are dropped (and counted).

Runtime (real data): see `acausal/acausal_timings.csv`; the standalone run took 1,229 s with 4 workers under a 16 GB memory cap (peak 5.4 GB for the whole process tree; A onset 800 s, its GLMMs on 2.9e6 bins; bootstrap 242 s).

## Acausal control: simulation calibration (`hz_acausal_sim.py`, step 5b)

The acausal statistics have no known expectation (risk sets are not
time-symmetric, in-chase and out-of-chase base rates differ 3.3-fold, chase
boundaries sit on a ~0.5 s lattice, and for onsets the after-window is raised
by the outcome itself), so step 5b simulates what each account produces for
exactly these statistics, with chase and chirp rates matched to the real
dyads.

**Generator.** `hz_sim`'s generator matched to the real trials, with the
first 5 s of its post-offset chirp-rate profile set to the 5-10 s level
(`flat_early`), so chirps locked to the offset come only from explicit
mechanisms. New `hz_sim` options, all off by default and drawn from their own
random streams (a sha256 fingerprint of seven existing configurations,
including the offset-locked chirps, is identical to the previous commit's):

- *response burst*: Poisson(`resp_n`) sub chirps at `off + L` after each true
  offset, `L` from the **empirical** post-offset excess profile: real
  post-offset chirps per latency bin (0.25 s to 2 s, 0.5 s to 5 s) minus the
  same counts in `flat_early` simulations without offset-locked chirps
  (`acausal_sim_profiles.csv`); `resp_n` per dyad. The default background
  ends every chirp burst at the chase end, so it has fewer chirps just after
  an offset than later; part of the short-latency excess therefore replaces
  burst continuations. The drift account below uses profiles estimated
  against its own background (`acausal_sim_profiles_drift.csv`). Negative
  excess (pre-offset, 2-4 s) is clipped to 0;
- *symmetric cluster*: with probability `sym_n / sym_m` per offset,
  Poisson(`sym_m` = 1, not calibrated) chirps at `off - L` and Poisson(1) at
  `off + L`, `L` from the empirical **pre**-offset excess profile; the
  post-offset excess it does not cover is a residual response burst;
- *bout*: the response mean multiplied after offsets with a sub chirp in the
  preceding 1 s. The multiplier is the data's same-offset Mantel-Haenszel
  rate ratio (post-offset chirps after offsets with vs without a pre-offset
  chirp, 3.02) **divided by** the same ratio computed with the post-offset
  chirps of the neighbouring offsets (1.60), where no bout can be shared:
  1.89, the offset-specific part (the raw 3.02 mostly reflects slow
  co-variation, below);
- *drift* (`sub_mod_frac`, `sub_mod_dur`, `sub_mod_mult`, `burst_cross`): a
  slow latent two-state process (25% of the time in a high state, mean
  episode 300 s) multiplies the sub chirp rate in both chase states and the
  offset-locked chirps (renormalised; no effect on the hazards), and chirp
  bursts continue across chase boundaries. Its high / low ratio is matched to
  the data's neighbouring-offset co-occurrence (below) on a grid
  (`acausal_sim_drift.csv`).

**Offset accounts** (lattice boundary error, SD 0.3 s rounded to 0.5 s,
unless `_exact`): `causal_only` (HR 2 for 0-1 s after a sub chirp, no
response burst), `burst_only` / `_exact` (no causal effect, response burst),
`causal_plus_burst` / `_exact` (both), `causal_lag_plus_burst` (HR 2 for
0-1.5 s: the data's association outlasts 1 s), `causal_het_plus_burst`
(dyad-specific effects, SD of the dyad log HR = the data's random-effects
SD, 0.36), `causal_plus_burst_drift` (on the drift background),
`causal_plus_burst_matched` (causal effect raised so that the standard 1 s HR
reproduces the observed 2.11: true log HR = log 2.11 / (median log HR of
`causal_plus_burst` / log 2)), `causal_plus_bout` (matched effect +
bout-dependent response), `common_trigger` / `_exact` (`hz_sim`'s hidden
trigger at nominal share 0.12, ~12% realized, no causal effect, plus the
response burst), `symmetric_cluster` (no causal effect).

**Onset accounts** (no offset effect; `hz_sim`'s generator with its own
post-offset profile; lattice boundary error): no effect; no mean effect with
dyad-specific effects (SD 1.26, the data's onset RE SD); `hz_sim`'s short
latent 'subordinate away' state (10% of time, 3 s episodes, chirps x6, onset
hazard x0.1); a causal HR 0.5 for 0-1 s (rates calibrated under the effect).
The default generator produces fewer sub chirps in the first 1-1.5 s of a
chase than the data (every chirp burst starts afresh at the onset; 0.5-1 s
after annotated onsets ~23 simulated vs 36 real), so the after-window of an
onset bin is raised less than in the data. Two further accounts, **chase
start matched** (`onset_start_matched`: no effect; `onset_causal_start`:
causal HR 0.5), scale the in-chase chirp-rate profile in the first 2 s of a
chase (four 0.5 s bins, one multiplier per bin pooled over dyads;
`calibrate_start`: each of 4 rounds runs `calibrate_params` with one rate
update, then rescales the multipliers by real / simulated counts after
annotated onsets) so that chase-start chirping matches the data. Onset
designs keep all event bins and 10% of the other bins (case-control, IPW
cloglog); the observed onset values in this block use the same subsampling
(full-data values are in the acausal block).

**Calibration.** `hz_sim.calibrate_params` per scenario (3 iterations x 8
datasets: baseline hazards, in/out-of-chase chirp rates per dyad), extended by
two optional hooks (`extra_stats`, `extra_update`) that move `resp_n` and
`sym_n` per dyad toward the real per-offset counts within 2 s after (response)
and before (cluster) offsets. Pre-offset counts in the causal, burst and
trigger scenarios are not targeted; whether a scenario reproduces them,
pooled and per dyad, is part of the result. Only the symmetric cluster is
calibrated per dyad on the pre-offset counts, so a poorer per-dyad match of
the causal accounts is not by itself evidence against them; a per-dyad
calibrated causal effect was not simulated.

**Statistics per dataset** (glm_fe, the `hz_acausal` functions; `observed` =
the same code on the real trials and identical to `acausal_effects.csv`):
standard 1 s HR, HR_past / HR_future and their ratio (1 s and guarded), the
two-sided kernel means and their ratio, restricted real vs mirror (W = 2 s,
1 s and guarded), chirp counts 0-1 / 0-2 s before and after offsets, and the
co-occurrence of a sub chirp in the last 1 s of a chase and the first 1 s
after its offset: relative to within-dyad independence (`both_ratio_1s`), the
same pairing across **neighbouring** offsets (pre side of offset i x post side
of offsets i +- 1, tens of seconds apart; `nb_ratio_1s`, the slow
co-variation baseline), their ratio (`local_ratio_1s`, the offset-specific
excess), and pre 0-1 s x post 2-3 s of the same offset (`far_ratio`). The
within-dyad independence null of the earlier version was wrong: in the data
the neighbouring-offset ratio is 1.38 and the post 2-3 s ratio 1.77, so the
1.94 "both sides" ratio mostly reflects chirping that co-varies over seconds
to minutes. Onsets add the sub chirp counts in the first 1 s and 0.5-1.5 s
of chases and in the 1 s before onsets. 100 datasets per account (4 with
`--quick`); median, 95% range, the observed value's percentile and
exceedance and, for the contrasts, the share of datasets with p < 0.05 in
each direction (`acausal_sim_summary.csv` / `.txt`; onsets in
`acausal_sim_onset_*.csv`).

**Criteria.** Each observed statistic is compared with every account either
by its **exceedance fraction** (the share of the account's simulated datasets
at or beyond the observed value, on the side where the observed value lies:
min(share >= observed, share <= observed); reported as e.g. "above 98/100",
the observed value lies above 98 of 100 datasets) or by z = (log observed -
log account median) / SE of the observed statistic and its two-sided p (`acausal_sim_observed.csv`: cluster-robust SE
of the real fit, `hz_acausal`'s dyad bootstrap for restricted real /
mirror, a dyad bootstrap for the co-occurrence ratios). Which one is
calibrated is decided per statistic (`criterion`): the exceedance when the
observed SE lies within the simulated fits' own cluster-robust SEs (97.5%
quantile, median over accounts), i.e. the real dyads show no more spread
than the simulated ones; z when it exceeds them (the dyads differ more than
simulated; for statistics without a simulated SE: observed SE / simulated
SD > 1.5, with the criterion itself called borderline between 1.25 and 2,
where the verdict is shown for both criteria). **Status** of an account
(`status_of`): *inconsistent* if the exceedance fraction is < 2.5% (z: p <
0.05), *borderline* if 2.5-10% (z: 0.05 <= p < 0.2), *consistent* otherwise.
An account fits if none of the fit statistics (standard HR and the pooled
chirp counts 0-1 s before / after offsets; onsets: the standard HR) is
inconsistent. A verdict counts consistent and borderline accounts as
compatible: *does not discriminate* (compatible causal and non-causal
accounts), *only causal* / *only non-causal*, or *inconsistent with every
account*; when counting borderline accounts as inconsistent changes it, the
reading says so ("hinges on borderline accounts"). This replaces the earlier
inside / outside-the-95%-range rule, which with 30 datasets turned on single
datasets (e.g. the kernel ratio 1.25 above 29 of 30 `causal + bout`
datasets). In the simulations the cluster-robust intervals of the
acausal contrasts cover the replicate mean in 100% of datasets (their SEs
are ~2x the actual spread), so an observed SE wider than the simulated range
is not by itself evidence of between-dyad differences; the earlier "lenient
z" framing was wrong. Here: the standard HR by z (observed SE 0.186 above
the simulated fits' 97.5% quantile 0.140: the dyads differ); past / future
1 s, guarded and kernel by the exceedance (observed 0.168 / 0.295 / 0.133,
within 0.334 / 0.340 / 0.181); restricted / mirror 1 s and the co-occurrence
ratios by the exceedance (SE / SD 1.16, 1.21); restricted / mirror guarded by
z with a borderline criterion (1.51; its verdict does not change with the
criterion). Onsets: the standard HR by z (0.538 above 0.277), the contrasts
by the exceedance (0.216 / 0.305 within 0.548 / 0.505).

**What the design implies** (100 datasets per account; numbers in
`summary.txt` and `acausal_sim_summary.txt`; "above k/100" = the observed
value lies above k of the 100 datasets; the per-account table is in
`acausal_sim_summary.csv`, columns `exceed_*` and `status*`).

- *1 s windows: no discrimination.* HR_past / HR_future 1.12 and restricted
  real / mirror 1.05 are consistent with causal and non-causal accounts
  alike (past / future: every fitting account between above 53/100 and below
  72/100; medians causal 1.04-1.19, trigger and symmetric cluster 1.11-1.14);
  the causal accounts give p < 0.05 for before > after in 0-3% of datasets.
  With the post-offset response the data contain, a causal effect produces
  past ~ future, so past ~ future is not evidence against it; `causal_only`
  (no response) gives 3.1.
- *Lags 0.5-1.5 s: more before-specific than the accounts, but the verdict
  hinges on borderline accounts.* The guarded ratio (1.66) is inconsistent
  with every fitting account (above 99-100/100) except the causal effect
  lasting 1.5 s (1.37 [1.02, 1.78], above 95/100: borderline); the kernel
  ratio (1.25) is inconsistent with every fitting account except that one
  (above 97/100) and the symmetric cluster (above 94/100), both borderline.
  So the guarded ratio reads "compatible only with causal accounts" and the
  kernel "does not discriminate", and both become "above every account" if
  borderline accounts are counted as inconsistent. With 30 datasets the
  earlier rule had called the kernel "consistent only with non-causal
  accounts" on the strength of one dataset. The B guarded ratio (1.27, judged
  by z) is compatible with every fitting account (common trigger borderline,
  z p = 0.19). No simulated account is compatible with all statistics. What
  is missing is an account with a stronger before-specific component at
  0.5-1.5 s (e.g. a causal effect peaking there, or a trigger whose chirp
  precedes the chase end by longer). The common-effect GLMM agrees (guarded
  1.66 [1.30, 2.12], kernel 1.25 [1.09, 1.44]); B guarded at W = 5 s
  (1.53 [1.08, 2.16], p = 0.028) is 1 of 4 uncorrected B contrasts.
- *Co-occurrence: no causal / non-causal contrast, and no 'one bout'
  evidence.* After removing the slow co-variation, the offset-specific excess
  is 1.41 [0.92, 1.86]. `causal_plus_burst_drift` reproduces it (1.47
  [1.26, 1.95], below 75/100) together with the neighbour baseline (1.34;
  data 1.38), the same-offset ratio (2.00; data 1.94) and the 2-3 s pairing
  (1.60; data 1.77), without any symmetric cluster: minute-scale co-variation
  plus bursts that continue across the chase end are enough. The reading
  labels this statistic "compatible only with causal accounts", but only a
  causal account has the drift background, the trigger has no
  offset-specific co-occurrence (0.92) and the symmetric cluster's 1.94
  depends on its uncalibrated size, so the label is flagged as not a causal /
  non-causal contrast. The data's raw same-offset bout ratio (3.02) is
  mostly the same slow co-variation (neighbouring offsets 1.60).
- *Per dyad.* Pre-offset counts (0-2 s, sum of |simulated - real| over dyads,
  real total 387): symmetric cluster 27 (calibrated on them), causal + burst
  accounts 83-96 (causal only 100), trigger 121. The causal accounts carry one
  common (or randomly varying) effect; a per-dyad calibrated causal effect was
  not simulated.
- *Onsets.* Under no effect the onset contrasts are 0.84 [0.53, 1.27] (1 s) and
  0.66 [0.43, 0.99] (guarded), not 1, and the joint HR_past falls below the
  standalone HR_past in 90% (1 s) and 100% (guarded) of the no-effect
  datasets (collider). The earlier
  reading said the observed contrasts (0.30, 0.14; subsampled designs) exceed
  "what chirping at chase start and these confounds produce", although the
  default generator chirps less at chase start than the data. Two facts
  change that picture. (i) The joint HR_future overstates chase-start
  chirping: it is adjusted for the past window, which in the data holds few
  chirps before onsets (27 in the 1 s before onsets vs ~50 without an
  effect), so it is larger than the future window alone (1.47 vs 1.26 for
  1 s, 2.08 vs 1.74 guarded). By the raw counts (60 sub chirps in the first
  1 s of chases, 74 at 0.5-1.5 s) and the future window alone, the plain
  no-effect account under-produces chase-start chirping only mildly (above
  85/100, 86/100; future window alone guarded above 91/100, borderline).
  (ii) The chase-start matched no-effect account reproduces the counts
  (62 [49, 76], 78 [59, 95]) and the future window alone (1.35, 1.72), yet the
  contrasts stay below all of its datasets (0.67 [0.43, 1.08] and 0.54
  [0.38, 0.82]; below 100/100). Against it, the asymmetry is on the pre-onset
  side (joint HR_past 0.45 vs 0.91, 27 vs 51 chirps in the 1 s before onsets),
  i.e. the pre-onset deficit that the standard past-only HR measures, here
  compared within dyads with the chase-start rate; the standard HR itself is
  only borderline against that account by z (p = 0.19) because the dyads
  disagree. The 1 s contrast is compatible with the causal HR 0.5 accounts
  (below 79/100 with the chase-start match; below 94/100, borderline,
  without); the guarded one (0.14) lies below every account, causal ones
  included. This points the same way as (b) but does not change the (b)
  verdict, which rests on the standard past-only HR and the dyads'
  disagreement.

For the manuscript: the acausal control does not show the offset
association to be an artefact of chirp clustering around the chase end (the
1 s contrasts cannot tell, and the clustering statistic is reproduced by a
causal account), nor does it establish direction-specific causation (the
lagged contrasts exceed nearly every simulated account, causal ones
included, and the remaining compatibility is borderline). Direction-
specificity still needs playback during chases or an independent measure of
chase end.

**Runtime.** Defaults (100 datasets x 13 offset + 6 onset accounts) took 4,949 s
(82 min) with 4 workers under a 16 GB memory cap (peak 2.3 GB for the process
tree): offset replicates 3,003 + 560 s (1,300 datasets, ~2.7 s wall per
dataset), onset calibration 114 s (incl. the chase-start profile), onset
accounts 1,149 s (601 case-control datasets), calibration 88 s, drift grid 14
s, profiles 2 s; per-step times in `acausal_sim_timings.csv`. A run with
`--n-reps 3` took 5 min; the whole pipeline in `--quick` mode took 31 min.

**Limits.** Only the accounts listed were simulated; mixtures (e.g. a
trigger plus a causal effect), other latent processes and a per-dyad
calibrated causal effect fall outside them. The empirical latency profiles
are relative to annotated offsets and are applied at true offsets, so under
the lattice error they are smeared twice. The HR-matched scenarios scale the
true log HR by the attenuation seen at HR 2, which is not exactly linear.
The symmetric cluster's size (1 chirp per side) and the drift process's
shape (two states, 300 s episodes, 25% high) are assumptions; only the
drift strength is matched to the data. The onset accounts calibrate the
onset statistics against no effect, heterogeneity, one latent state and one
causal effect, each also with the chase-start profile matched to the data,
not against everything that could make chirps precede quiet periods; the
chase-start match is pooled over dyads (four multipliers on the first 2 s of
the in-chase chirp-rate profile: 0.90 / 1.45 / 0.94 / 1.17 without, 1.05 /
1.36 / 0.98 / 1.23 with the causal effect). The exceedance fractions rest
on 100 datasets per account, so a status near a band edge (2.5% or 10%)
can still change with another set of seeds.

## Assumptions

- One dominant (winner) and one subordinate (loser) per trial, from
  `trial_summary.csv` (`win_ID`, `lose_ID`); draws are excluded.
- Chasing is dominant -> subordinate. **This cannot be verified** (Open
  issues 2).
- A chirp's sender is the RCNN track id. Every chirp in `chirp_ids_rcnn.npy`
  carries one of the two fish's ids, but **only because the detector dropped
  unassigned detections before writing the file** (Data contradictions 10).
  The assignment itself is not verified; for most chirps in five included dyads
  the detector box contains both fish's EODf (Data contradictions 11).
- Same-sender detections < 50 ms apart are double detections and are merged
  (logged).
- The analysis window is the first 3 h of each recording (`--max-time`), the
  "active period" of earlier analyses. The light schedule is not recorded in
  the dataset, but the data support the cut: in included dyads chase onsets
  drop to 0-6 per dyad in 3.0-3.5 h and none follow, and the last annotated
  BORIS event falls at 2.95-3.52 h, while some subordinates keep chirping
  (later time would create a spurious onset effect). The draw 2020-06-08 is
  annotated to the recording end (6.0 h), so chasing most likely stopped, but
  a shorter annotation effort cannot be excluded. The window reaches past the
  last annotation by 27 s (2020-05-13) and 166 s (2020-06-11).
- Chases running past the window end are censored (two: 2020-05-14 and
  2020-06-09); the time after the one unpaired onset is excluded from both risk
  sets.
- "Contact" (BORIS code 2) is a behaviour annotated outside chases (1 of 567
  contacts in included trials lies inside a chase), not contact during
  pursuit. Inside chases, the "time since contact" control is the time since
  a contact before the chase.
- Rises are off by default (Open issues 3).

## Data contradictions found (flagged, see `validation/flags.csv`)

1. **BORIS codes.** The dataset `README.txt` gives 1/2/3 for
   onset/offset/contact; the csv files use **0/1/2** (every 0 is followed by a
   1). The BORIS project file `Competition_experiment.boris` confirms this:
   `behaviors_conf` defines 0 = "aggression" (key a), 1 = "passiv" (key p),
   2 = "contact" (key c). The README's conversion (factor 1.034141) is
   outdated. The video runs at ~25.86 frames per grid second, not the nominal
   25 FPS.
2. **Chase actor not annotated.** BORIS `Subject` is "No focal subject" in all
   20 group-5/6 files, and the project file configures no subjects
   (`subjects_conf` is empty).
3. **File names.** The time axis is `times.npy`, not `time.npy`.
4. **LED misalignment in 2020-05-13 and 2020-06-09 under the reference
   converter**, which pairs the first and last LEDs of video and grid. In
   2020-06-09 the video's LED list starts at grid LED 3 (the flag text says "the
   video starts two grid LEDs late"): the first two flashes are in the video, but
   the LED was moved by ~30 pixels before its third flash and the LED detector
   only watched the later spot (`paper/output/led_alignment/03_*.png`). In
   2020-05-13 the first and last LEDs are the same pulse, but one grid
   LED is missing from the video mid-recording, at a video frame gap (the
   per-frame wall clock in `2020-05-13_0_times.dat` jumps 4.87 s at frame
   458,116, ~4.9 h). Event times shift by up to 2.3 s and 9.9 s inside the
   window. The data layer pairs LEDs by their gap fingerprint and fits the map
   piecewise (median residual ~10 ms in all 20 group-5/6 trials). **Earlier
   analyses that used the reference converter carry this error for these two
   trials** (the paper's loader used it until 2026-11-19; since then it uses the
   same map, `paper/scripts/led_alignment.py`). These two dyads hold about half of the offsets preceded by a sub
   chirp (see the `drop_led_corrected` sensitivity and the "Support" line in
   `summary.txt`).
5. **Unpaired chase onset** in 2020-05-13 (1686.2 s): dropped, 1686.2-1745.4 s
   marked as unknown chase state; `trial_summary.csv` counts 141 chases, the
   paired count is 140. No overlapping chases, zero-length chases or events
   outside the recording in included trials.
6. **Chase boundaries are precise to ~0.5 s only.** Minimum duration 1.00 s;
   93% of chases < 3 s last 1.0, 1.5, 2.0 or 2.5 s (+-0.1 s).
7. **The draw 2020-06-08 has annotated chases**, including ones of 808 s and
   8,943 s (the latter past the recording end). Excluded as a draw.
8. **The "EODfs too close" list is inconsistent.** Included 2020-06-03,
   -06-04 and -06-05 have |dEODf| < 30 Hz for 99%, 30% and 51% of the window,
   comparable to or worse than the excluded 2020-05-28. Kept and flagged; see
   the `drop_close_eodf` sensitivity.
9. **`EODf_win` / `EODf_lose` in `trial_summary.csv`** are higher than the
   track medians in every flagged case, by 15-74 Hz (up to ~55 Hz in included
   dyads, e.g. 2020-05-15); unflagged differences (< 10 Hz) go either way. The
   systematic positive offset suggests temperature-normalised values (the
   definition is not documented). The tracks are used.
10. **Unassigned chirps are dropped upstream.** The chirp detector removes
    detections without a track id before writing `chirp_times_rcnn.npy`
    (`chirpdetector/detection/detect_chirps.py`, "drop unassigned"); the
    per-trial `chirpdetector_bboxes.csv` keeps them. In included dyads there
    are 95 distinct unassigned detections (>= 50 ms from any assigned chirp):
    2020-06-02 86 (71 in the window), 06-11 5, 05-11 2, 05-12 1, 06-04 1. The
    large NaN counts in 05-11, 05-13 and 05-29 are duplicates of assigned
    chirps. Flag `chirp_unassigned_upstream`; these chirps are not used.
11. **Sender assignment is often geometrically ambiguous.** For most chirps in
    2020-05-15, 06-02, 06-03, 06-05 and 06-11 both fish's EODf lie inside the
    chirp's bounding box (flag `sender_ambiguous_bbox`; per-trial shares in
    `trial_summary.csv`), so the sender comes from the RCNN emitter model only.
    Causes: close EODfs (06-03, 06-05) and very tall boxes (06-02, 06-11:
    median box height 509 and 688 Hz). See the `drop_sender_ambiguous` and
    `unambiguous_chirps` sensitivities.
12. **Other chirp-file issues.** Times are unsorted in 26 files (sorted on
    load); 236 same-sender duplicates < 50 ms (37 in included trials) were
    merged; one dominant/subordinate pair within 22 ms in 2020-06-02 is flagged
    as a possible double assignment. `chirp_notes.csv` rates the older CNN
    detector, not the RCNN detections used here.
13. **Chirp counts are very uneven.** In the analysis window five included
    dyads have no dominant chirps (05-12, 05-14, 05-18, 05-19, 05-27) and two
    more have <= 3 (05-11, 06-09); in 2020-06-02 and 2020-06-11 the dominant
    chirps 5-12 times more than the subordinate (1,230 vs 267; 1,343 vs 110),
    and both are dyads with ambiguous sender assignment (contradiction 11).
14. **Rises exist** as `analysis/rise_idx.npy` (frame indices, ~0.33 s),
    although they were believed to be missing. Counts match
    `trial_summary.csv`.
15. **Group 4 has BORIS files but no usable LED sync** (`led_times.csv` holds
    nominal values), and group 3 has no BORIS files.

## Open issues

1. **Common trigger or annotation lag vs causal effect.** A latent event that
   both elicits a subordinate chirp and ends the chase (or keeps the dominant
   from starting one) produces the same association. For (a), a uniform
   late-marking lag of the offsets is tested directly (offsets moved earlier
   by up to 0.75 s) and in simulation (lag scenarios); a variable lag or a
   common trigger remains possible. The lag profile (tau scan) does not
   separate these. The acausal control (step 4b) asks whether the
   association is specific to chirps before the event; its statistics have no
   null of 1, and step 5b shows which of them can separate the simulated
   accounts here (see its section). It cannot establish causality, and for
   onsets it does not test (b) at all. Only chirp playback during chases, or
   an independent measure of chase end (video tracking of the subordinate's
   escape), can separate a causal effect from the rest.
2. **Chase direction is unverifiable from BORIS.** Reversed chases
   (subordinate -> dominant) cannot be counted or dropped. The analysis
   assumes all chases are dominant -> subordinate.
3. **Rises.** `analysis/rise_idx.npy` exists despite the belief that it did
   not. Rises enter the hazard models only with `--use-rises` (delayed by one
   ~0.33 s spectrogram frame to stay causal). Not part of the default results;
   the flag now runs end to end (a column-selection bug that crashed the
   jitter test and leave-one-dyad-out with `--use-rises` is fixed and tested).
4. **Excluded trials.** 28 of 43 trials: group 3 (14, no BORIS), group 4 (9,
   no LED sync; 5 are also draws), the four "EODfs too close" recordings, and
   the draw 2020-06-08. Results cover 15 dyads of 10 fish from groups 5 and 6.
5. **Chirp detection and assignment.** Detection may be poorer during chases.
   Sender assignment is ambiguous for most chirps in five dyads (contradiction
   11), and unassigned detections are dropped upstream (contradiction 10).
   Missed chirps are simulated (little bias); misassigned chirps are not.
6. **Heterogeneity between dyads.** The chirp association differs between
   dyads for both questions (per-dyad O/E and G tests in `summary.txt`). For
   (b) one well-sampled dyad (2020-05-11) runs significantly the other way.
   The random-effects meta-analysis treats each dyad's baseline as known and
   fish recur across dyads; with ~15 dyads its interval is approximate
   (calibration in `sim_summary.txt`).
7. **Chase boundary precision (~0.5 s)** limits the interpretable kernel
   shape and attenuates the onset effect in simulation. For (b), the
   across-dyad verdict depends on whether chirps within 0.5 s of the bin are
   counted (compare the 1 s and the lag-guarded rows).
8. **Minor bookkeeping.** One 2020-05-13 onset falls at the end of the
   unknown-state interval and is not in the onset risk set.
9. **`outputs/` is not gitignored.** Whether the outputs are committed is left
   to the repository owner.
10. **dEODf baseline (fixed).** Earlier versions interpolated the trailing
    median linearly between raw samples and back-filled the time before a
    fish's first tracked sample, so up to ~1% of bins used EODf from >10 s in
    the future (inside track gaps) and the first 50-390 s of each trial used
    the first later value. Now step-hold plus a `deodf_missing` indicator;
    `test_eodf_baseline_is_step_hold` checks it. The earlier verification refit
    showed chirp log HRs move by <= 0.0025, so no reported result depends on it.

## Modules

| file | role |
|---|---|
| `chirp_chase_hazard.py` | CLI, sensitivity refits, `effects.csv`, `summary.txt` |
| `hz_config.py` | `Config` dataclass shared by all modules |
| `hz_data.py` | `Trial`, `load_trials`, `events_table`, validation report, chirp box checks |
| `hz_desc.py` | descriptive alignments, circular-shift and matched nulls |
| `hz_hazard.py` | designs, GLMM / GLM fits, per-dyad O/E + random-effects meta, jitter tests, leave-one-dyad-out, figures |
| `hz_sim.py` | generative model, calibration, simulation scenarios, figures |
| `hz_acausal.py` | acausal control: future-window covariates, two-sided kernel, mirror / restricted-real / shift surrogate trains (with the shift comparators), state rates, dyad bootstrap, figures, summary blocks and acausal_summary.txt; `acausal_stats` for simulations |
| `hz_acausal_sim.py` | simulation calibration of the acausal control: latency profiles, drift calibration, offset and onset accounts (incl. the chase-start profile calibration), calibration, replicates, co-occurrence and chase-start statistics, per-statistic criteria, exceedance fractions and status bands, data-driven reading, summary, figure |
| `test_hz_hazard.py`, `test_hz_sim.py`, `test_hz_acausal.py`, `test_hz_acausal_sim.py` | tests |
