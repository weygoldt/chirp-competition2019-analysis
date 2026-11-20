# Ported from chirp_competition d040123 (src_paper/summarize_reported_posteriors.py).
# Changes: inputs/output under $PAPER_OUT/data/processed; a missing Bernoulli trace is reported and
# skipped (as the script already did for the Gamma GLMMs): the before/after-offset input has no writer (O5);
# the coupling corners (10th/90th percentiles of the predictors) are computed from the coupling model's
# data instead of being copied from a printout; every quantity also goes to write_stats ("posteriors").
# The offset-window labels come from offset_window.py (were "-3..13 s"); the Gamma predictor is
# loser_chirped_offset_window (was loser_chirped_fixed_m3p13).
"""
Posterior median and 95% HDI for every model-based number in the manuscript.

The manuscript summarizes all posteriors the same way: posterior median and
95% highest-density interval (HDI). The individual analysis scripts grew
different conventions over time (posterior means, equal-tailed intervals), so
this script recomputes the reported quantities from the saved posteriors in one
place. It does not refit anything.

Not covered here: the hurdle model, whose summaries are printed by
`stats_bayesian_hurdle_glmm_chirprate_deviation_during_chasing_offsets.py`
and `plot_hurdle_decomposition.py` (both use median + HDI).

Output:
    data/processed/reported_posteriors_median_hdi.csv
    $PAPER_OUT/stats/posteriors.json (the same quantities, keyed for the text)
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import PROCESSED
from stats_chirp_coupling import prepare_data

PROC = PROCESSED
GAMMA_DIR = PROC / "chirp_effect_on_dominant_adaptive_window/gamma_glmm_binary"
COUPLING_DIR = PROC / "chirp_coupling/negbin"
STATUS_DIR = PROC / "stats_bayesian_glmm_chirps_status_v3"
OUT_CSV = PROC / "reported_posteriors_median_hdi.csv"
HDI_PROB = 0.95
WINDOW_PREDICTOR = "loser_chirped_offset_window"  # stats_chirp_affects_chasing.PRIMARY_PREDICTOR


def draws(path: Path, var: str, **sel) -> np.ndarray:
    """Flattened posterior draws of `var` from a saved InferenceData."""
    da = az.from_netcdf(path).posterior[var]
    if sel:
        da = da.sel(**sel)
    return da.to_numpy().ravel()


def invlogit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def summary(key: str, name: str, samples: np.ndarray, unit: str) -> dict:
    """Median and HDI of `samples`; `key` is the dotted stats key, `name` the CSV label."""
    lo, hi = az.hdi(samples, hdi_prob=HDI_PROB)
    return dict(key=key, quantity=name, unit=unit, median=float(np.median(samples)),
                hdi_low=float(lo), hdi_high=float(hi))


def stats_tree(rows: list[dict]) -> dict:
    """The rows as nested dictionaries along their dotted keys: {"median", "hdi": [low, high]}."""
    tree: dict = {}
    for row in rows:
        *path, leaf = row["key"].split(".")
        node = tree
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = {"median": row["median"], "hdi": [row["hdi_low"], row["hdi_high"]]}
    return tree


def main() -> None:
    rows = []
    left, right = offset_window()
    window = f"{left:g}..{right:g} s window"

    # Chirp production and competition status (negative binomial, v3).
    status_nc = STATUS_DIR / "posterior.nc"
    rows.append(summary("status.loser_winner", "status: loser/winner count multiplier",
                        np.exp(-draws(status_nc, "status", status_dim="winner")), "x"))
    rows.append(summary("status.male_female", "status: male/female count multiplier",
                        np.exp(draws(status_nc, "sex", sex_dim="male")), "x"))
    rows.append(summary("status.size_per_sd", "status: body size multiplier per SD",
                        np.exp(draws(status_nc, "size_z")), "x"))

    # Echo responses and chirp occurrence around chasing offsets (intercept-only
    # Bernoulli GLMMs): probability for a typical recording.
    for key, name, file in [
        ("echo.loser_after_winner", "echo: loser chirp within 10 s after winner chirp",
         "stats_bayesian_logreg_echo_responses_new_loser_after_winner_obs_trace.nc"),
        ("echo.winner_after_loser", "echo: winner chirp within 10 s after loser chirp",
         "stats_bayesian_logreg_echo_responses_new_winner_after_loser_obs_trace.nc"),
        ("offset.loser", f"offset: P(loser chirp in {window})",
         "stats_bayesian_logreg_loser_chirps_around_chasing_offset_loser_trace.nc"),
        ("offset.winner", f"offset: P(winner chirp in {window})",
         "stats_bayesian_logreg_loser_chirps_around_chasing_offset_winner_trace.nc"),
        ("before_offset.loser", "before offset: P(loser chirps before offset)",
         "stats_bayesian_logreg_loser_chirps_before_vs_after_chasing_offset_trace.nc"),
    ]:
        if not (PROC / file).exists():
            print(f"missing {PROC / file}")
            continue
        rows.append(summary(key, name, invlogit(draws(PROC / file, "Intercept")), "probability"))

    # Inter-chase interval Gamma GLMMs: factor on the expected interval when the
    # loser chirped.
    for model, var in [
        ("M1_causal", WINDOW_PREDICTOR),
        ("M2_acausal", WINDOW_PREDICTOR),
        ("M3_causal_adj_prev", WINDOW_PREDICTOR),
        ("M4_causal_adj_full", WINDOW_PREDICTOR),
        ("M5_duration_from_chirp", WINDOW_PREDICTOR),
        ("M5b_duration_from_chirp_adj", WINDOW_PREDICTOR),
        ("Mtail", "loser_chirped_chase_tail"),
        ("Mpost_clipped", "loser_chirped_post_offset_clipped"),
        ("Mpost_unclipped", "loser_chirped_post_offset_unclipped"),
    ]:
        path = GAMMA_DIR / f"primary_recording__full__{model}.nc"
        if not path.exists():
            print(f"missing {path}")
            continue
        posterior = az.from_netcdf(path).posterior
        if var not in posterior:
            var = next(v for v in posterior.data_vars if v.startswith("loser_chirped"))
        rows.append(summary(f"ici.{model}", f"ICI {model}: exp(beta_chirp)",
                            np.exp(posterior[var].to_numpy().ravel()), "x"))

    # The same models on the chasing events whose window was never shortened
    # (chase at least as long as the pre-offset part, next chase no sooner than
    # the post-offset part): there the window cannot depend on the response.
    for model in ["M1_causal", "M3_causal_adj_prev", "M4_causal_adj_full",
                  "M5_duration_from_chirp"]:
        path = GAMMA_DIR / f"primary_recording__untruncated__{model}.nc"
        if not path.exists():
            print(f"missing {path}")
            continue
        rows.append(summary(f"ici_complete_windows.{model}",
                            f"ICI {model}, complete windows: exp(beta_chirp)",
                            np.exp(draws(path, WINDOW_PREDICTOR)), "x"))

    # Random-effects comparison for M4 (appendix).
    crossed = GAMMA_DIR / "sensitivity_loserid_winnerid_recording__full__M4_causal_adj_full.nc"
    primary = GAMMA_DIR / "primary_recording__full__M4_causal_adj_full.nc"
    rows.append(summary("re.crossed_M4", "RE crossed M4: exp(beta_chirp)",
                        np.exp(draws(crossed, WINDOW_PREDICTOR)), "x"))
    rows.append(summary("re.primary_M4_sigma.recording", "RE primary M4: sigma_recording",
                        draws(primary, "1|recording_sigma"), "log scale"))
    for g in ["recording", "loser_id", "winner_id"]:
        rows.append(summary(f"re.crossed_M4_sigma.{g}", f"RE crossed M4: sigma_{g}",
                            draws(crossed, f"1|{g}_sigma"), "log scale"))

    # Coupling models (negative binomial): per-doubling rate multipliers.
    prev, dur = "log_time_since_previous_chasing_c", "log_chasing_duration_c"
    for model, vars_ in [
        ("C1_prev_ICI", [prev]),
        ("C2_prev_ICI_and_D", [prev, dur]),
        ("C2_zinb_check", [prev, dur]),
        ("C2_full_width_only", [prev, dur]),
    ]:
        for var in vars_:
            label, key = ("prev ICI", "prev_ici") if var == prev else ("duration", "duration")
            rows.append(summary(f"coupling.{model}.{key}_per_doubling",
                                f"coupling {model}: per doubling of {label}",
                                2.0 ** draws(COUPLING_DIR / f"{model}.nc", var), "x"))

    # Coupling figure: multipliers across the central 80% of each predictor
    # (10th to 90th percentile of the observations the coupling models were fitted to).
    obs = prepare_data()
    prev_p10, prev_p90 = np.quantile(obs["prev_ICI"], [0.10, 0.90])
    dur_p10, dur_p90 = np.quantile(obs["D"], [0.10, 0.90])
    c2 = COUPLING_DIR / "C2_prev_ICI_and_D.nc"
    b_prev, b_dur = draws(c2, prev), draws(c2, dur)
    m_prev = (prev_p90 / prev_p10) ** b_prev
    m_dur = (dur_p90 / dur_p10) ** b_dur
    rows.append(summary("coupling.C2_prev_ICI_and_D.prev_ici_p10_p90",
                        "coupling C2: prev ICI p10 -> p90", m_prev, "x"))
    rows.append(summary("coupling.C2_prev_ICI_and_D.duration_p10_p90",
                        "coupling C2: duration p10 -> p90", m_dur, "x"))
    rows.append(summary("coupling.C2_prev_ICI_and_D.joint_contrast",
                        "coupling C2: joint contrast (short ICI, long chase vs long ICI, short chase)",
                        m_dur / m_prev, "x"))

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="key").to_csv(OUT_CSV, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(out.drop(columns="key").round(4).to_string(index=False))
    print(f"\nSaved {OUT_CSV}")
    print(f"coupling corners: prev ICI {prev_p10:.2f} -> {prev_p90:.2f} s, "
          f"duration {dur_p10:.2f} -> {dur_p90:.2f} s (n = {len(obs)})")

    stats = stats_tree(rows)
    stats["coupling"]["corners"] = {"prev_ici_s": [prev_p10, prev_p90],
                                    "duration_s": [dur_p10, dur_p90], "n": len(obs)}
    write_stats("posteriors", stats)


if __name__ == "__main__":
    main()
