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
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

PROC = Path("data/processed")
GAMMA_DIR = PROC / "chirp_effect_on_dominant_adaptive_window/gamma_glmm_binary"
COUPLING_DIR = PROC / "chirp_coupling/negbin"
STATUS_DIR = PROC / "stats_bayesian_glmm_chirps_status_v3"
OUT_CSV = PROC / "reported_posteriors_median_hdi.csv"
HDI_PROB = 0.95

# Corners of the central 80% of the observed predictors in the coupling
# figure (plot_chirp_coupling.py prints these values).
PREV_ICI_P10, PREV_ICI_P90 = 5.45, 172.25
DURATION_P10, DURATION_P90 = 2.01, 15.03


def draws(path: Path, var: str, **sel) -> np.ndarray:
    """Flattened posterior draws of `var` from a saved InferenceData."""
    da = az.from_netcdf(path).posterior[var]
    if sel:
        da = da.sel(**sel)
    return da.to_numpy().ravel()


def invlogit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def summary(name: str, samples: np.ndarray, unit: str) -> dict:
    lo, hi = az.hdi(samples, hdi_prob=HDI_PROB)
    return dict(quantity=name, unit=unit, median=float(np.median(samples)),
                hdi_low=float(lo), hdi_high=float(hi))


def main() -> None:
    rows = []

    # Chirp production and competition status (negative binomial, v3).
    status_nc = STATUS_DIR / "posterior.nc"
    rows.append(summary("status: loser/winner count multiplier",
                        np.exp(-draws(status_nc, "status", status_dim="winner")), "x"))
    rows.append(summary("status: male/female count multiplier",
                        np.exp(draws(status_nc, "sex", sex_dim="male")), "x"))
    rows.append(summary("status: body size multiplier per SD",
                        np.exp(draws(status_nc, "size_z")), "x"))

    # Echo responses and chirp occurrence around chasing offsets (intercept-only
    # Bernoulli GLMMs): probability for a typical recording.
    for name, file in [
        ("echo: loser chirp within 10 s after winner chirp",
         "stats_bayesian_logreg_echo_responses_new_loser_after_winner_obs_trace.nc"),
        ("echo: winner chirp within 10 s after loser chirp",
         "stats_bayesian_logreg_echo_responses_new_winner_after_loser_obs_trace.nc"),
        ("offset: P(loser chirp in -3..13 s window)",
         "stats_bayesian_logreg_loser_chirps_around_chasing_offset_loser_trace.nc"),
        ("offset: P(winner chirp in -3..13 s window)",
         "stats_bayesian_logreg_loser_chirps_around_chasing_offset_winner_trace.nc"),
        ("before offset: P(loser chirps before offset)",
         "stats_bayesian_logreg_loser_chirps_before_vs_after_chasing_offset_trace.nc"),
    ]:
        rows.append(summary(name, invlogit(draws(PROC / file, "Intercept")), "probability"))

    # Inter-chase interval Gamma GLMMs: factor on the expected interval when the
    # loser chirped.
    for model, var in [
        ("M1_causal", "loser_chirped_fixed_m3p13"),
        ("M2_acausal", "loser_chirped_fixed_m3p13"),
        ("M3_causal_adj_prev", "loser_chirped_fixed_m3p13"),
        ("M4_causal_adj_full", "loser_chirped_fixed_m3p13"),
        ("M5_duration_from_chirp", "loser_chirped_fixed_m3p13"),
        ("M5b_duration_from_chirp_adj", "loser_chirped_fixed_m3p13"),
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
        rows.append(summary(f"ICI {model}: exp(beta_chirp)",
                            np.exp(posterior[var].to_numpy().ravel()), "x"))

    # Random-effects comparison for M4 (appendix).
    crossed = GAMMA_DIR / "sensitivity_loserid_winnerid_recording__full__M4_causal_adj_full.nc"
    primary = GAMMA_DIR / "primary_recording__full__M4_causal_adj_full.nc"
    rows.append(summary("RE crossed M4: exp(beta_chirp)",
                        np.exp(draws(crossed, "loser_chirped_fixed_m3p13")), "x"))
    rows.append(summary("RE primary M4: sigma_recording", draws(primary, "1|recording_sigma"), "log scale"))
    for g in ["recording", "loser_id", "winner_id"]:
        rows.append(summary(f"RE crossed M4: sigma_{g}", draws(crossed, f"1|{g}_sigma"), "log scale"))

    # Coupling models (negative binomial): per-doubling rate multipliers.
    prev, dur = "log_time_since_previous_chasing_c", "log_chasing_duration_c"
    for model, vars_ in [
        ("C1_prev_ICI", [prev]),
        ("C2_prev_ICI_and_D", [prev, dur]),
        ("C2_zinb_check", [prev, dur]),
        ("C2_full_width_only", [prev, dur]),
    ]:
        for var in vars_:
            label = "prev ICI" if var == prev else "duration"
            rows.append(summary(f"coupling {model}: per doubling of {label}",
                                2.0 ** draws(COUPLING_DIR / f"{model}.nc", var), "x"))

    # Coupling figure: multipliers across the central 80% of each predictor.
    c2 = COUPLING_DIR / "C2_prev_ICI_and_D.nc"
    b_prev, b_dur = draws(c2, prev), draws(c2, dur)
    m_prev = (PREV_ICI_P90 / PREV_ICI_P10) ** b_prev
    m_dur = (DURATION_P90 / DURATION_P10) ** b_dur
    rows.append(summary("coupling C2: prev ICI p10 -> p90", m_prev, "x"))
    rows.append(summary("coupling C2: duration p10 -> p90", m_dur, "x"))
    rows.append(summary("coupling C2: joint contrast (short ICI, long chase vs long ICI, short chase)",
                        m_dur / m_prev, "x"))

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(out.round(4).to_string(index=False))
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
