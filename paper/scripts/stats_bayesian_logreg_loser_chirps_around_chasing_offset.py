# Ported from chirp_competition d040123 (src/stats_bayesian_logreg_loser_chirps_around_chasing_offset.py).
# Changes: input/output under $PAPER_OUT/data/processed (imports paper_config, which also sets the fork
# start method for pymc on Python 3.14). Model, priors, sampler settings and seed (20250101) unchanged.
# Events, observed fractions and posterior median/HDI also go to write_stats ("offset_logreg").
"""
Estimate whether loser/winner chirps occur in a fixed window around chasing
offsets using a Bayesian Bernoulli intercept-only GLMM with a random
intercept on recording.

The estimand is the population-level probability that an event has at least
one chirp from the focal role (loser or winner) in the analysis window
(`*_chirps_fixed_m3p13`, i.e. ~-3 to +13 s relative to chasing offset).
"""

from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from manuscript_stats import write_stats
from paper_config import PROCESSED
from stats_utils import parse_bambi_model_output, log_odds_to_prob, save_json


HDI_PROB = 0.95
SEED = 20250101


def _bernoulli_summary(p_draws, observed_fraction, n_events, hdi_prob=HDI_PROB):
    """Posterior summaries on the probability scale for an intercept-only model."""
    p_draws = np.asarray(p_draws).ravel()
    lo, hi = az.hdi(p_draws, hdi_prob=hdi_prob).tolist()
    return {
        "n_events": int(n_events),
        "observed_fraction": float(observed_fraction),
        "posterior_mean": float(p_draws.mean()),
        "posterior_median": float(np.median(p_draws)),
        "hdi_prob": hdi_prob,
        "hdi_lower": float(lo),
        "hdi_upper": float(hi),
    }


def _print_summary(summary, title):
    pct = int(round(summary["hdi_prob"] * 100))
    width = max(60, len(title) + 4)
    bar = "=" * width
    print()
    print(bar)
    print(title)
    print(bar)
    print(f"  events (n)              {summary['n_events']}")
    print(f"  observed fraction       {summary['observed_fraction']:.3f}")
    print(
        f"  posterior median        {summary['posterior_median']:.3f}   "
        f"[{pct}% HDI: {summary['hdi_lower']:.3f}, {summary['hdi_upper']:.3f}]"
    )
    print(f"  posterior mean          {summary['posterior_mean']:.3f}")
    print(bar)


def _load_data():
    dataroot = PROCESSED / "chirp_event_correlations_m3p13" / "data" / "chasing"
    files = list(dataroot.glob("*obs*"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file with 'obs' in name, got {files}")
    return pd.read_csv(files[0])


def fit_for_mode(df, mode, savepath):
    if mode == "loser":
        df = df.assign(chirped=df["n_loser_chirps_fixed_m3p13"] > 0)
    elif mode == "winner":
        df = df.assign(chirped=df["n_winner_chirps_fixed_m3p13"] > 0)
    else:
        raise ValueError(f"mode must be 'loser' or 'winner', got {mode!r}")

    n_events = len(df)
    observed_fraction = float(df["chirped"].mean())

    bmb_model = bmb.Model(
        formula="chirped ~ 1 + (1|recording)",
        data=df,
        family="bernoulli",
        categorical=["recording"],
    )

    trace = bmb_model.fit(
        tune=6000,
        draws=1000,
        chains=25,
        cores=25,
        target_accept=0.99,
        random_seed=SEED,
    )

    intercept_draws = trace.posterior["Intercept"].to_numpy().ravel()
    p_draws = log_odds_to_prob(intercept_draws)

    summary = _bernoulli_summary(
        p_draws,
        observed_fraction=observed_fraction,
        n_events=n_events,
        hdi_prob=HDI_PROB,
    )
    _print_summary(
        summary,
        title=f"P({mode} chirp in fixed window around chasing offset)",
    )

    filename = Path(__file__).stem + "_" + mode
    trace.to_netcdf(savepath / f"{filename}_trace.nc")
    save_json(
        {
            "model_info": parse_bambi_model_output(bmb_model),
            "posterior_summary": summary,
        },
        savepath / f"{filename}_summary.json",
    )
    return summary


def main():
    savepath = PROCESSED
    savepath.mkdir(parents=True, exist_ok=True)

    df = _load_data()
    stats = {}
    for mode in ("loser", "winner"):
        s = fit_for_mode(df, mode, savepath)
        stats[mode] = {"n_events": s["n_events"], "observed_fraction": s["observed_fraction"],
                       "model": {"median": s["posterior_median"],
                                 "hdi": [s["hdi_lower"], s["hdi_upper"]]}}
    write_stats("offset_logreg", stats)


if __name__ == "__main__":
    main()
