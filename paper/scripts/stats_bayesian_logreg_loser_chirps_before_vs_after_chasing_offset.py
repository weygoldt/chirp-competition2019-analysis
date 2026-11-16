# Ported from chirp_competition d040123 (src/stats_bayesian_logreg_loser_chirps_before_vs_after_chasing_offset.py).
# Changes: input/output under $PAPER_OUT/data/processed; the input is written by the
# reconstructed make_chirps_around_chasing_obs.py (its old writer was never committed, audit O5).
# Events, observed fraction and posterior median/HDI also go to write_stats ("before_offset_logreg").
"""
Estimate P(loser chirp in the 10 s BEFORE chasing offset | at least one loser
chirp in the 20 s window around chasing offset) using an intercept-only
Bayesian Bernoulli mixed model with a random intercept on recording.
"""

from pathlib import Path
from IPython import embed

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from stats_utils import parse_bambi_model_output, save_json, log_odds_to_prob
from manuscript_stats import write_stats
from paper_config import PROCESSED


HDI_PROB = 0.95
SEED = 20250101


def _bernoulli_summary(p_draws, observed_fraction, n_events, hdi_prob=HDI_PROB):
    """Return a dict of posterior summaries for an intercept-only Bernoulli model.

    p_draws : 1-D array of posterior draws of the success probability.
    """
    p_draws = np.asarray(p_draws).ravel()
    mean_p = float(p_draws.mean())
    median_p = float(np.median(p_draws))
    lo, hi = az.hdi(p_draws, hdi_prob=hdi_prob).tolist()
    p_gt_half = float((p_draws > 0.5).mean())
    return {
        "n_events": int(n_events),
        "observed_fraction": float(observed_fraction),
        "posterior_mean": mean_p,
        "posterior_median": median_p,
        "hdi_prob": hdi_prob,
        "hdi_lower": float(lo),
        "hdi_upper": float(hi),
        "prob_gt_0.5": p_gt_half,
    }


def _print_summary(summary, title):
    """Banner-style terminal printout for inclusion in Results."""
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
    print(f"  Pr(p > 0.5 | data)      {summary['prob_gt_0.5']:.3f}")
    print(bar)


def main():
    dataroot = PROCESSED / "chirp_event_correlations" / "data" / "chasing"
    files = list(dataroot.glob("*obs*"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file with 'obs' in name, got {files}")
    file = files[0]

    df = pd.read_csv(file)
    df["loser_chirped_before_offset"] = df["n_loser_chirps_10s_before_chasing_off"] > 0

    # Condition on the loser having chirped at least once in the 20 s window
    mask = df["n_loser_chirps_20s_around_chasing_off"] > 0
    n_discarded = int((~mask).sum())
    print(
        f"Discarding {n_discarded} rows where loser did not chirp in the 20 s window."
    )
    df = df[mask].copy()

    n_events = len(df)
    observed_fraction = float(df["loser_chirped_before_offset"].mean())
    print(f"Observed fraction P(before | chirped in window): {observed_fraction:.3f}")

    model = bmb.Model(
        formula="loser_chirped_before_offset ~ 1 + (1|recording)",
        data=df,
        family="bernoulli",
        categorical=["recording"],
    )

    trace = model.fit(
        tune=6000,
        draws=1000,
        chains=10,
        cores=10,
        target_accept=0.99,
        random_seed=SEED,
    )

    # Posterior on probability scale: transform each draw, then summarise on p
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
        title="P(loser chirp in 10 s BEFORE offset | any chirp in 20 s window)",
    )

    # Persist outputs
    filename = Path(__file__).stem
    savepath = PROCESSED
    savepath.mkdir(parents=True, exist_ok=True)

    trace.to_netcdf(savepath / f"{filename}_trace.nc")

    model_summary = parse_bambi_model_output(model)
    save_json(
        {"model_info": model_summary, "posterior_summary": summary},
        savepath / f"{filename}_summary.json",
    )
    write_stats("before_offset_logreg", {
        "n_events": summary["n_events"], "n_discarded": n_discarded,
        "observed_fraction": summary["observed_fraction"],
        "model": {"median": summary["posterior_median"],
                  "hdi": [summary["hdi_lower"], summary["hdi_upper"]],
                  "p_gt_0_5": summary["prob_gt_0.5"]},
    })


if __name__ == "__main__":
    main()
