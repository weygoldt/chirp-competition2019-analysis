# Ported from chirp_competition d040123 (src_paper/stats_chirp_coupling.py).
# Changes: input/output under $PAPER_OUT/data/processed. Models, priors, sampler settings and seed unchanged.
# The data anchors, divergences, LOO differences, Pareto-k counts and the full-width subset go to
# write_stats ("coupling"). The window is the offset window of offset_window.py (was -3, +13 s; its columns
# were *_fixed_m3p13 and fixed_window_duration); the full-width subset is its width minus 0.1 s (was the
# constant 15.9 s).
"""
Coupling models — how do chirp rates track competition intensity?

Builds a three-step Bayesian Negative Binomial GLMM hierarchy on chirp
counts in the offset window around each chase offset
(``offset_window.py``; clipped at the chase onset and the next chase),
using log-window-width as an exposure offset so the slope of each
predictor is interpreted on the *rate* scale (chirps per unit time),
not on raw counts.

C0 — intercept only.
    log(rate) = beta_0 + u_loser + v_winner
    Establishes the baseline chirp rate and random-effect variance.

C1 — add previous ICI (bout-density coupling).
    log(rate) = beta_0 + beta_1 * log(prev_ICI)_c + RE
    Predicted slope: negative (short prev_ICI -> more chirps).

C2 — add current chase duration (competition-intensity coupling).
    log(rate) = beta_0 + beta_1 * log(prev_ICI)_c
                      + beta_2 * log(D)_c + RE
    Predicted: beta_1 remains negative, beta_2 positive (longer chases
    -> higher chirp rates beyond exposure effect).

We also fit a zero-inflated NegBin variant of C2 to check whether the
standard NegBin absorbs the zero-heavy chirp distribution, and we
refit C2 on the full-width subset as a sensitivity check.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import PROCESSED

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
IN_DATA = (
    PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data"
    / "chirp_effect_on_dominant_obs.csv"
)
OUT_DIR = PROCESSED / "chirp_coupling" / "negbin"
TRACE_DIR = OUT_DIR
OUT_CSV = OUT_DIR / "slope_summary.csv"
LOO_MODELS = ("C0_intercept", "C1_prev_ICI", "C2_prev_ICI_and_D")
# A window counts as full width when it is at most this much shorter than
# the offset window (right - left).
FULL_WIDTH_TOLERANCE_S = 0.1

USE_CACHE = os.environ.get("USE_CACHE", "0") == "1"  # load cached .nc traces when available

SAMPLER_KWARGS = dict(
    draws=1000,
    tune=2000,
    chains=8,
    cores=8,
    target_accept=0.99,
    random_seed=42,
)

# Priors (weakly informative, matched to the chasing-effect script).
PRIORS_COMMON = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=5),
    "log_time_since_previous_chasing_c": bmb.Prior("Normal", mu=0, sigma=2),
    "log_chasing_duration_c": bmb.Prior("Normal", mu=0, sigma=2),
    # Dispersion (alpha in NegBin). Half-Normal keeps it positive; sigma=5
    # is loose enough that the data drive the estimate.
    "alpha": bmb.Prior("HalfNormal", sigma=5),
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def prepare_data(path: Path = IN_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Rename so offset/predictor names are short and self-explanatory.
    df = df.rename(
        columns={
            "n_loser_chirps_offset_window": "chirp_count",
            "offset_window_duration": "window_width",
            "time_since_previous_chasing": "prev_ICI",
            "chasing_duration": "D",
            "loser_id": "loser",
            "winner_id": "winner",
        }
    )

    # Drop rows with missing core fields. prev_ICI is NaN on the first chase
    # per recording (by construction, no prior chase to reference).
    keep_cols = ["chirp_count", "window_width", "prev_ICI", "D", "loser", "winner"]
    n_before = len(df)
    df = df.dropna(subset=keep_cols).copy()
    print(f"Rows before/after NaN drop: {n_before} -> {len(df)}")

    # Guard: log-exposure requires strictly positive widths.
    bad_width = (df["window_width"] <= 0).sum()
    if bad_width:
        print(f"Dropping {bad_width} rows with non-positive window_width.")
        df = df[df["window_width"] > 0].copy()

    # Transforms.
    df["log_exposure"] = np.log(df["window_width"])
    df["log_prev_ICI"] = np.log(df["prev_ICI"])
    df["log_D"] = np.log(df["D"])

    # Mean-centre the log covariates so the intercept is the rate at the
    # average log(prev_ICI) and log(D), and so the slopes are interpretable
    # per log-unit deviation from the mean.
    df["log_time_since_previous_chasing_c"] = (
        df["log_prev_ICI"] - df["log_prev_ICI"].mean()
    )
    df["log_chasing_duration_c"] = df["log_D"] - df["log_D"].mean()

    # Cast IDs to category so Bambi treats them as grouping factors.
    df["loser"] = df["loser"].astype("category")
    df["winner"] = df["winner"].astype("category")

    print(
        f"N rows: {len(df)}  |  "
        f"n losers: {df['loser'].nunique()}  |  "
        f"n winners: {df['winner'].nunique()}  |  "
        f"n recordings: {df['recording'].nunique()}"
    )
    print(
        f"Zero-chirp rows: {(df['chirp_count'] == 0).sum()} "
        f"({(df['chirp_count'] == 0).mean():.1%})"
    )
    print(f"Max chirp count: {int(df['chirp_count'].max())}")

    return df


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------
# RE_TERMS = "(1|loser) + (1|winner)"
RE_TERMS = "(1|recording)"
OFFSET = "offset(log_exposure)"  # Bambi exposes offset() on the RHS.

MODEL_SPECS = [
    {
        "name": "C0_intercept",
        "formula": f"chirp_count ~ 1 + {OFFSET} + {RE_TERMS}",
        "family": "negativebinomial",
    },
    {
        "name": "C1_prev_ICI",
        "formula": (
            f"chirp_count ~ 1 + log_time_since_previous_chasing_c "
            f"+ {OFFSET} + {RE_TERMS}"
        ),
        "family": "negativebinomial",
    },
    {
        "name": "C2_prev_ICI_and_D",
        "formula": (
            f"chirp_count ~ 1 + log_time_since_previous_chasing_c "
            f"+ log_chasing_duration_c + {OFFSET} + {RE_TERMS}"
        ),
        "family": "negativebinomial",
    },
    # Zero-inflation check on C2.
    {
        "name": "C2_zinb_check",
        "formula": (
            f"chirp_count ~ 1 + log_time_since_previous_chasing_c "
            f"+ log_chasing_duration_c + {OFFSET} + {RE_TERMS}"
        ),
        "family": "zero_inflated_negativebinomial",
    },
]


# ---------------------------------------------------------------------------
# Fit one model
# ---------------------------------------------------------------------------
def fit_model(
    spec: dict,
    df: pd.DataFrame,
    cache_dir: Path,
    use_cache: bool = USE_CACHE,
) -> az.InferenceData:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{spec['name']}.nc"

    if use_cache and cache_path.exists():
        print(f"[cache]  {spec['name']}")
        return az.from_netcdf(cache_path)

    print(f"[fit]    {spec['name']}")
    # Restrict priors to terms that actually appear in this model.
    priors = {
        k: v
        for k, v in PRIORS_COMMON.items()
        if k in ("Intercept", "alpha") or k in spec["formula"]
    }

    model = bmb.Model(
        formula=spec["formula"],
        data=df,
        family=spec["family"],
        priors=priors,
    )
    idata = model.fit(**SAMPLER_KWARGS, idata_kwargs={"log_likelihood": True})

    az.to_netcdf(idata, cache_path)
    return idata


# ---------------------------------------------------------------------------
# Summarise one fit
# ---------------------------------------------------------------------------
RATE_MULT_PARAMS = [
    "log_time_since_previous_chasing_c",
    "log_chasing_duration_c",
]


def summarise_fit(name: str, idata: az.InferenceData) -> pd.DataFrame:
    post = idata.posterior
    rows = []

    # Intercept on the rate scale (per-second rate at mean prev_ICI and D).
    b0 = post["Intercept"].values.ravel()
    rows.append(
        {
            "model": name,
            "parameter": "Intercept",
            "scale": "log_rate_per_sec",
            "mean": b0.mean(),
            "hdi_low": az.hdi(b0, hdi_prob=0.95)[0],
            "hdi_high": az.hdi(b0, hdi_prob=0.95)[1],
            "mult_median": np.median(np.exp(b0)),
            "mult_low": az.hdi(np.exp(b0), hdi_prob=0.95)[0],
            "mult_high": az.hdi(np.exp(b0), hdi_prob=0.95)[1],
            "p_neg": (b0 < 0).mean(),
        }
    )

    # Slopes that exist in this model.
    for param in RATE_MULT_PARAMS:
        if param not in post:
            continue
        draws = post[param].values.ravel()
        hdi = az.hdi(draws, hdi_prob=0.95)
        mult = np.exp(draws)
        rows.append(
            {
                "model": name,
                "parameter": param,
                "scale": "log_slope",
                "mean": draws.mean(),
                "hdi_low": hdi[0],
                "hdi_high": hdi[1],
                "mult_median": np.median(mult),
                "mult_low": az.hdi(mult, hdi_prob=0.95)[0],
                "mult_high": az.hdi(mult, hdi_prob=0.95)[1],
                "p_neg": (draws < 0).mean(),
            }
        )

    # Dispersion.
    if "alpha" in post:
        alpha = post["alpha"].values.ravel()
        rows.append(
            {
                "model": name,
                "parameter": "alpha",
                "scale": "dispersion",
                "mean": alpha.mean(),
                "hdi_low": az.hdi(alpha, hdi_prob=0.95)[0],
                "hdi_high": az.hdi(alpha, hdi_prob=0.95)[1],
                "mult_median": np.nan,
                "mult_low": np.nan,
                "mult_high": np.nan,
                "p_neg": np.nan,
            }
        )

    return pd.DataFrame(rows)


def print_summary(name: str, summary: pd.DataFrame) -> None:
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    print(
        f"{'parameter':<42} {'beta':>8}  {'95% HDI':>20}  "
        f"{'× rate':>8}  {'95% CrI':>18}  {'P(b<0)':>8}"
    )
    print("-" * 120)
    for _, r in summary.iterrows():
        hdi = f"[{r['hdi_low']:+.3f}, {r['hdi_high']:+.3f}]"
        if r["parameter"] == "alpha":
            print(
                f"{r['parameter']:<42} {r['mean']:>8.3f}  "
                f"{hdi:>20}  {'--':>8}  {'--':>18}  {'--':>8}"
            )
        else:
            mult = f"{r['mult_median']:.3f}"
            cri = f"[{r['mult_low']:.3f}, {r['mult_high']:.3f}]"
            p_neg_s = f"{r['p_neg']:.1%}" if not np.isnan(r["p_neg"]) else "--"
            print(
                f"{r['parameter']:<42} {r['mean']:>+8.3f}  "
                f"{hdi:>20}  {mult:>8}  {cri:>18}  {p_neg_s:>8}"
            )


# ---------------------------------------------------------------------------
# Posterior predictive zero-count check (diagnostic for NegBin vs ZINB)
# ---------------------------------------------------------------------------
def zero_count_ppc(
    idata: az.InferenceData,
    df: pd.DataFrame,
    name: str,
) -> None:
    """Compare observed zero count against the posterior predictive distribution.

    Reports the observed fraction of zeros alongside the posterior predictive
    fraction. If the observed fraction is in the extreme tail, NegBin is
    under-fitting the zeros and a ZINB variant is warranted.
    """
    if "posterior_predictive" not in idata.groups():
        print(
            f"[ppc]    {name}: no posterior_predictive group; "
            "re-fit with idata_kwargs={'include_posterior_predictive': True}."
        )
        return

    obs_zero_frac = (df["chirp_count"] == 0).mean()
    y_pp = idata.posterior_predictive["chirp_count"].values
    pp_zero_frac = (y_pp == 0).mean(axis=(-1,))

    lo, hi = np.quantile(pp_zero_frac, [0.025, 0.975])
    print(f"\n[zero PPC] {name}:")
    print(f"  observed zero frac:   {obs_zero_frac:.3f}")
    print(
        f"  predicted zero frac:  median {np.median(pp_zero_frac):.3f}, "
        f"95% [{lo:.3f}, {hi:.3f}]"
    )
    if obs_zero_frac < lo or obs_zero_frac > hi:
        print("  WARNING: observed outside 95% predictive range — consider ZINB.")
    else:
        print("  OK: observed inside 95% predictive range.")


# ---------------------------------------------------------------------------
# Sensitivity refit on full-width windows
# ---------------------------------------------------------------------------
def full_width(df: pd.DataFrame) -> pd.Series:
    """Rows whose window is effectively full: the whole offset window, shortened
    neither by a short chase nor by the next chase."""
    left, right = offset_window()
    return df["window_width"] >= (right - left) - FULL_WIDTH_TOLERANCE_S


def refit_c2_full_width(
    df: pd.DataFrame,
    cache_dir: Path,
) -> pd.DataFrame:
    """C2 restricted to rows where window_width is effectively full."""
    full = df[full_width(df)].copy()
    print(
        f"\n[sensitivity] C2 on full-width subset: "
        f"{len(full)}/{len(df)} rows ({len(full) / len(df):.1%})"
    )

    spec = {
        "name": "C2_full_width_only",
        "formula": (
            f"chirp_count ~ 1 + log_time_since_previous_chasing_c "
            f"+ log_chasing_duration_c + {OFFSET} + {RE_TERMS}"
        ),
        "family": "negativebinomial",
    }
    idata = fit_model(spec, full, cache_dir=cache_dir)
    summary = summarise_fit(spec["name"], idata)
    print_summary(spec["name"], summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = prepare_data()
    all_summaries: list[pd.DataFrame] = []
    nested: dict[str, az.InferenceData] = {}
    counts = df["chirp_count"]
    stats: dict = {
        "data": {
            "n_rows": len(df),
            "n_losers": df["loser"].nunique(),
            "n_winners": df["winner"].nunique(),
            "n_recordings": df["recording"].nunique(),
            "zero_fraction": (counts == 0).mean(),
            "max_count": counts.max(),
            "variance_to_mean": counts.var() / counts.mean(),
        },
        "divergences": {},
    }

    for spec in MODEL_SPECS:
        idata = fit_model(spec, df, cache_dir=TRACE_DIR)
        if spec["name"] in LOO_MODELS:
            nested[spec["name"]] = idata
        summary = summarise_fit(spec["name"], idata)
        print_summary(spec["name"], summary)
        all_summaries.append(summary)
        stats["divergences"][spec["name"]] = int(idata.sample_stats["diverging"].sum())

        # Sampler diagnostics.
        try:
            div = int(idata.sample_stats["diverging"].sum())
            summ = az.summary(
                idata, var_names=["~loser", "~winner"], filter_vars="like"
            )
            max_rhat = summ["r_hat"].max() if "r_hat" in summ else np.nan
            min_ess = summ["ess_bulk"].min() if "ess_bulk" in summ else np.nan
            print(
                f"  diagnostics: div={div}, "
                f"max r-hat={max_rhat:.3f}, min ess_bulk={min_ess:.0f}"
            )
        except Exception as e:
            print(f"  (diagnostics skipped: {e})")

    # Leave-one-out comparison of the nested models C0-C2.
    loo_table = az.compare(nested, ic="loo")
    loo_table.to_csv(OUT_DIR / "model_comparison_loo.csv")
    print("\nLOO comparison (ELPD; differences relative to the best model):")
    print(loo_table[["elpd_loo", "elpd_diff", "dse", "weight", "warning"]])
    stats["loo"] = {name: {"elpd": row.elpd_loo, "weight": row.weight}
                    for name, row in loo_table.iterrows()}
    for better, worse in (("C2_prev_ICI_and_D", "C1_prev_ICI"), ("C1_prev_ICI", "C0_intercept")):
        la = az.loo(nested[better], pointwise=True)
        lb = az.loo(nested[worse], pointwise=True)
        d = la.loo_i.values - lb.loo_i.values
        print(f"  {better} - {worse}: dELPD = {d.sum():.1f}, SE = {np.sqrt(len(d)) * d.std():.1f}")
        stats["loo"][f"{better}_minus_{worse}"] = {"delpd": d.sum(),
                                                    "se": np.sqrt(len(d)) * d.std()}
    for name, idata in nested.items():
        k = az.loo(idata, pointwise=True).pareto_k.values
        stats["loo"][name]["n_pareto_k_above_0_7"] = int((k > 0.7).sum())
        print(f"  {name}: {int((k > 0.7).sum())} observations with Pareto k > 0.7 (max {k.max():.2f})")

    # Sensitivity refit.
    sens = refit_c2_full_width(df, cache_dir=TRACE_DIR)
    all_summaries.append(sens)
    n_full = full_width(df).sum()  # the rows of refit_c2_full_width
    stats["full_width_subset"] = {"n_rows": n_full, "fraction": n_full / len(df)}
    write_stats("coupling", stats)

    # Persist the combined summary table.
    out = pd.concat(all_summaries, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # High-level interpretation pointer.
    print("\nInterpretation key:")
    print("  log_time_since_previous_chasing_c slope:")
    print("    negative -> short prev ICI -> more chirps (bout-density coupling)")
    print("  log_chasing_duration_c slope:")
    print(
        "    positive -> longer chases -> more chirps above exposure "
        "(intensity coupling)"
    )


if __name__ == "__main__":
    main()
