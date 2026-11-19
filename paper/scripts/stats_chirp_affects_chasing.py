# Ported from chirp_competition d040123 (src_paper/stats_chirp_affects_chasing.py).
# Changes: input/output under $PAPER_OUT/data/processed. Models, priors, subsets, sampler settings and seed unchanged.
# Observations and divergences per fit go to write_stats ("receiver_effect_fits"). The window is the offset
# window of offset_window.py (was the constant -3, +13 s): predictor loser_chirped_offset_window and column
# offset_window_duration (were loser_chirped_fixed_m3p13 and fixed_window_duration).
"""
Bayesian Gamma GLMMs: does loser chirp presence in the offset window
[offset + left, offset + right] around a chasing offset
(``offset_window.py``) change the dominant's chasing dynamics?

Primary analysis (M1--M5b)
--------------------------
  M1  causal             : time_to_next_chasing        ~ chirped + RE
  M2  acausal (control)  : time_since_previous_chasing ~ chirped + RE
  M3  causal_adj_prev    : time_to_next_chasing        ~ chirped
                           + log(tsp)_c + RE
  M4  causal_adj_full    : time_to_next_chasing        ~ chirped
                           + log(tsp)_c + log(D)_c + RE
  M5  duration_from_chirp: chasing_duration            ~ chirped + RE
  M5b duration_from_chirp_adj
                         : chasing_duration            ~ chirped
                           + log(tsp)_c + RE

Window decomposition (Mtail, Mpost_*, Mjoint_*)
-----------------------------------------------
Splits `loser_chirped_offset_window` into a chase-tail half
([max(offset + left, onset), offset]) and a post-offset half
([offset, min(offset + right, next_onset)] or, unclipped, [offset, offset + right]).
Diagnoses whether the M4 chirp slope is driven by:
  - chirps during the last -left s of the chase (chase_tail), or
  - chirps after the chase (post_offset), or
  - a windowing-response coupling in which short next-ICIs mechanically
    shrink the post_offset window and produce an apparent positive
    slope (clipped vs. unclipped contrast).

  Mtail              : time_to_next_chasing ~ chirped_chase_tail
                       + log(tsp)_c + log(D)_c + RE
  Mpost_clipped      : time_to_next_chasing ~ chirped_post_offset_clipped
                       + log(tsp)_c + log(D)_c + RE
  Mpost_unclipped    : time_to_next_chasing ~ chirped_post_offset_unclipped
                       + log(tsp)_c + log(D)_c + RE
  Mjoint_clipped     : time_to_next_chasing ~ chirped_chase_tail
                       + chirped_post_offset_clipped
                       + log(tsp)_c + log(D)_c + RE
  Mjoint_unclipped   : time_to_next_chasing ~ chirped_chase_tail
                       + chirped_post_offset_unclipped
                       + log(tsp)_c + log(D)_c + RE

Decomposition runs only on primary_actor_partner x full; the windowing
question is independent of subset/RE sensitivity, which was already
characterised in the primary analysis.

Sensitivity design (primary analysis only)
------------------------------------------
  data subsets  : full, untruncated, matched
  RE structures : primary_actor_partner (1|loser)+(1|winner),
                  sensitivity_dyad (1|recording),
                  fallback_loser_only (1|loser)

Family         : Gamma with log link
Priors         : Intercept ~ Normal(0, 5); slopes ~ Normal(0, 2)
"""

from __future__ import annotations

import os
from pathlib import Path
from IPython import embed

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import PROCESSED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = (
    PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "data"
    / "chirp_effect_on_dominant_obs.csv"
)
OUT_DIR = PROCESSED / "chirp_effect_on_dominant_adaptive_window" / "gamma_glmm_binary"

PRIMARY_PREDICTOR = "loser_chirped_offset_window"
TAIL_PREDICTOR = "loser_chirped_chase_tail"
POST_CLIPPED_PREDICTOR = "loser_chirped_post_offset_clipped"
POST_UNCLIPPED_PREDICTOR = "loser_chirped_post_offset_unclipped"

PREV_COVARIATE = "log_time_since_previous_chasing"
PREV_COVARIATE_C = PREV_COVARIATE + "_c"
DUR_COVARIATE = "log_chasing_duration"
DUR_COVARIATE_C = DUR_COVARIATE + "_c"

ALL_RE_COLUMNS = ["loser_id", "winner_id", "recording"]

RE_STRUCTURES: dict[str, list[str]] = {
    "primary_recording": ["recording"],
    "sensitivity_loserid_winnerid_recording": ["loser_id", "winner_id", "recording"],
    "fallback_loser_only": ["loser_id"],
}
PRIMARY_RE = "primary_recording"

RE_SENSITIVITY_SUBSETS = ["full", "untruncated", "matched"]

SAMPLER_KWARGS = dict(
    tune=3000,
    draws=1000,
    chains=8,
    cores=8,
    random_seed=42,
    target_accept=0.99,
)
MODEL_KWARGS = dict(
    family="gamma",
    link="log",
    dropna=True,
)
PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=5),
    "common": bmb.Prior("Normal", mu=0, sigma=2),
}

MIN_N_FIT = 50

# If True, any fit whose InferenceData `.nc` is already on disk is
# loaded instead of re-sampled. The cache key is the filename
# `{re}__{subset}__{model}.nc`, so a fit is only considered stale when
# you change its RE structure, subset filter, model spec, or delete
# the file. Changing priors, sampler kwargs, or the data source will
# NOT invalidate the cache; set USE_CACHE = False for those cases.
USE_CACHE = os.environ.get("USE_CACHE", "0") == "1"


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------
def primary_specs() -> list[dict]:
    return [
        {
            "name": "M1_causal",
            "role": "causal",
            "response": "time_to_next_chasing",
            "predictors": [PRIMARY_PREDICTOR],
            "covariates": [],
        },
        {
            "name": "M2_acausal",
            "role": "acausal",
            "response": "time_since_previous_chasing",
            "predictors": [PRIMARY_PREDICTOR],
            "covariates": [],
        },
        {
            "name": "M3_causal_adj_prev",
            "role": "causal (+tsp)",
            "response": "time_to_next_chasing",
            "predictors": [PRIMARY_PREDICTOR],
            "covariates": [PREV_COVARIATE_C],
        },
        {
            "name": "M4_causal_adj_full",
            "role": "causal (+tsp,+D)",
            "response": "time_to_next_chasing",
            "predictors": [PRIMARY_PREDICTOR],
            "covariates": [PREV_COVARIATE_C, DUR_COVARIATE_C],
        },
        {
            "name": "M5_duration_from_chirp",
            "role": "chirp -> D",
            "response": "chasing_duration",
            "predictors": [PRIMARY_PREDICTOR],
            "covariates": [],
        },
        {
            "name": "M5b_duration_from_chirp_adj",
            "role": "chirp -> D (+tsp)",
            "response": "chasing_duration",
            "predictors": [PRIMARY_PREDICTOR],
            "covariates": [PREV_COVARIATE_C],
        },
    ]


def decomposition_specs(window: tuple[float, float]) -> list[dict]:
    _, right = window
    base_covs = [PREV_COVARIATE_C, DUR_COVARIATE_C]
    return [
        {
            "name": "Mtail",
            "role": "tail-only (D-coupled)",
            "response": "time_to_next_chasing",
            "predictors": [TAIL_PREDICTOR],
            "covariates": base_covs,
        },
        {
            "name": "Mpost_clipped",
            "role": "post-only (clip: next-ICI-coupled)",
            "response": "time_to_next_chasing",
            "predictors": [POST_CLIPPED_PREDICTOR],
            "covariates": base_covs,
        },
        {
            "name": "Mpost_unclipped",
            "role": f"post-only (unclip: {right:g} s fixed)",
            "response": "time_to_next_chasing",
            "predictors": [POST_UNCLIPPED_PREDICTOR],
            "covariates": base_covs,
        },
        {
            "name": "Mjoint_clipped",
            "role": "tail + post (clipped)",
            "response": "time_to_next_chasing",
            "predictors": [TAIL_PREDICTOR, POST_CLIPPED_PREDICTOR],
            "covariates": base_covs,
        },
        {
            "name": "Mjoint_unclipped",
            "role": "tail + post (unclipped)",
            "response": "time_to_next_chasing",
            "predictors": [TAIL_PREDICTOR, POST_UNCLIPPED_PREDICTOR],
            "covariates": base_covs,
        },
    ]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_and_prepare(
    path: Path, specs: list[dict], window: tuple[float, float]
) -> pd.DataFrame:
    df = pd.read_csv(path)

    all_predictors = {p for s in specs for p in s["predictors"]}
    required = (
        list(all_predictors)
        + [
            "time_to_next_chasing",
            "time_since_previous_chasing",
            "chasing_duration",
            "offset_window_duration",
        ]
        + ALL_RE_COLUMNS
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from {path.name}: {missing}.")

    for p in all_predictors:
        df[p] = df[p].astype(bool).astype(int)

    tsp = df["time_since_previous_chasing"]
    valid_tsp = (tsp > 0) & tsp.notna()
    df[PREV_COVARIATE] = np.where(valid_tsp, np.log(tsp.where(valid_tsp)), np.nan)
    df[PREV_COVARIATE_C] = df[PREV_COVARIATE] - np.nanmean(df[PREV_COVARIATE])

    cd = df["chasing_duration"]
    valid_cd = (cd > 0) & cd.notna()
    df[DUR_COVARIATE] = np.where(valid_cd, np.log(cd.where(valid_cd)), np.nan)
    df[DUR_COVARIATE_C] = df[DUR_COVARIATE] - np.nanmean(df[DUR_COVARIATE])

    for r in ALL_RE_COLUMNS:
        df[r] = df[r].astype("category")

    _print_data_summary(df, all_predictors, window)

    # df.groupby("loser_id")["recording"].nunique().value_counts().sort_index()
    # embed()
    # exit()

    return df


def _print_data_summary(
    df: pd.DataFrame, all_predictors: set[str], window: tuple[float, float]
) -> None:
    left, right = window
    full_duration = right - left
    print("\n--- data summary ---")
    print(f"Total rows:       {len(df)}")
    print(f"Recordings:       {df['recording'].nunique()}")
    print(f"Unique loser_id:  {df['loser_id'].nunique()}")
    print(f"Unique winner_id: {df['winner_id'].nunique()}")

    print("\nPredictor prevalence:")
    for p in sorted(all_predictors):
        n1 = int((df[p] == 1).sum())
        print(f"  {p:<44}  n(=1) = {n1:>5d}  ({100 * n1 / len(df):>5.1f}%)")

    fwd = df["offset_window_duration"]
    n_full = int((fwd >= full_duration - 1e-6).sum())
    print(
        f"\noffset_window_duration: full-duration rows "
        f"(~{full_duration:.0f} s): "
        f"{n_full} of {len(df)} ({100 * n_full / len(df):.1f}%)"
    )

    cd_vals = df["chasing_duration"].dropna()
    print(
        f"\nChasing duration: median={cd_vals.median():.2f} s, "
        f"mean={cd_vals.mean():.2f} s, max={cd_vals.max():.2f} s; "
        f"rows with D < {-left:.0f} s: "
        f"{int((cd_vals < -left).sum())}"
    )


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------
def subset_masks(df: pd.DataFrame, window: tuple[float, float]) -> dict[str, pd.Series]:
    left, right = window
    ones = pd.Series(True, index=df.index)
    untruncated = df["offset_window_duration"] >= (right - left) - 1e-6
    matched = (
        df["time_to_next_chasing"].notna()
        & (df["time_to_next_chasing"] > 0)
        & df["time_since_previous_chasing"].notna()
        & (df["time_since_previous_chasing"] > 0)
    )
    return {"full": ones, "untruncated": untruncated, "matched": matched}


def usable_rows(
    df: pd.DataFrame, spec: dict, base_mask: pd.Series, re_groups: list[str]
) -> pd.Series:
    mask = base_mask.copy()
    for p in spec["predictors"]:
        mask &= df[p].notna()
    resp = spec["response"]
    mask &= df[resp].notna() & (df[resp] > 0)
    for c in spec["covariates"]:
        mask &= df[c].notna()
    for r in re_groups:
        mask &= df[r].notna()
    return mask


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def build_formula(spec: dict, re_groups: list[str]) -> str:
    fixed = " + ".join(spec["predictors"] + spec["covariates"])
    re_terms = " + ".join(f"(1|{g})" for g in re_groups)
    return f"{spec['response']} ~ {fixed} + {re_terms}"


def fit_one(df: pd.DataFrame, spec: dict, mask: pd.Series, re_groups: list[str]):
    cols = [spec["response"]] + spec["predictors"] + spec["covariates"] + re_groups
    sub = df.loc[mask, cols].copy()
    formula = build_formula(spec, re_groups)
    print(f"    formula: {formula}")
    print(f"    n_fit:   {len(sub)}")
    model = bmb.Model(formula, data=sub, priors=PRIORS, **MODEL_KWARGS)
    trace = model.fit(**SAMPLER_KWARGS)
    return model, trace, len(sub)


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------
def _find_sigma_var(trace, grp: str) -> str | None:
    """Locate the posterior variable holding the group-level SD for RE term `grp`.

    Bambi convention is "1|{grp}_sigma"; older/newer versions occasionally
    drop the "1|" prefix. Falls back to a substring search on _sigma vars.
    """
    for cand in (f"1|{grp}_sigma", f"{grp}_sigma"):
        if cand in trace.posterior.data_vars:
            return cand
    for v in trace.posterior.data_vars:
        if v.endswith("_sigma") and grp in v:
            return v
    return None


def summarise_variance_components(
    trace,
    spec: dict,
    subset_label: str,
    re_label: str,
    re_groups: list[str],
    n_fit: int,
    df_fit_re: pd.DataFrame,
) -> list[dict]:
    """One row per RE term: posterior of sigma on the linear-predictor scale.

    `df_fit_re` is the slice of the data used for this fit, restricted to
    the RE columns, so we can report the number of group levels actually
    observed in the fit (not just declared in the categorical).
    """
    rows = []
    for grp in re_groups:
        var = _find_sigma_var(trace, grp)
        if var is None:
            print(f"    [warn] no sigma found for RE term '{grp}'; skipping.")
            continue
        s = _flatten(trace.posterior[var].values)
        hdi = az.hdi(s, hdi_prob=0.95)
        rows.append(
            {
                "model": spec["name"],
                "subset": subset_label,
                "re": re_label,
                "re_term": grp,
                "sigma_var_name": var,
                "n_fit": n_fit,
                "n_levels": int(df_fit_re[grp].nunique()),
                "sigma_median": float(np.median(s)),
                "sigma_mean": float(s.mean()),
                "sigma_hdi95_low": float(hdi[0]),
                "sigma_hdi95_high": float(hdi[1]),
                # Heuristic: how much posterior mass piles up near zero.
                # Useful as a single-number "this RE is collapsing" indicator.
                "sigma_p_below_0p1": float((s < 0.1).mean()),
                "sigma_p_below_0p05": float((s < 0.05).mean()),
            }
        )
    return rows


def _flatten(arr) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def summarise_predictor(
    trace,
    spec: dict,
    predictor: str,
    subset_label: str,
    re_label: str,
    n_fit: int,
) -> dict:
    beta = _flatten(trace.posterior[predictor].values)
    exp_beta = np.exp(beta)
    hdi_log = az.hdi(beta, hdi_prob=0.95)
    hdi_exp = az.hdi(exp_beta, hdi_prob=0.95)

    out = {
        "model": spec["name"],
        "subset": subset_label,
        "re": re_label,
        "role": spec["role"],
        "response": spec["response"],
        "predictor": predictor,
        "covariates": ",".join(spec["covariates"]) or "-",
        "other_predictors": ",".join(p for p in spec["predictors"] if p != predictor)
        or "-",
        "n_fit": n_fit,
        "beta_mean": float(beta.mean()),
        "beta_hdi95_low": float(hdi_log[0]),
        "beta_hdi95_high": float(hdi_log[1]),
        "rr_median": float(np.median(exp_beta)),
        "rr_hdi95_low": float(hdi_exp[0]),
        "rr_hdi95_high": float(hdi_exp[1]),
        "prob_beta_lt_0": float((beta < 0).mean()),
        "prob_beta_gt_0": float((beta > 0).mean()),
    }

    for cov in spec["covariates"]:
        if cov not in trace.posterior:
            continue
        b = _flatten(trace.posterior[cov].values)
        q = az.hdi(b, hdi_prob=0.95)
        out[f"{cov}_beta_median"] = float(np.median(b))
        out[f"{cov}_beta_hdi95_low"] = float(q[0])
        out[f"{cov}_beta_hdi95_high"] = float(q[1])

    return out


def diagnostics(trace) -> dict:
    fixed_vars = [
        v
        for v in trace.posterior.data_vars
        if "|" not in v and not v.endswith("_sigma")
    ]
    summ = az.summary(trace, var_names=fixed_vars, kind="diagnostics")
    div = int(trace.sample_stats["diverging"].sum())
    return {
        "divergent": div,
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _fit_or_load(
    df: pd.DataFrame,
    spec: dict,
    mask: pd.Series,
    re_groups: list[str],
    re_label: str,
    subset_label: str,
    out_dir: Path,
):
    """Load a cached InferenceData if present; otherwise fit and save.

    Cache key: `{re_label}__{subset_label}__{spec['name']}.nc`.
    Invalidation is manual: delete the file, or set USE_CACHE = False,
    for anything that changes priors, sampler kwargs, or model structure
    without changing the cache key.
    """
    stem = out_dir / f"{re_label}__{subset_label}__{spec['name']}"
    nc_path = stem.with_suffix(".nc")

    if USE_CACHE and nc_path.exists():
        print(f"    [cached] loading {nc_path.name}")
        trace = az.from_netcdf(nc_path)
        return None, trace, int(mask.sum())

    model, trace, n_fit = fit_one(df, spec, mask, re_groups)
    az.to_netcdf(trace, nc_path)
    return model, trace, n_fit


def run_one(
    df: pd.DataFrame,
    spec: dict,
    subset_label: str,
    base_mask: pd.Series,
    re_label: str,
    re_groups: list[str],
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    mask = usable_rows(df, spec, base_mask, re_groups)
    n = int(mask.sum())
    print(f"\n  [{re_label}][{subset_label}] {spec['name']} ({spec['response']})")
    if n < MIN_N_FIT:
        print(f"    n={n} < MIN_N_FIT={MIN_N_FIT}; skipping.")
        return [], []

    _, trace, n_fit = _fit_or_load(
        df, spec, mask, re_groups, re_label, subset_label, out_dir
    )
    diag = diagnostics(trace)
    flag = (
        diag["divergent"] > 0
        or diag["max_rhat"] > 1.01
        or diag["min_ess_bulk"] < 400
        or diag["min_ess_tail"] < 400
    )
    tag = "[!]" if flag else "[ok]"
    print(
        f"    {tag} div={diag['divergent']}, "
        f"max R-hat={diag['max_rhat']:.3f}, "
        f"min ESS bulk/tail = "
        f"{diag['min_ess_bulk']:.0f}/{diag['min_ess_tail']:.0f}"
    )

    slope_rows = []
    for pred in spec["predictors"]:
        summary = summarise_predictor(trace, spec, pred, subset_label, re_label, n_fit)
        summary.update({f"diag_{k}": v for k, v in diag.items()})
        slope_rows.append(summary)

    variance_rows = summarise_variance_components(
        trace,
        spec,
        subset_label,
        re_label,
        re_groups,
        n_fit,
        df_fit_re=df.loc[mask, re_groups],
    )
    for r in variance_rows:
        r.update({f"diag_{k}": v for k, v in diag.items()})

    return slope_rows, variance_rows


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------
def _print_row(r: dict, fields: list[tuple[str, str, int]]) -> None:
    parts = []
    for key, kind, width in fields:
        v = r[key]
        if kind == "str":
            parts.append(f"{str(v)[:width]:<{width}}")
        elif kind == "int":
            parts.append(f"{v:>{width}d}")
        elif kind == "sign":
            parts.append(f"{v:>+{width}.3f}")
        elif kind == "rr":
            parts.append(f"{v:>{width}.2f}")
        elif kind == "pct":
            parts.append(f"{v:>{width}.1%}")
    print("  ".join(parts))


def print_primary_table(results: list[dict]) -> None:
    rows = [
        r
        for r in results
        if r["re"] == PRIMARY_RE and r["model"] in {s["name"] for s in primary_specs()}
    ]
    if not rows:
        return
    print("\n" + "=" * 132)
    print(f"Primary analysis — RE = {PRIMARY_RE}")
    print("=" * 132)
    print(
        f"{'model':<30}  {'subset':<12}  {'role':<18}  "
        f"{'n':>5}  {'beta':>7}  {'95% HDI (log)':>18}  "
        f"{'RR':>5}  {'95% HDI (RR)':>14}  {'P(b<0)':>7}"
    )
    print("-" * 132)
    for r in rows:
        print(
            f"{r['model']:<30}  {r['subset']:<12}  {r['role'][:18]:<18}  "
            f"{r['n_fit']:>5d}  {r['beta_mean']:>+7.3f}  "
            f"[{r['beta_hdi95_low']:>+7.3f}, {r['beta_hdi95_high']:>+7.3f}]  "
            f"{r['rr_median']:>5.2f}  "
            f"[{r['rr_hdi95_low']:>4.2f}, {r['rr_hdi95_high']:>4.2f}]  "
            f"{r['prob_beta_lt_0']:>6.1%}"
        )
    print("=" * 132)


def print_re_sensitivity_table(results: list[dict], subset: str) -> None:
    rows = [
        r
        for r in results
        if r["subset"] == subset and r["model"] in {s["name"] for s in primary_specs()}
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r["model"], r["re"]))
    print("\n" + "=" * 132)
    print(f"RE sensitivity — subset = {subset}")
    print("=" * 132)
    print(
        f"{'model':<30}  {'RE structure':<24}  "
        f"{'n':>5}  {'beta':>7}  {'95% HDI (log)':>18}  "
        f"{'RR':>5}  {'95% HDI (RR)':>14}  {'P(b<0)':>7}"
    )
    print("-" * 132)
    last_model = None
    for r in rows:
        if last_model is not None and r["model"] != last_model:
            print("-" * 132)
        print(
            f"{r['model']:<30}  {r['re'][:24]:<24}  "
            f"{r['n_fit']:>5d}  {r['beta_mean']:>+7.3f}  "
            f"[{r['beta_hdi95_low']:>+7.3f}, {r['beta_hdi95_high']:>+7.3f}]  "
            f"{r['rr_median']:>5.2f}  "
            f"[{r['rr_hdi95_low']:>4.2f}, {r['rr_hdi95_high']:>4.2f}]  "
            f"{r['prob_beta_lt_0']:>6.1%}"
        )
        last_model = r["model"]
    print("=" * 132)


def print_decomposition_table(results: list[dict], specs: list[dict]) -> None:
    decomp_names = {s["name"] for s in specs}
    rows = [r for r in results if r["model"] in decomp_names]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r["model"], r["predictor"]))
    print("\n" + "=" * 148)
    print("Window decomposition — RE = primary_actor_partner, subset = full")
    print("=" * 148)
    print(
        f"{'model':<20}  {'predictor':<40}  {'role':<30}  "
        f"{'n':>5}  {'beta':>7}  {'95% HDI (log)':>18}  "
        f"{'RR':>5}  {'95% HDI (RR)':>14}  {'P(b<0)':>7}"
    )
    print("-" * 148)
    last_model = None
    for r in rows:
        if last_model is not None and r["model"] != last_model:
            print("-" * 148)
        print(
            f"{r['model']:<20}  {r['predictor'][:40]:<40}  "
            f"{r['role'][:30]:<30}  "
            f"{r['n_fit']:>5d}  {r['beta_mean']:>+7.3f}  "
            f"[{r['beta_hdi95_low']:>+7.3f}, {r['beta_hdi95_high']:>+7.3f}]  "
            f"{r['rr_median']:>5.2f}  "
            f"[{r['rr_hdi95_low']:>4.2f}, {r['rr_hdi95_high']:>4.2f}]  "
            f"{r['prob_beta_lt_0']:>6.1%}"
        )
        last_model = r["model"]
    print("=" * 148)


def print_covariate_slopes(results: list[dict]) -> None:
    cov_rows = [r for r in results if r["covariates"] != "-"]
    if not cov_rows:
        return
    print("\nCovariate slopes (log link, 95% HDI):")
    seen = set()
    for r in cov_rows:
        # Covariate slopes are the same per fit regardless of which
        # predictor row we're on, so skip duplicate (fit, cov) pairs.
        fit_key = (r["re"], r["subset"], r["model"])
        for cov in r["covariates"].split(","):
            k = fit_key + (cov,)
            if k in seen:
                continue
            seen.add(k)
            key = f"{cov}_beta_median"
            if key not in r:
                continue
            print(
                f"  [{r['re']:<22}][{r['subset']:<12}] "
                f"{r['model']:<30} {cov}: "
                f"beta = {r[key]:+.3f}, "
                f"95% HDI [{r[f'{cov}_beta_hdi95_low']:+.3f}, "
                f"{r[f'{cov}_beta_hdi95_high']:+.3f}]"
            )


def print_diagnostic_report(results: list[dict]) -> None:
    # One entry per fit, deduplicated over multi-predictor specs.
    seen = set()
    fits = []
    for r in results:
        k = (r["re"], r["subset"], r["model"])
        if k in seen:
            continue
        seen.add(k)
        fits.append(r)

    bad = [
        r
        for r in fits
        if r["diag_divergent"] > 0
        or r["diag_max_rhat"] > 1.01
        or r["diag_min_ess_bulk"] < 400
        or r["diag_min_ess_tail"] < 400
    ]
    print("\n--- diagnostics summary ---")
    print(f"Total fits:    {len(fits)}")
    print(f"Fits flagged:  {len(bad)}")
    if bad:
        print("\nFlagged fits (investigate before reporting):")
        for r in bad:
            print(
                f"  [{r['re']:<22}][{r['subset']:<12}] {r['model']:<30}  "
                f"div={r['diag_divergent']}, "
                f"max R-hat={r['diag_max_rhat']:.3f}, "
                f"min ESS bulk/tail = "
                f"{r['diag_min_ess_bulk']:.0f}/"
                f"{r['diag_min_ess_tail']:.0f}"
            )


def fit_stats(results: list[dict]) -> dict:
    """Per fit (RE structure -> subset -> model): observations and divergent
    transitions, and how many fits diverged (for write_stats)."""
    fits: dict = {}
    for r in results:
        fits.setdefault(r["re"], {}).setdefault(r["subset"], {})[r["model"]] = {
            "n_fit": r["n_fit"], "divergences": r["diag_divergent"]}
    flat = [f for subsets in fits.values() for models in subsets.values() for f in models.values()]
    return {"n_fits": len(flat),
            "n_fits_with_divergences": sum(f["divergences"] > 0 for f in flat),
            "fits": fits}


def print_variance_table(variance_rows: list[dict]) -> None:
    if not variance_rows:
        return
    rows = sorted(
        variance_rows,
        key=lambda r: (r["model"], r["subset"], r["re"], r["re_term"]),
    )
    print("\n" + "=" * 140)
    print("Variance components by RE structure (sigma on linear-predictor scale)")
    print("=" * 140)
    print(
        f"{'model':<28}  {'subset':<12}  {'RE':<44}  {'term':<12}  "
        f"{'lvls':>4}  {'sigma':>6}  {'95% HDI':>16}  {'P(s<0.1)':>9}"
    )
    print("-" * 140)
    last_block = None
    for r in rows:
        block = (r["model"], r["subset"], r["re"])
        if last_block is not None and block != last_block:
            print("-" * 140)
        print(
            f"{r['model']:<28}  {r['subset']:<12}  {r['re'][:44]:<44}  "
            f"{r['re_term']:<12}  {r['n_levels']:>4d}  "
            f"{r['sigma_median']:>6.3f}  "
            f"[{r['sigma_hdi95_low']:>5.3f}, {r['sigma_hdi95_high']:>5.3f}]  "
            f"{r['sigma_p_below_0p1']:>8.1%}"
        )
        last_block = block
    print("=" * 140)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    window = offset_window()
    print(f"Offset window: {window[0]:+g} to {window[1]:+g} s around chase offsets")
    p_specs = primary_specs()
    d_specs = decomposition_specs(window)
    all_specs = p_specs + d_specs

    df = load_and_prepare(DATA_PATH, all_specs, window)
    masks = subset_masks(df, window)

    # ----- cache survey ---------------------------------------------------
    planned: list[tuple[str, str, str]] = []
    for re_label in RE_STRUCTURES:
        subsets = (
            list(masks)
            if re_label == PRIMARY_RE
            else [s for s in RE_SENSITIVITY_SUBSETS if s in masks]
        )
        for subset_label in subsets:
            for spec in p_specs:
                planned.append((re_label, subset_label, spec["name"]))
    for spec in d_specs:
        planned.append((PRIMARY_RE, "full", spec["name"]))

    if USE_CACHE:
        hits = sum(
            1
            for re_l, sub_l, name in planned
            if (OUT_DIR / f"{re_l}__{sub_l}__{name}.nc").exists()
        )
        print(
            f"\n--- cache: {hits}/{len(planned)} fits cached, "
            f"{len(planned) - hits} to run ---"
        )
    else:
        print(f"\n--- cache disabled: all {len(planned)} fits will be refit ---")

    # ----- primary analysis: RE x subset x spec grid ------------------------
    print("\n========== primary analysis ==========")
    primary_planned = len(masks) * len(p_specs) + (len(RE_STRUCTURES) - 1) * len(
        RE_SENSITIVITY_SUBSETS
    ) * len(p_specs)
    print(f"  planned fits: {primary_planned}")

    all_results: list[dict] = []
    all_variance: list[dict] = []
    for re_label, re_groups in RE_STRUCTURES.items():
        subsets_for_this_re = (
            list(masks)
            if re_label == PRIMARY_RE
            else [s for s in RE_SENSITIVITY_SUBSETS if s in masks]
        )
        print(
            f"\n---------- RE structure: {re_label} ({re_groups}) "
            f"across subsets {subsets_for_this_re} ----------"
        )
        for subset_label in subsets_for_this_re:
            base_mask = masks[subset_label]
            print(
                f"\n--- block: {subset_label} "
                f"({int(base_mask.sum())} candidate rows) ---"
            )
            for spec in p_specs:
                slope_rows, variance_rows = run_one(
                    df,
                    spec,
                    subset_label,
                    base_mask,
                    re_label,
                    re_groups,
                    OUT_DIR,
                )
                all_results.extend(slope_rows)
                all_variance.extend(variance_rows)

    # ----- decomposition analysis: primary RE x full only -------------------
    print("\n========== window decomposition ==========")
    print(f"  planned fits: {len(d_specs)}")
    re_groups = RE_STRUCTURES[PRIMARY_RE]
    base_mask = masks["full"]
    print(
        f"\n--- block: full "
        f"({int(base_mask.sum())} candidate rows), RE = {PRIMARY_RE} ---"
    )
    for spec in d_specs:
        slope_rows, variance_rows = run_one(
            df,
            spec,
            "full",
            base_mask,
            PRIMARY_RE,
            re_groups,
            OUT_DIR,
        )
        all_results.extend(slope_rows)
        all_variance.extend(variance_rows)

    # ----- output -----------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "slope_summary.csv", index=False)

    variance_df = pd.DataFrame(all_variance)
    variance_df.to_csv(OUT_DIR / "variance_summary.csv", index=False)

    print_primary_table(all_results)
    for subset in RE_SENSITIVITY_SUBSETS:
        print_re_sensitivity_table(all_results, subset=subset)
    print_decomposition_table(all_results, d_specs)
    print_covariate_slopes(all_results)
    print_variance_table(all_variance)
    print_diagnostic_report(all_results)
    write_stats("receiver_effect_fits", fit_stats(all_results))

    print(f"\nFull results CSV:   {OUT_DIR / 'slope_summary.csv'}")
    print(f"Variance comps CSV: {OUT_DIR / 'variance_summary.csv'}")
    print(f"Traces:             {OUT_DIR}/<re>__<subset>__<model>.nc")


if __name__ == "__main__":
    main()
