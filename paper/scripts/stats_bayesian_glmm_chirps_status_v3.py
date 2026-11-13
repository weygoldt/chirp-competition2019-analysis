# Ported from chirp_competition d040123 (src/stats_bayesian_glmm_chirps_status_v3.py).
# Changes: input/output under $PAPER_OUT/data/processed (the old save path was hard-coded into the old repo); random_seed=SEED for the fit and np.random.seed(SEED) before the new-group predictions (both unseeded before). Model, priors and sampler settings unchanged.
"""
Test whether chirps differ between winner and loser (negative binomial GLMM).

Reports main effects (status, sex, size) as count-scale multiplicative rate
ratios by exponentiating posterior draws and recomputing HDIs on the
transformed scale.
"""

import json
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from stats_utils import parse_bambi_model_output
from paper_config import SEED, processed, processed_dir


# ---------------------------------------------------------------------------
# Posterior summaries on the count-multiplier (exp-link) scale
# ---------------------------------------------------------------------------


def _hdi_bounds(da, hdi_prob):
    """Return (lower, upper) of the HDI of a 1-D posterior DataArray."""
    h = az.hdi(da, hdi_prob=hdi_prob)
    if isinstance(h, xr.Dataset):
        h = next(iter(h.data_vars.values()))
    return float(h.sel(hdi="lower")), float(h.sel(hdi="higher"))


def _categorical_multiplier(beta, df, var, focus, hdi_prob=0.95):
    """Posterior mean and HDI of mu(focus) / mu(other) on the count scale.

    Reference (treatment-coded baseline) is df[var].cat.categories[0]. If
    `focus` is the reference, we invert: exp(-beta) gives ref / non-ref.
    Assumes a binary categorical (one non-reference level in the posterior).
    """
    levels = list(df[var].cat.categories)
    if focus not in levels:
        raise ValueError(f"focus {focus!r} not in {var} categories {levels!r}")
    ref = levels[0]

    cat_dim = next(d for d in beta.dims if d not in ("chain", "draw"))
    non_ref_levels = beta[cat_dim].values.tolist()

    if focus == ref:
        # focus IS the reference; report ref / non-ref
        other = non_ref_levels[0]
        mult = np.exp(-beta.sel({cat_dim: other}))
        denom = other
    else:
        mult = np.exp(beta.sel({cat_dim: focus}))
        denom = ref

    lo, hi = _hdi_bounds(mult, hdi_prob)
    return {
        "var": var,
        "contrast": f"{focus}/{denom}",
        "mean": float(mult.mean()),
        "hdi_lower": lo,
        "hdi_upper": hi,
    }


def _continuous_multiplier(beta, var, hdi_prob=0.95, unit_label="per +1 unit"):
    """Posterior mean and HDI of exp(beta) for a continuous predictor."""
    mult = np.exp(beta)
    lo, hi = _hdi_bounds(mult, hdi_prob)
    return {
        "var": var,
        "contrast": unit_label,
        "mean": float(mult.mean()),
        "hdi_lower": lo,
        "hdi_upper": hi,
    }


def report_count_multipliers(
    idata,
    df,
    *,
    categorical_focus=None,
    continuous=None,
    hdi_prob=0.95,
):
    """Print a manuscript-ready table of count-scale multipliers.

    Parameters
    ----------
    idata : arviz.InferenceData
    df : pd.DataFrame
        Modelling frame (used to look up category levels).
    categorical_focus : dict, optional
        {var: focus_level}. Reports mu(focus) / mu(other_level) so the
        direction of the contrast is explicit and does not depend on
        alphabetic sort. Example: {"status": "loser", "sex": "male"}.
    continuous : dict, optional
        {var: unit_label}, e.g. {"size_z": "per +1 SD"}.
    hdi_prob : float

    Returns
    -------
    pd.DataFrame with columns var, contrast, mean, hdi_lower, hdi_upper.
    """
    cat_focus = categorical_focus or {}
    cont = continuous or {}
    rows = []

    for var, focus in cat_focus.items():
        if var not in idata.posterior:
            print(f"  [{var}] not in posterior; skipped")
            continue
        rows.append(
            _categorical_multiplier(
                idata.posterior[var],
                df,
                var,
                focus,
                hdi_prob=hdi_prob,
            )
        )

    for var, unit_label in cont.items():
        if var not in idata.posterior:
            print(f"  [{var}] not in posterior; skipped")
            continue
        rows.append(
            _continuous_multiplier(
                idata.posterior[var],
                var,
                hdi_prob=hdi_prob,
                unit_label=unit_label,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["var", "contrast", "mean", "hdi_lower", "hdi_upper"]
        )

    label_w = max(len(f"{r['var']} ({r['contrast']})") for r in rows)
    bar = "=" * (label_w + 30)
    pct = int(round(hdi_prob * 100))

    print()
    print(bar)
    print(f"Count-scale multipliers ({pct}% HDI)")
    print(bar)
    for r in rows:
        label = f"{r['var']} ({r['contrast']})"
        print(
            f"  {label:<{label_w}}  "
            f"{r['mean']:5.2f}x   "
            f"[{r['hdi_lower']:5.2f}, {r['hdi_upper']:5.2f}]"
        )
    print(bar)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------------------------
# Load and preprocess data
df = pd.read_csv(processed("trial_summary_long.csv"))
df["rises_z"] = (df["rises"] - df["rises"].mean()) / df["rises"].std()
df["size_z"] = (df["size"] - df["size"].mean()) / df["size"].std()

# Relabel binary codings to semantic strings BEFORE making them categorical,
# so posterior coords, predictions.csv, and plots all carry readable labels.
# CONFIRM these mappings match your data dictionary.
df["status"] = df["status"].map({1: "winner", 0: "loser"})
# If sex is also 0/1-coded, do the same; otherwise leave it:
if df["sex"].dtype != object and set(df["sex"].dropna().unique()) <= {0, 1}:
    df["sex"] = df["sex"].map({0: "female", 1: "male"})

df["sex"] = df["sex"].map({"f": "female", "m": "male"}).astype("category")

df["chirps"] = df["chirps"].astype(int)
df["sex"] = df["sex"].astype("category")
df["status"] = df["status"].astype("category")
df["trial_idx"] = df["trial_idx"].astype("category")
df["focal_fish"] = df["focal_fish"].astype("category")
df["paired_fish"] = df["paired_fish"].astype("category")

df = df[
    [
        "chirps",
        "rises_z",
        "size_z",
        "sex",
        "status",
        "focal_fish",
        "paired_fish",
        "trial_idx",
    ]
]

# ---------------------------------------------------------------------------
# Quick EDA plots
# ---------------------------------------------------------------------------

grid = int(np.ceil(np.sqrt(len(df.columns))))
fig, axs = plt.subplots(grid, grid, figsize=(10, 10))
axs = axs.flatten()
last_used = -1
for i, col in enumerate(df.columns):
    ax = axs[i]
    if df[col].dtype == "category":
        sns.countplot(x=col, data=df, ax=ax)
    else:
        sns.histplot(x=col, data=df, kde=True, ax=ax)
    ax.set_title(col)
    last_used = i
for j in range(last_used + 1, len(axs)):
    axs[j].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = bmb.Model(
    "chirps ~ status + sex + size_z + (1 | trial_idx)",
    # "chirps ~ status + sex + size_z + (1 | focal_fish) + (1 | paired_fish)",
    # "chirps ~ status + sex + size_z + (1 | focal_fish) + (1 | paired_fish) + (1 | trial_idx)",
    data=df,
    family="negativebinomial",
    dropna=True,
    link="log",
)

idata = model.fit(
    tune=1000,
    draws=1000,
    chains=30,
    cores=30,
    target_accept=0.99,
    max_treedepth=15,
    random_seed=SEED,
)

print(model)

# Log-scale summary (for diagnostics; var_names match the active formula)
print(
    az.summary(
        idata,
        var_names=[
            "Intercept",
            "status",
            "sex",
            "size_z",
            "1|trial_idx",
            # "1|focal_fish",
            # "1|paired_fish",
        ],
        hdi_prob=0.95,
    )
)

# Count-multiplier-scale summary (for the manuscript)
multipliers = report_count_multipliers(
    idata,
    df,
    categorical_focus={"status": "loser", "sex": "male"},
    continuous={"size_z": "per +1 SD"},
    hdi_prob=0.95,
)

# ---------------------------------------------------------------------------
# Posterior predictive checks
# ---------------------------------------------------------------------------

idata_ppc = model.predict(idata, data=df, inplace=False, kind="response")
fig, ax = plt.subplots()
az.plot_ppc(idata_ppc, ax=ax)
plt.show()

az.plot_trace(idata)
plt.show()

fig, ax = plt.subplots()
az.plot_forest(
    idata,
    var_names=["Intercept", "status", "sex", "size_z"],
    ax=ax,
    combined=True,
    kind="ridgeplot",
)
ax.axvline(0, color="k", linestyle="dashed")
plt.show()

# ---------------------------------------------------------------------------
# Predictions for status x sex at mean size
# ---------------------------------------------------------------------------

status_levels = df["status"].cat.categories.tolist()
sex_levels = df["sex"].cat.categories.tolist()

new_data = pd.DataFrame(
    {
        "status": [
            status_levels[0],
            status_levels[1],
            status_levels[0],
            status_levels[1],
        ],
        "sex": [sex_levels[0], sex_levels[0], sex_levels[1], sex_levels[1]],
        "size_z": 0.0,
        "focal_fish": "average_fish",
        "paired_fish": "average_fish",
        "trial_idx": "average_trial",
    }
)

np.random.seed(SEED)
new_idata = model.predict(
    idata,
    data=new_data,
    inplace=False,
    kind="response",
    sample_new_groups=True,
)

new_data = (
    new_data.reset_index(drop=True).reset_index().rename(columns={"index": "__obs__"})
)

pred_draws = new_idata.posterior["mu"]
pred_df = pred_draws.to_dataframe(name="mu").reset_index()
df_merged = pred_df.merge(new_data, on="__obs__", how="left")
df_merged["sample"] = (
    df_merged["chain"].astype(str) + "_" + df_merged["draw"].astype(str)
)

hdi = az.hdi(pred_draws, hdi_prob=0.95)
if isinstance(hdi, xr.Dataset):
    hdi = hdi["mu"] if "mu" in hdi.data_vars else next(iter(hdi.data_vars.values()))
hdi = hdi.to_dataframe().reset_index()

hdi_wide = (
    hdi.pivot(index="__obs__", columns="hdi", values="mu")
    .reset_index()
    .rename_axis(None, axis=1)
    .rename(columns={"lower": "hdi_2.5", "higher": "hdi_97.5"})
)

df_final = df_merged.merge(hdi_wide, on="__obs__", how="left")

sns.violinplot(
    data=df_final,
    x="status",
    y="mu",
    cut=0,
    bw_method="scott",
    hue="sex",
    split=True,
)
plt.show()

# ---------------------------------------------------------------------------
# Persist outputs (column names and JSON keys unchanged)
# ---------------------------------------------------------------------------

savepath = processed_dir(Path(__file__).stem)

df_final.to_csv(savepath / "predictions.csv", sep=",", index=False)

model_info = parse_bambi_model_output(model)
summary_df = az.summary(idata, hdi_prob=0.95)
summary_dict = summary_df.to_dict()

output_dict = {"model_info": model_info, "model_summary": summary_dict}
with open(savepath / "summary.json", "w") as f:
    json.dump(output_dict, f, indent=2)

# Multiplier-scale table for downstream reporting / plotting
multipliers.to_csv(savepath / "count_multipliers.csv", index=False)

idata.to_netcdf(savepath / "posterior.nc")
