"""
Hurdle / two-part Bayesian GLMM for chirp activity around chase offsets,
comparing observed activity against a shuffle-time null.

For each chase event we count chirps in the fixed
[chase_offset - 3, chase_offset + 13] s window (the m3p13 window),
separately for the winner and the loser. The dataset is stacked so each
chase contributes two rows -- one with the loser as focal, one with the
winner as focal -- with a `status` column distinguishing them
(0 = loser, 1 = winner).

For every observed row we also compute, by permuting chirp times
uniformly across the active phase (B=1000 surrogates of the same recording),
two row-aligned shuffle-null statistics:

  * `p_null_pos[j]`  : P(>=1 chirp in this row's window) under the null
  * `mu_null_pos[j]` : E[chirp count | >0] under the null

These enter as offsets (logit and log) so the hurdle-model coefficients
estimate the *residual* deviation of observed activity from what
random chirp placement alone would produce.

Two parts of the hurdle:

  Model 1 (presence) : is_chirp ~ 0 + status + sex_c + REs
                       + offset(logit_p_null_pos)        family = bernoulli
  Model 2 (intensity): chirps   ~ 0 + status + sex_c + REs
                       + offset(log_mu_null_pos)         family = negativebinomial
                       fit only on rows with chirps > 0.

Focal sex enters as a centred contrast, sex_c = +0.5 (male) / -0.5
(female), so that the status coefficients describe the average of both
sexes rather than the female reference level of a treatment-coded factor.

With `0 + status`, each status coefficient is the residual log-odds
(Model 1) or log-mean (Model 2) shift vs the shuffle baseline for that
status. A coefficient near 0 means the status-specific activity is
indistinguishable from the shuffle null; a positive coefficient means
chirping is elevated above what random placement would produce.

Inputs (from `summarize_chirp_event_correlation_around_chasing.py`):

  data/processed/chirp_event_correlations/data/chasing/
      chirps_around_chasing_obs.csv
      chirps_around_chasing_perm_{0..299}.csv

Outputs:
  data/processed/chirp_event_correlations/hurdle_shuffle_baseline/
      <resp>_obs_with_shuffle_baselines.csv
      <resp>_obs_vs_nullmean.png
      <resp>_p_null_pos_by_status.png
      <resp>_hurdle_presence_trace.nc
      <resp>_hurdle_positive_trace.nc
      summary_hurdle_presence.json
      summary_hurdle_positive.json
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats_utils import parse_bambi_model_output


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed/chirp_event_correlations_m3p13/data/chasing")
OBS_FILE = DATA_DIR / "chirps_around_chasing_obs.csv"
OUTDIR = Path("data/processed/chirp_event_correlations_m3p13/hurdle_shuffle_baseline")

# Primary response: the m3p13 window. Listed as a list so other windows
# can be appended later (e.g. 20s symmetric) for sensitivity checks.
RESPONSE_VARS = ["n_chirps_fixed_m3p13"]

MODEL_KWARGS = dict(dropna=True)

SAMPLER_KWARGS = dict(
    tune=1500,
    draws=1500,
    chains=10,
    cores=10,
    random_seed=42,
    target_accept=0.99,
    max_treedepth=15,
)


# ---------------------------------------------------------------------------
# Stacking and loading
# ---------------------------------------------------------------------------
def stack_df_winner_loser(df: pd.DataFrame) -> pd.DataFrame:
    """Stack one wide row per chase into two long rows (loser + winner).

    After stacking, each chase contributes two rows that share the same
    `interaction_id` but differ in `status` (0 = loser, 1 = winner). The
    paired chirp-count columns `n_loser_chirps_*` and `n_winner_chirps_*`
    collapse to a single `n_chirps_*` column thanks to the prefix
    stripping at the bottom of each branch.
    """
    common_cols = [c for c in df.columns if ("loser" not in c) and ("winner" not in c)]
    loser_cols = [c for c in df.columns if "loser" in c]
    loser_cols.append("winner_id")
    winner_cols = [c for c in df.columns if "winner" in c]
    winner_cols.append("loser_id")

    df_loser = df[common_cols + loser_cols].copy()
    df_loser["status"] = 0
    df_loser.rename(
        columns={"loser_id": "focal_id", "winner_id": "opponent_id"},
        inplace=True,
    )
    df_loser.rename(
        columns={"loser_sex": "focal_sex", "winner_sex": "opponent_sex"},
        inplace=True,
    )
    df_loser.rename(
        columns={
            "loser_experience": "focal_experience",
            "winner_experience": "opponent_experience",
        },
        inplace=True,
    )
    df_loser.columns = [c.replace("loser_", "") for c in df_loser.columns]

    df_winner = df[common_cols + winner_cols].copy()
    df_winner["status"] = 1
    df_winner.rename(
        columns={"loser_id": "opponent_id", "winner_id": "focal_id"},
        inplace=True,
    )
    df_winner.rename(
        columns={"loser_sex": "opponent_sex", "winner_sex": "focal_sex"},
        inplace=True,
    )
    df_winner.rename(
        columns={
            "loser_experience": "opponent_experience",
            "winner_experience": "focal_experience",
        },
        inplace=True,
    )
    df_winner.columns = [c.replace("winner_", "") for c in df_winner.columns]

    df_long = pd.concat([df_loser, df_winner], ignore_index=True)
    return df_long


def load_data_with_interaction_id(path: Path) -> pd.DataFrame:
    """Read a wide CSV, attach a per-row interaction_id, and stack.

    The interaction_id is just the wide row index. It survives stacking
    (no `winner` / `loser` substring) and is the alignment anchor across
    permutation files.
    """
    df_wide = pd.read_csv(path).copy()
    df_wide["interaction_id"] = np.arange(len(df_wide), dtype=np.int64)
    return stack_df_winner_loser(df_wide)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Permutation alignment + null statistics
# ---------------------------------------------------------------------------
def choose_alignment_cols(df_obs: pd.DataFrame, df_perm: pd.DataFrame) -> list[str]:
    candidates = ["interaction_id", "status", "focal_id", "opponent_id", "recording"]
    cols = [c for c in candidates if c in df_obs.columns and c in df_perm.columns]
    if "interaction_id" not in cols or "status" not in cols:
        raise ValueError(
            f"Need at least interaction_id and status for alignment; got {cols}"
        )
    return cols


def assert_row_alignment(
    df_obs: pd.DataFrame, df_perm: pd.DataFrame, id_cols: list[str]
) -> None:
    if len(df_obs) != len(df_perm):
        raise ValueError(f"Row count mismatch: obs={len(df_obs)} perm={len(df_perm)}")
    for c in id_cols:
        a = df_obs[c].to_numpy()
        b = df_perm[c].to_numpy()
        if not np.array_equal(a, b):
            idx = int(np.where(a != b)[0][0])
            raise ValueError(
                f"Alignment failed for {c} at row {idx}: obs={a[idx]} perm={b[idx]}"
            )


def compute_shuffle_baselines(
    df_obs: pd.DataFrame,
    perm_files: list[Path],
    resp: str,
    log_every: int = 25,
) -> pd.DataFrame:
    """Compute per-row null statistics from B permutation files.

    Returns df_obs with three new columns:
      null_mean    : mean of shuffled chirp counts at this row
      p_null_pos   : P(shuffled count > 0) at this row, Jeffreys-smoothed
      mu_null_pos  : E[shuffled count | shuffled count > 0]; defaults to 0
                     where no permutation produced a positive count
    """
    if resp not in df_obs.columns:
        raise ValueError(
            f"Response column not found in observed data: {resp}\n"
            f"available chirp columns: "
            f"{[c for c in df_obs.columns if 'chirp' in c]}"
        )
    B = len(perm_files)
    if B == 0:
        raise ValueError("No perm files found.")

    y_obs = df_obs[resp].to_numpy()
    n = len(y_obs)

    sum_all = np.zeros(n, dtype=np.float64)
    sumsq_all = np.zeros(n, dtype=np.float64)
    pos_count = np.zeros(n, dtype=np.int64)
    pos_sum = np.zeros(n, dtype=np.float64)

    alignment_cols = None
    for i, f in enumerate(perm_files, start=1):
        df_perm = load_data_with_interaction_id(f)
        if resp not in df_perm.columns:
            raise ValueError(f"Response column not found in perm file {f}: {resp}")
        if i == 1:
            alignment_cols = choose_alignment_cols(df_obs, df_perm)
            assert_row_alignment(df_obs, df_perm, alignment_cols)

        y = df_perm[resp].to_numpy()
        sum_all += y
        sumsq_all += y.astype(np.float64) ** 2
        is_pos = y > 0
        pos_count += is_pos
        pos_sum += y * is_pos

        del df_perm, y, is_pos
        if log_every and i % log_every == 0:
            gc.collect()
            print(f"  processed {i}/{B} perm files...")

    null_mean = sum_all / B
    null_var = (sumsq_all / B) - (null_mean**2)
    null_sd = np.sqrt(np.maximum(null_var, 0.0))

    # Jeffreys-style smoothing keeps logit() finite at the boundaries.
    p_null_pos = (pos_count + 0.5) / (B + 1.0)

    mu_null_pos = np.zeros(n, dtype=np.float64)
    mask = pos_count > 0
    mu_null_pos[mask] = pos_sum[mask] / pos_count[mask]

    out = df_obs.copy()
    out["null_mean"] = null_mean
    out["null_sd"] = null_sd
    out["p_null_pos"] = p_null_pos
    out["mu_null_pos"] = mu_null_pos
    return out


# ---------------------------------------------------------------------------
# Quick diagnostic plots
# ---------------------------------------------------------------------------
def quick_sanity_plots(df: pd.DataFrame, resp: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter(df["null_mean"], df[resp], alpha=0.25, s=8)
    ax.plot([0, 1], [0, 1], c="k", ls="dashed", transform=ax.transAxes)
    maxxy = float(np.max([ax.get_xlim(), ax.get_ylim()]))
    ax.set_xlim(0, maxxy)
    ax.set_ylim(0, maxxy)
    ax.set_xlabel("null_mean (mean perm count)")
    ax.set_ylabel(f"observed {resp}")
    ax.set_title("Observed vs shuffle-null mean (row-wise)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outdir / f"{resp}_obs_vs_nullmean.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(
        df.loc[df["status"] == 0, "p_null_pos"],
        bins=50,
        density=True,
        alpha=0.5,
        label="loser (status=0)",
    )
    ax.hist(
        df.loc[df["status"] == 1, "p_null_pos"],
        bins=50,
        density=True,
        alpha=0.5,
        label="winner (status=1)",
    )
    ax.set_xlabel("p_null_pos = P(perm count > 0)")
    ax.set_ylabel("density")
    ax.set_title("Shuffle-null probability of any chirp in the window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{resp}_p_null_pos_by_status.png", dpi=200)
    plt.close(fig)

    n0 = int((df["mu_null_pos"].to_numpy() == 0).sum())
    print(f"Rows with mu_null_pos == 0 (never positive in shuffles): {n0} / {len(df)}")


# ---------------------------------------------------------------------------
# Hurdle modelling
# ---------------------------------------------------------------------------
def fit_hurdle_models_with_bambi(
    df: pd.DataFrame, resp: str, outdir: Path
) -> tuple[
    bmb.Model,
    az.InferenceData,
    bmb.Model,
    az.InferenceData,
    dict[int, dict],
    dict[int, dict],
]:
    """Fit Model 1 (presence) and Model 2 (positive intensity).

    `0 + status` parametrises the model so that each `status` coefficient
    is the per-status residual shift vs the shuffle baseline. Coefficient
    = 0 means "indistinguishable from null"; positive means elevated.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["is_chirp"] = (df[resp] > 0).astype(int)
    df["logit_p_null_pos"] = logit(df["p_null_pos"].to_numpy())

    eps = 1e-6
    df["log_mu_null_pos"] = np.log(df["mu_null_pos"].to_numpy() + eps)
    df["sex_c"] = np.where(df["focal_sex"] == "m", 0.5, -0.5)

    categorical = [
        "status",
        "focal_sex",
        "focal_id",
        "opponent_id",
        "recording",
        "interaction_id",
    ]

    print(
        "\nFitting Model 1 (presence): is_chirp ~ 0 + status + ... "
        "anchored to p_null_pos"
    )
    formula1 = (
        "is_chirp ~ 0 + status + sex_c "
        # "+ (1|focal_id) + (1|opponent_id) + (1|recording) + (1|interaction_id) "
        # "+ (1|focal_id) + (1|opponent_id) + (1|interaction_id) "
        "+ (1|recording) + (1|interaction_id) "
        "+ offset(logit_p_null_pos)"
    )
    model1 = bmb.Model(
        formula1,
        data=df,
        family="bernoulli",
        link="logit",
        categorical=categorical,
        **MODEL_KWARGS,
    )
    trace1 = model1.fit(**SAMPLER_KWARGS)

    df_pos = df.loc[df[resp] > 0].copy()
    print(
        f"\nFitting Model 2 (positive counts): {resp} | >0 "
        f"(n={len(df_pos)}) anchored to mu_null_pos"
    )
    formula2 = (
        f"{resp} ~ 0 + status + sex_c "
        # "+ (1|focal_id) + (1|opponent_id) + (1|recording) + (1|interaction_id) "
        # "+ (1|focal_id) + (1|opponent_id) + (1|interaction_id) "
        "+ (1|recording) + (1|interaction_id) "
        "+ offset(log_mu_null_pos)"
    )
    model2 = bmb.Model(
        formula2,
        data=df_pos,
        family="negativebinomial",
        link="log",
        categorical=categorical,
        **MODEL_KWARGS,
    )
    trace2 = model2.fit(**SAMPLER_KWARGS)

    presence_summary = summarize_status_multipliers(trace1, "Model 1 (presence)")
    intensity_summary = summarize_status_multipliers(
        trace2, "Model 2 (positive counts)"
    )

    return model1, trace1, model2, trace2, presence_summary, intensity_summary


def _extract_status_draws(
    trace: az.InferenceData,
) -> dict[int, np.ndarray]:
    """Return {status_level: posterior_draws_1d} for the `status` term,
    handling either Bambi encoding (single vector var or scalar vars).
    """
    post = trace.posterior
    varnames = list(post.data_vars)

    if "status" in varnames:
        da = post["status"]
        level_dims = [d for d in da.dims if d not in ("chain", "draw")]
        if len(level_dims) != 1:
            raise ValueError(
                f"Expected a single level dimension on 'status', "
                f"found dims={da.dims} -> level_dims={level_dims}"
            )
        lvl_dim = level_dims[0]
        return {
            k: da.isel({lvl_dim: k}).stack(sample=("chain", "draw")).to_numpy()
            for k in range(da.sizes[lvl_dim])
        }

    scalar_vars = sorted(v for v in varnames if v.startswith("status["))
    if scalar_vars:
        out: dict[int, np.ndarray] = {}
        for v in scalar_vars:
            k = int(v.replace("status[", "").rstrip("]"))
            out[k] = post[v].stack(sample=("chain", "draw")).to_numpy()
        return out

    raise ValueError(f"No 'status' posterior variable found. Available: {varnames}")


def _status_label(k: int) -> str:
    return {0: "loser", 1: "winner"}.get(k, f"level {k}")


def summarize_status_multipliers(
    trace: az.InferenceData, model_name: str, hdi_prob: float = 0.95
) -> dict[int, dict]:
    """Print and return per-status posterior summaries on the linear and
    multiplicative scales.

    Returned dict, keyed by status level: {beta, beta_lo, beta_hi, mult,
    mult_lo, mult_hi, p_gt0, p_mult_gt1}, all on whichever interpretable
    scale the caller likes.
    """
    print(f"\n[{model_name}] Status posterior (linear-predictor scale):")
    summ = az.summary(
        trace,
        var_names=["status"],
        kind="stats",
        hdi_prob=hdi_prob,
        filter_vars="like",
    )
    print(summ)

    out: dict[int, dict] = {}
    draws_per_level = _extract_status_draws(trace)
    print(
        f"\n[{model_name}] exp(status) multiplier "
        f"(vs shuffle-null baseline; HDI{int(hdi_prob * 100)}):"
    )
    for k, samples in draws_per_level.items():
        mult = np.exp(samples)
        beta_hdi = az.hdi(samples, hdi_prob=hdi_prob)
        mult_hdi = az.hdi(mult, hdi_prob=hdi_prob)
        out[k] = {
            "label": _status_label(k),
            "beta": float(np.median(samples)),
            "beta_lo": float(beta_hdi[0]),
            "beta_hi": float(beta_hdi[1]),
            "mult": float(np.median(mult)),
            "mult_lo": float(mult_hdi[0]),
            "mult_hi": float(mult_hdi[1]),
            "p_gt0": float((samples > 0).mean()),
            "p_mult_gt1": float((mult > 1).mean()),
            "samples": samples,  # carried for cross-status contrasts
        }
        print(
            f"  status[{k}] ({_status_label(k):>6}): "
            f"x{out[k]['mult']:.3f}  "
            f"HDI [{out[k]['mult_lo']:.3f}, {out[k]['mult_hi']:.3f}]   "
            f"P(mult > 1) = {out[k]['p_mult_gt1']:.3f}"
        )
    return out


def print_hurdle_summary(
    df_scored: pd.DataFrame,
    presence_summary: dict[int, dict],
    intensity_summary: dict[int, dict],
    resp: str,
    hdi_prob: float = 0.95,
) -> None:
    """Final digest of the hurdle results on interpretable scales.

    Reports, per status:
      * Model 1: predicted P(any chirp) vs shuffle-null baseline.
        Concrete anchor at the median row's p_null_pos.
      * Model 2: rate multiplier of conditional positive count.
        Concrete anchor at the median row's mu_null_pos.
      * Cross-status contrast: difference of betas (and exp of that =
        multiplier of one status' multiplier vs the other).
    Plus dataset-level descriptive numbers.
    """
    print("\n" + "=" * 72)
    print(f"Hurdle-model digest for {resp}")
    print("=" * 72)

    # Dataset anchors per status.
    print("\nDataset anchors (observed):")
    print(
        f"  total rows         : {len(df_scored):>6}  "
        f"({df_scored['interaction_id'].nunique()} chases x 2 statuses)"
    )
    for status, label in ((0, "loser"), (1, "winner")):
        sub = df_scored[df_scored["status"] == status]
        n = len(sub)
        n_pos = int((sub[resp] > 0).sum())
        rate = float(sub[resp].mean())
        print(
            f"  {label:>6}: n = {n}, "
            f"P(any chirp) = {n_pos / n:.3f} ({n_pos}/{n}), "
            f"mean count = {rate:.3f}"
        )

    # Null baselines differ strongly between losers and winners, so the
    # concrete anchors are computed per status, not pooled.
    median_p_null, median_mu_null = {}, {}
    for status in (0, 1):
        sub = df_scored[df_scored["status"] == status]
        median_p_null[status] = float(sub["p_null_pos"].median())
        median_mu_null[status] = float(
            sub.loc[sub["mu_null_pos"] > 0, "mu_null_pos"].median()
        )
        print(
            f"\nShuffle-null medians for {_status_label(status)}s:\n"
            f"  median p_null_pos   = {median_p_null[status]:.4f}  "
            f"(probability of any chirp under random placement)\n"
            f"  median mu_null_pos  = {median_mu_null[status]:.3f}  "
            f"(expected positive-count chirps under random placement)"
        )

    # Model 1 — presence: show on probability scale at the median null.
    print(
        f"\nModel 1 (presence, Bernoulli) -- elevation of P(any chirp) vs shuffle null"
    )
    print(
        "  status   odds-mult [HDI]                "
        "P(any) at median null       P(elevated)"
    )
    for k, s in presence_summary.items():
        # Posterior of P(any chirp) at the status' median null baseline,
        # averaged over sexes: logistic(logit(p_null) + status_coef).
        logit_pred = logit(np.array([median_p_null[k]]))[0] + s["samples"]
        p_pred = 1.0 / (1.0 + np.exp(-logit_pred))
        p_med = float(np.median(p_pred))
        p_hdi = az.hdi(p_pred, hdi_prob=hdi_prob)
        print(
            f"  {s['label']:>6}   "
            f"x{s['mult']:.3f} [{s['mult_lo']:.3f}, {s['mult_hi']:.3f}]   "
            f"{p_med:.4f} [{p_hdi[0]:.4f}, {p_hdi[1]:.4f}]      "
            f"{s['p_mult_gt1']:.3f}"
        )

    # Model 2 -- intensity: rate multiplier and concrete count.
    print(
        f"\nModel 2 (intensity, NegBin, rows with chirps>0) -- "
        f"elevation of expected count vs shuffle null"
    )
    print(
        "  status   rate-mult [HDI]                "
        "predicted count at median null       P(elevated)"
    )
    for k, s in intensity_summary.items():
        # Posterior count = mu_null * exp(status_coef).
        pred_count = median_mu_null[k] * np.exp(s["samples"])
        c_med = float(np.median(pred_count))
        c_hdi = az.hdi(pred_count, hdi_prob=hdi_prob)
        print(
            f"  {s['label']:>6}   "
            f"x{s['mult']:.3f} [{s['mult_lo']:.3f}, {s['mult_hi']:.3f}]   "
            f"{c_med:.3f} [{c_hdi[0]:.3f}, {c_hdi[1]:.3f}]              "
            f"{s['p_mult_gt1']:.3f}"
        )

    # Cross-status contrasts.
    if 0 in presence_summary and 1 in presence_summary:
        print("\nCross-status contrast (winner vs loser):")
        for name, summary in (
            ("Model 1 (presence) odds ratio", presence_summary),
            ("Model 2 (intensity) rate ratio", intensity_summary),
        ):
            diff = summary[1]["samples"] - summary[0]["samples"]
            ratio = np.exp(diff)
            r_mean = float(np.median(ratio))
            r_hdi = az.hdi(ratio, hdi_prob=hdi_prob)
            print(
                f"  {name:<32}  "
                f"x{r_mean:.3f} [{r_hdi[0]:.3f}, {r_hdi[1]:.3f}]   "
                f"P(winner > loser) = {float((diff > 0).mean()):.3f}"
            )

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    perm_files = sorted(DATA_DIR.glob("*perm*.csv"))

    print(f"Observed:   {OBS_FILE}")
    print(f"Perm files: {len(perm_files)} in {DATA_DIR}")
    if not OBS_FILE.exists():
        raise SystemExit(f"Observed file not found: {OBS_FILE}")
    if len(perm_files) == 0:
        raise SystemExit("No permuted CSVs found.")

    df_obs = load_data_with_interaction_id(OBS_FILE)
    print(
        f"Stacked observed data: {len(df_obs)} rows "
        f"({df_obs['interaction_id'].nunique()} chases x 2 statuses)"
    )

    for resp in RESPONSE_VARS:
        print(f"\nComputing shuffle baselines for: {resp}")
        df_scored = compute_shuffle_baselines(df_obs, perm_files, resp, log_every=25)

        scored_path = OUTDIR / f"{resp}_obs_with_shuffle_baselines.csv"
        df_scored.to_csv(scored_path, index=False)
        print(f"Saved scored observed data: {scored_path}")

        quick_sanity_plots(df_scored, resp, OUTDIR)

        mod1, trace1, mod2, trace2, presence_summary, intensity_summary = (
            fit_hurdle_models_with_bambi(df_scored, resp, OUTDIR)
        )

        print_hurdle_summary(
            df_scored=df_scored,
            presence_summary=presence_summary,
            intensity_summary=intensity_summary,
            resp=resp,
        )

        mod1_info = parse_bambi_model_output(mod1)
        mod2_info = parse_bambi_model_output(mod2)
        summary1 = az.summary(trace1, hdi_prob=0.95).to_dict()
        summary2 = az.summary(trace2, hdi_prob=0.95).to_dict()

        with open(OUTDIR / f"{resp}_summary_hurdle_presence.json", "w") as f:
            json.dump({"model_info": mod1_info, "model_summary": summary1}, f, indent=2)
        with open(OUTDIR / f"{resp}_summary_hurdle_positive.json", "w") as f:
            json.dump({"model_info": mod2_info, "model_summary": summary2}, f, indent=2)

        az.to_netcdf(trace1, OUTDIR / f"{resp}_hurdle_presence_trace.nc")
        az.to_netcdf(trace2, OUTDIR / f"{resp}_hurdle_positive_trace.nc")


if __name__ == "__main__":
    main()
