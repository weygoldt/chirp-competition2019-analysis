# Ported from chirp_competition d040123 (src/stats_bayesian_logreg_echo_responses_new.py).
# Changes: input/output under $PAPER_OUT/data/processed (imports paper_config, which also sets the fork
# start method for pymc on Python 3.14). Model, priors, sampler settings and seed unchanged. The printed
# numbers (plus trials per direction and the observed/chance ratios) also go to write_stats ("echo").
"""Estimate P(chirp-after | other chirped) in two directions using
intercept-only Bayesian GLMMs, and compare against a permutation-based
chance level.

For each direction (loser-after-winner, winner-after-loser) we:
  1. Fit a GLMM on the single *obs* dataset → posterior for p_obs.
  2. Load all surrogate datasets (1000; sender chirps jittered by
     U(-120, 120) s), compute the raw response fraction
     in each → null distribution of chance-level fractions.
  3. Report chance level as median [2.5%, 97.5%] of null distribution.
  4. Compute a permutation p-value: (k + 1) / (B + 1), with k the number
     of surrogate fractions ≥ observed fraction.
"""

from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from manuscript_stats import write_stats
from paper_config import PROCESSED
from stats_utils import log_odds_to_prob, parse_bambi_model_output, save_json

CONFIGS = [
    {
        "tag": "loser_after_winner",
        "dataroot": PROCESSED / "chirp_event_correlations" / "data" / "winner_chirps",
        "raw_count_col": "loser_chirps_10s_after",
        "y_col": "loser_chirped_after",
        "pretty": "Loser chirped after winner chirped",
        "tex_key": r"$P(\mathrm{loser\ after\ winner})$",
    },
    {
        "tag": "winner_after_loser",
        "dataroot": PROCESSED / "chirp_event_correlations" / "data" / "loser_chirps",
        "raw_count_col": "winner_chirps_10s_after",
        "y_col": "winner_chirped_after",
        "pretty": "Winner chirped after loser chirped",
        "tex_key": r"$P(\mathrm{winner\ after\ loser})$",
    },
]

SAMPLER_KWARGS = dict(
    tune=6000,
    draws=1000,
    chains=5,
    cores=5,
    target_accept=0.99,
    random_seed=42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_single_csv(dataroot: Path, pattern: str) -> Path:
    """Return the single CSV whose name matches *pattern*."""
    files = sorted(dataroot.glob(f"*{pattern}*"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one file matching '*{pattern}*' in "
            f"{dataroot}, got {len(files)} files"
        )
    return files[0]


def _load_all_csvs(dataroot: Path, pattern: str) -> list[Path]:
    """Return all CSVs whose names match *pattern*, sorted for reproducibility."""
    files = sorted(dataroot.glob(f"*{pattern}*"))
    if not files:
        raise ValueError(f"No files matching '*{pattern}*' in {dataroot}")
    return files


def _binarize(df: pd.DataFrame, raw_count_col: str, y_col: str) -> pd.DataFrame:
    df[y_col] = df[raw_count_col] > 0
    return df


def _fit_intercept_glmm(df: pd.DataFrame, y_col: str):
    model = bmb.Model(
        # formula=f"{y_col} ~ 1 + (1|recording) + (1|loser_id) + (1|winner_id)",
        formula=f"{y_col} ~ 1 + (1|recording)",
        data=df,
        family="bernoulli",
        categorical=["recording", "loser_id", "winner_id"],
    )
    idata = model.fit(**SAMPLER_KWARGS)
    return model, idata


def _intercept_prob_draws(idata) -> np.ndarray:
    """Return probability-scale posterior draws from the intercept."""
    intercept = idata.posterior["Intercept"].to_numpy().reshape(-1)
    return log_odds_to_prob(intercept)


def _summarize(draws, hdi_prob=0.95):
    mean_p = float(np.median(draws))  # point estimate: posterior median
    lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
    return mean_p, float(lo), float(hi)


def sampler_diagnostics(idata) -> dict:
    """Divergences, largest R-hat and smallest bulk ESS over all parameters."""
    summ = az.summary(idata, kind="diagnostics")
    return {"divergences": int(idata.sample_stats["diverging"].sum()),
            "max_rhat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min())}


def _build_null_distribution(
    perm_paths: list[Path],
    raw_count_col: str,
    y_col: str,
) -> np.ndarray:
    """Compute the raw response fraction for each permutation file."""
    fractions = []
    for path in perm_paths:
        df = _binarize(pd.read_csv(path), raw_count_col, y_col)
        fractions.append(float(df[y_col].mean()))
    return np.array(fractions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    savepath = PROCESSED
    results = []
    stats = {}

    for cfg in CONFIGS:
        tag = cfg["tag"]
        y_col = cfg["y_col"]
        raw_col = cfg["raw_count_col"]

        print(f"\n{'=' * 60}")
        print(f"  {cfg['pretty']}")
        print(f"{'=' * 60}")

        # ---- observed: fit GLMM ----
        obs_path = _load_single_csv(cfg["dataroot"], "obs")
        df_obs = _binarize(pd.read_csv(obs_path), raw_col, y_col)
        model_obs, idata_obs = _fit_intercept_glmm(df_obs, y_col)
        p_obs_draws = _intercept_prob_draws(idata_obs)
        obs_frac = float(df_obs[y_col].mean())
        obs_mean, obs_lo, obs_hi = _summarize(p_obs_draws)

        # ---- permuted: null distribution from all perm files ----
        perm_paths = _load_all_csvs(cfg["dataroot"], "perm")
        null_fracs = _build_null_distribution(perm_paths, raw_col, y_col)
        chance_median = float(np.median(null_fracs))
        chance_lo = float(np.percentile(null_fracs, 2.5))
        chance_hi = float(np.percentile(null_fracs, 97.5))
        n_perm = len(null_fracs)

        # ---- permutation p-value (one-sided: observed ≥ chance) ----
        perm_p = float((np.sum(null_fracs >= obs_frac) + 1) / (len(null_fracs) + 1))

        # ---- console output ----
        print(f"  n events:           {len(df_obs)}")
        print(f"  Observed fraction:  {obs_frac:.3f}")
        print(
            f"  Model (obs):        {obs_mean:.3f}  "
            f"95% HDI [{obs_lo:.3f}, {obs_hi:.3f}]"
        )
        print(f"  Permutations:       {n_perm}")
        print(
            f"  Chance level:       median={chance_median:.3f}  "
            f"95% CI [{chance_lo:.3f}, {chance_hi:.3f}]"
        )
        print(f"  Perm. p-value:      {perm_p:.4f}  (fraction of perms ≥ observed)")

        # ---- save ----
        stem = f"{Path(__file__).stem}_{tag}"
        az.to_netcdf(idata_obs, savepath / f"{stem}_obs_trace.nc")
        save_json(
            parse_bambi_model_output(model_obs),
            savepath / f"{stem}_obs_summary.json",
        )

        stats[tag] = {
            "n_events": len(df_obs),
            "n_trials": df_obs["recording"].nunique(),
            "observed": obs_frac,
            "observed_over_chance": obs_frac / chance_median,
            "chance": {"median": chance_median, "ci_95": [chance_lo, chance_hi]},
            "n_permutations": n_perm,
            "p_perm": perm_p,
            "model": {"median": obs_mean, "hdi": [obs_lo, obs_hi]},
            "diagnostics": sampler_diagnostics(idata_obs),
        }
        results.append(
            {
                "pretty": cfg["pretty"],
                "tex_key": cfg["tex_key"],
                "n": len(df_obs),
                "obs_frac": obs_frac,
                "obs_mean": obs_mean,
                "obs_lo": obs_lo,
                "obs_hi": obs_hi,
                "chance_median": chance_median,
                "chance_lo": chance_lo,
                "chance_hi": chance_hi,
                "n_perm": n_perm,
                "perm_p": perm_p,
            }
        )

    # ---- LaTeX-ready output ----
    print("\n" + "=" * 72)
    print("LaTeX (copy-paste):")
    print(r"\begin{itemize}")
    for r in results:
        print(
            rf"  \item {r['tex_key']}: "
            rf"observed {r['obs_frac']:.1%} "
            rf"(chance level {r['chance_median']:.1%}, "
            rf"95\,\%\ CI [{r['chance_lo']:.1%}, {r['chance_hi']:.1%}], "
            rf"$n_{{\text{{perm}}}}={r['n_perm']}$, "
            rf"$p_{{\text{{perm}}}}={r['perm_p']:.4f}$); "
            rf"GLMM $\hat{{p}}={r['obs_mean']:.3f}$, "
            rf"95\,\%\ HDI $[{r['obs_lo']:.3f}, {r['obs_hi']:.3f}]$."
        )
    print(r"\end{itemize}")
    print("=" * 72)

    # The events of each direction are the other fish's chirps: winner chirps
    # (loser after winner) and loser chirps (winner after loser).
    stats["loser_to_winner_chirps"] = (
        stats["winner_after_loser"]["n_events"] / stats["loser_after_winner"]["n_events"]
    )
    write_stats("echo", stats)


if __name__ == "__main__":
    main()
