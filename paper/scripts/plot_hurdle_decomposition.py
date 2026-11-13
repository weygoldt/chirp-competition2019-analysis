"""
Figure: hurdle-model decomposition of chirp activity vs shuffle null
around chase offsets (m3p13 window).

Three-column layout (eLife two-column width). Each column corresponds
to one component of the hurdle decomposition:

  Column 1 (a, d) -- presence:    P(any chirp) per trial vs null
                                  / posterior odds multiplier vs null
  Column 2 (b, e) -- intensity:   mean count given >=1 chirp
                                  / posterior rate multiplier
  Column 3 (c, f) -- combined:    mean count over all windows
                                  / posterior combined-total multiplier

Top row: per-trial paired observed-vs-null dots, one pair per
recording, connected by a thin line within trial. Loser/winner
overlaid by colour (paper-wide convention: loser=blue, winner=green).

Bottom row: forest with floating KDE clouds. HDI lines and mean dots
are black (paper convention from earlier figures); status is encoded
only in the KDE fill colour. Reference line at multiplier = 1 on
linear x-axis.

Inputs:
  data/processed/chirp_event_correlations/aggregated/chasing/
      hurdle_shuffle_baseline/
          n_chirps_fixed_m3p13_obs_with_shuffle_baselines.csv
          n_chirps_fixed_m3p13_hurdle_presence_trace.nc
          n_chirps_fixed_m3p13_hurdle_positive_trace.nc
"""

from __future__ import annotations

from pathlib import Path

import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from style import (
    cm,
    letter_subplots,
    set_light_style,
    winner_color,
    loser_color,
)

set_light_style()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESP = "n_chirps_fixed_m3p13"
DATA_DIR = Path("data/processed/chirp_event_correlations_m3p13/hurdle_shuffle_baseline")
SCORED_CSV = DATA_DIR / f"{RESP}_obs_with_shuffle_baselines.csv"
PRESENCE_TRACE = DATA_DIR / f"{RESP}_hurdle_presence_trace.nc"
POSITIVE_TRACE = DATA_DIR / f"{RESP}_hurdle_positive_trace.nc"
OUT_FIG = (
    Path(os.environ.get("PAPER_FIGURE_DIR", "/home/weygoldt/wrk/analyses/chirp_competition_paper/figures"))
    / "plot_hurdle_decomposition.pdf"
)

# Forest x-axis: multipliers on linear scale.
FOREST_XLIM = (0.0, 4.0)
FOREST_XTICKS = [0.0, 1.0, 2.0, 3.0, 4.0]

# Per-trial intensity panel: a trial-status cell needs at least this many
# positive windows to contribute a point (otherwise the per-trial mean is
# pure noise from one or two events).
MIN_POSITIVE_FOR_INTENSITY = 3

# Colour palette.
C_TEXT = "#2C2C2A"
C_NEUTRAL = "#5F5E5A"
C_STRUCTURE = "#888780"
C_LOSER = loser_color
C_WINNER = winner_color

STATUS_COLOR = {0: C_LOSER, 1: C_WINNER}
STATUS_LABEL = {0: "loser", 1: "winner"}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_scored() -> pd.DataFrame:
    df = pd.read_csv(SCORED_CSV)
    needed = {
        "recording",
        "status",
        "interaction_id",
        RESP,
        "null_mean",
        "p_null_pos",
        "mu_null_pos",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"scored CSV missing required columns: {missing}")
    return df


def per_trial_paired(
    df: pd.DataFrame,
    quantity: str,
) -> pd.DataFrame:
    """One row per (recording, status) with paired (observed, null)
    columns for the requested hurdle component.

    quantity:
      'presence'  -> P(N>0):    obs = mean(N>0),                  null = mean(p_null_pos)
      'intensity' -> E[N|N>0]:  obs = mean(N | N>0) on positive,  null = mean(mu_null_pos | N>0)
      'combined'  -> E[N]:      obs = mean(N),                    null = mean(null_mean)
    """
    g = df.groupby(["recording", "status"])

    if quantity == "presence":
        obs = g[RESP].apply(lambda s: float((s > 0).mean()))
        null = g["p_null_pos"].mean()
        n_eff = g.size()

    elif quantity == "intensity":
        # Positive subset only.
        def _pos_mean(sub: pd.DataFrame) -> float:
            sub_pos = sub.loc[sub[RESP] > 0]
            if len(sub_pos) < MIN_POSITIVE_FOR_INTENSITY:
                return np.nan
            return float(sub_pos[RESP].mean())

        def _pos_null(sub: pd.DataFrame) -> float:
            sub_pos = sub.loc[sub[RESP] > 0]
            if len(sub_pos) < MIN_POSITIVE_FOR_INTENSITY:
                return np.nan
            return float(sub_pos["mu_null_pos"].mean())

        obs = g[[RESP, "mu_null_pos"]].apply(_pos_mean)
        # NB: the apply above uses the whole sub-frame; null is computed
        # the same way to keep the row count aligned even when the cell
        # has too-few positives.
        null = g[[RESP, "mu_null_pos"]].apply(_pos_null)
        n_eff = g.apply(lambda s: int((s[RESP] > 0).sum()))

    elif quantity == "combined":
        obs = g[RESP].mean()
        null = g["null_mean"].mean()
        n_eff = g.size()

    else:
        raise ValueError(f"unknown quantity {quantity!r}")

    out = pd.DataFrame(
        {
            "recording": [r for r, _ in obs.index],
            "status": [s for _, s in obs.index],
            "obs": obs.values,
            "null": null.values,
            "n_eff": n_eff.values,
        }
    )
    return out.dropna(subset=["obs", "null"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------
def _extract_status_draws(trace: az.InferenceData) -> dict[int, np.ndarray]:
    post = trace.posterior
    if "status" in post.data_vars:
        da = post["status"]
        level_dims = [d for d in da.dims if d not in ("chain", "draw")]
        if len(level_dims) != 1:
            raise ValueError(f"unexpected dims on 'status': {da.dims}")
        lvl = level_dims[0]
        return {
            k: da.isel({lvl: k}).stack(sample=("chain", "draw")).to_numpy()
            for k in range(da.sizes[lvl])
        }
    scalars = sorted(v for v in post.data_vars if v.startswith("status["))
    if scalars:
        out = {}
        for v in scalars:
            k = int(v.replace("status[", "").rstrip("]"))
            out[k] = post[v].stack(sample=("chain", "draw")).to_numpy()
        return out
    raise ValueError("no `status` posterior variable found")


def _extract_scalar_draws(trace: az.InferenceData, name: str) -> np.ndarray:
    """Posterior draws of a scalar coefficient, flattened over chains."""
    return trace.posterior[name].stack(sample=("chain", "draw")).to_numpy()


def combined_multiplier_posterior(
    df: pd.DataFrame,
    presence_trace: az.InferenceData,
    positive_trace: az.InferenceData,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Per-status posterior of the *combined* expected-total multiplier
    vs the shuffle null.

    For each posterior draw and each row j, the predicted total is
        E_pred[j] = sigmoid(logit(p_null[j]) + beta + beta_sex * sex_c[j])
                    * mu_null[j] * exp(gamma + gamma_sex * sex_c[j])
    with each row's own focal sex (sex_c = +0.5 male, -0.5 female), and the
    multiplier is mean(E_pred) / mean(null_mean) over the rows of that
    status. Random effects are set to zero (typical recording and chase).
    Returns {0: samples, 1: samples}.
    """
    beta_per_status = _extract_status_draws(presence_trace)
    gamma_per_status = _extract_status_draws(positive_trace)
    beta_sex_all = _extract_scalar_draws(presence_trace, "sex_c")
    gamma_sex_all = _extract_scalar_draws(positive_trace, "sex_c")

    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    for k in (0, 1):
        n1, n2 = len(beta_per_status[k]), len(gamma_per_status[k])
        S = min(n1, n2)
        i1 = rng.choice(n1, size=S, replace=False) if n1 != S else np.arange(S)
        i2 = rng.choice(n2, size=S, replace=False) if n2 != S else np.arange(S)
        beta = beta_per_status[k].astype(float)[i1]
        beta_sex = beta_sex_all.astype(float)[i1]
        gamma = gamma_per_status[k].astype(float)[i2]
        gamma_sex = gamma_sex_all.astype(float)[i2]

        sub = df.loc[df["status"] == k]
        p0 = np.clip(sub["p_null_pos"].to_numpy(float), 1e-9, 1 - 1e-9)
        mu0 = np.maximum(sub["mu_null_pos"].to_numpy(float), 0.0)
        sex_c = np.where(sub["focal_sex"].to_numpy() == "m", 0.5, -0.5)
        denom = float(np.mean(sub["null_mean"].to_numpy(float))) + 1e-12
        lp0 = np.log(p0 / (1 - p0))

        lin_p = lp0[None, :] + beta[:, None] + beta_sex[:, None] * sex_c[None, :]
        p_hat = 1.0 / (1.0 + np.exp(-lin_p))
        mu_hat = mu0[None, :] * np.exp(gamma[:, None] + gamma_sex[:, None] * sex_c[None, :])
        e_hat = p_hat * mu_hat
        out[k] = e_hat.mean(axis=1) / denom
    return out


def kde_curve(
    samples: np.ndarray,
    log_x: bool,
    bw: float = 0.25,
    n_grid: int = 400,
    pad: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """KDE on either linear or log-x. For log-x, smooth in log-space and
    return points in original x for plotting on a log axis."""
    s = np.asarray(samples)
    s = s[np.isfinite(s) & (s > 0 if log_x else np.ones_like(s, bool))]
    if log_x:
        ls = np.log(s)
        lo, hi = np.quantile(ls, [0.005, 0.995])
        span = hi - lo
        grid_l = np.linspace(lo - pad * span, hi + pad * span, n_grid)
        kde = gaussian_kde(ls, bw_method=bw)
        return np.exp(grid_l), kde(grid_l)
    lo, hi = np.quantile(s, [0.005, 0.995])
    span = hi - lo
    grid = np.linspace(lo - pad * span, hi + pad * span, n_grid)
    kde = gaussian_kde(s, bw_method=bw)
    return grid, kde(grid)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def setup_plot() -> dict[str, plt.Axes]:
    """Three columns (presence / intensity / combined). Top row: paired
    per-trial scatter. Bottom row: compact forest with floating KDE."""
    fig = plt.figure(figsize=(17.4 * cm, 8.0 * cm), dpi=200)

    # GridSpec with three component columns and two rows. Forest row is
    # roughly half the height of the scatter row -- it only holds two
    # rows plus a KDE strip, so it doesn't need much vertical space.
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.09,
        right=0.98,
        top=0.92,
        bottom=0.22,
        height_ratios=[1.0, 0.4],
        wspace=0.3,
        hspace=0.5,
    )

    ax_pres_top = fig.add_subplot(gs[0, 0])
    ax_int_top = fig.add_subplot(gs[0, 1])
    ax_comb_top = fig.add_subplot(gs[0, 2])

    ax_pres_bot = fig.add_subplot(gs[1, 0])
    ax_int_bot = fig.add_subplot(gs[1, 1])
    ax_comb_bot = fig.add_subplot(gs[1, 2])

    # Top row -- per-trial scatter setup.
    for ax in (ax_pres_top, ax_int_top, ax_comb_top):
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["null", "observed"])
        ax.tick_params(length=3, width=0.5, color=C_STRUCTURE, labelsize=8)
        ax.set_xlim(-0.45, 1.45)

    # Bottom row -- forest setup.
    for ax in (ax_pres_bot, ax_int_bot, ax_comb_bot):
        ax.set_xlim(*FOREST_XLIM)
        ax.set_xticks(FOREST_XTICKS)
        ax.set_xticklabels([f"{t:g}" for t in FOREST_XTICKS])
        ax.tick_params(length=3, width=0.5, color=C_STRUCTURE, labelsize=8)
        ax.set_yticks([])

    return {
        "fig": fig,
        "pres_top": ax_pres_top,
        "int_top": ax_int_top,
        "comb_top": ax_comb_top,
        "pres_bot": ax_pres_bot,
        "int_bot": ax_int_bot,
        "comb_bot": ax_comb_bot,
    }


# ---------------------------------------------------------------------------
# Per-trial paired plots (panels a, b, c)
# ---------------------------------------------------------------------------
def draw_paired_panel(
    ax: plt.Axes,
    paired: pd.DataFrame,
    ylabel: str,
    yscale: str = "linear",
    label_status: bool = False,
) -> None:
    rng = np.random.default_rng(42)

    for status, color in STATUS_COLOR.items():
        sub = paired.loc[paired["status"] == status]
        if len(sub) == 0:
            continue

        # Jitter the (status=0, status=1) sides slightly so loser and
        # winner trials don't sit exactly on top of each other when their
        # values coincide.
        x_offset = -0.08 if status == 0 else 0.08
        jitter = rng.normal(0, 0.015, size=len(sub))
        x_null = np.full(len(sub), 0.0) + x_offset + jitter
        x_obs = np.full(len(sub), 1.0) + x_offset + jitter

        for xn, xo, yn, yo in zip(x_null, x_obs, sub["null"], sub["obs"]):
            ax.plot([xn, xo], [yn, yo], color=color, lw=0.5, alpha=0.4, zorder=2)

        # Null dots: lighter (open-style by alpha), observed: solid.
        ax.scatter(
            x_null,
            sub["null"],
            s=12,
            color=color,
            alpha=0.45,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
        ax.scatter(
            x_obs,
            sub["obs"],
            s=12,
            color=color,
            alpha=1.0,
            edgecolors="k",
            linewidths=0.4,
            zorder=4,
        )

    if label_status:
        # Direct labels so it is clear that both statuses are plotted.
        for status, color in STATUS_COLOR.items():
            sub = paired.loc[paired["status"] == status]
            if len(sub) == 0:
                continue
            ax.text(
                1.22,
                float(np.median(sub["obs"])),
                "losers" if status == 0 else "winners",
                color=color,
                fontsize=8,
                va="center",
                ha="left",
            )
        ax.set_xlim(right=1.6)

    ax.set_ylabel(ylabel, fontsize=8.5, color=C_TEXT)
    if yscale == "log":
        ax.set_yscale("log")

    sns.despine(ax=ax, offset=3, trim=False)


# ---------------------------------------------------------------------------
# Forest plots (panels d, e, f) -- one component per panel
# ---------------------------------------------------------------------------
def draw_forest_panel(
    ax: plt.Axes,
    samples_per_status: dict[int, np.ndarray],
    xlabel: str,
    show_yticks: bool = True,
    kde_scale: float = 0.40,
    kde_offset: float = 0.15,
    kde_bw: float = 0.25,
) -> None:
    """Two-row forest (loser top, winner bottom) with floating KDE clouds.

    Styling matches the paper's prior hurdle figure: thin black HDI line,
    white-fill black-edge median dot, KDE filled in status colour with no
    edge stroke, dashed reference line at multiplier = 1.
    """
    ax.axvline(1.0, ls="--", lw=0.5, c=C_NEUTRAL, alpha=1, zorder=1)

    y_for_status = {0: 1.0, 1: 0.0}

    print("-" * 80)
    for status, samples in samples_per_status.items():
        s = samples[np.isfinite(samples)]
        if len(s) < 5:
            continue

        y = y_for_status[status]
        color = STATUS_COLOR[status]

        mean = float(np.median(s))  # point estimate: posterior median
        hdi = az.hdi(s, hdi_prob=0.95)
        lo, hi = float(hdi[0]), float(hdi[1])

        print("winner" if status > 0 else "loser")
        print(mean, hdi)

        kx, ky = kde_curve(s, log_x=False, bw=kde_bw)
        if ky.max() > 0:
            ky = ky / ky.max() * kde_scale

        in_range = (kx >= FOREST_XLIM[0]) & (kx <= FOREST_XLIM[1])
        kx, ky = kx[in_range], ky[in_range]

        ax.fill_between(
            kx,
            y + kde_offset,
            y + kde_offset + ky,
            color=color,
            alpha=0.55,
            lw=0,
            zorder=3,
        )

        ax.plot([lo, hi], [y, y], color="k", lw=0.5, alpha=1, zorder=4)
        ax.scatter(
            [mean],
            [y],
            marker="o",
            facecolor="w",
            edgecolor="k",
            s=8,
            lw=0.5,
            zorder=10,
        )

    ax.set_ylim(-0.35, 1.0 + kde_offset + kde_scale + 0.10)
    if show_yticks:
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["winner", "loser"], fontsize=8)
    else:
        ax.set_yticks([])

    ax.set_xlabel(xlabel, fontsize=8.5, color=C_TEXT)
    sns.despine(ax=ax, left=not show_yticks, trim=True, offset=3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def add_legend(fig: plt.Figure) -> None:
    """Single figure-level legend at the top, encoding status colour and
    the null/observed marker convention used in the top row."""
    import matplotlib.lines as mlines

    handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            markerfacecolor=C_LOSER,
            markeredgecolor="white",
            markeredgewidth=0.4,
            color=C_LOSER,
            alpha=1.0,
            markersize=5,
            lw=0.5,
            label="losers",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            markerfacecolor=C_WINNER,
            markeredgecolor="white",
            markeredgewidth=0.4,
            color=C_WINNER,
            alpha=1.0,
            markersize=5,
            lw=0.5,
            label="winners",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=C_NEUTRAL,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.45,
            markersize=5,
            label="null (shuffled)",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=C_NEUTRAL,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=1.0,
            markersize=5,
            label="observed",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.4,
        handletextpad=0.5,
    )


def main() -> None:
    df = load_scored()
    presence_trace = az.from_netcdf(PRESENCE_TRACE)
    positive_trace = az.from_netcdf(POSITIVE_TRACE)

    paired_pres = per_trial_paired(df, "presence")
    paired_int = per_trial_paired(df, "intensity")
    paired_comb = per_trial_paired(df, "combined")

    presence_mult = {
        k: np.exp(v) for k, v in _extract_status_draws(presence_trace).items()
    }
    positive_mult = {
        k: np.exp(v) for k, v in _extract_status_draws(positive_trace).items()
    }
    combined_mult = combined_multiplier_posterior(df, presence_trace, positive_trace)

    axes = setup_plot()
    fig = axes["fig"]

    draw_paired_panel(
        axes["pres_top"],
        paired_pres,
        ylabel=r"P(chirp in window)",
        label_status=True,
    )
    draw_paired_panel(
        axes["int_top"],
        paired_int,
        ylabel=r"mean chirp count ($N > 0$)",
    )
    draw_paired_panel(
        axes["comb_top"],
        paired_comb,
        ylabel=r"mean chirp count",
    )

    draw_forest_panel(
        axes["pres_bot"],
        presence_mult,
        xlabel="odds multiplier",
        show_yticks=True,
    )
    draw_forest_panel(
        axes["int_bot"],
        positive_mult,
        xlabel="rate multiplier",
        show_yticks=False,
    )
    draw_forest_panel(
        axes["comb_bot"],
        combined_mult,
        xlabel="combined multiplier",
        show_yticks=False,
    )

    # Column titles above the top row.
    title_pad = 6
    axes["pres_top"].set_title(
        "Presence",
        loc="center",
        pad=title_pad,
        fontsize=9.5,
        color=C_TEXT,
        fontweight="medium",
    )
    axes["int_top"].set_title(
        "Intensity",
        loc="center",
        pad=title_pad,
        fontsize=9.5,
        color=C_TEXT,
        fontweight="medium",
    )
    axes["comb_top"].set_title(
        "Combined",
        loc="center",
        pad=title_pad,
        fontsize=9.5,
        color=C_TEXT,
        fontweight="medium",
    )

    # Letter labels: a/b/c top row, d/e/f bottom.
    letter_subplots(
        [
            axes["pres_top"],
            axes["int_top"],
            axes["comb_top"],
            axes["pres_bot"],
            axes["int_bot"],
            axes["comb_bot"],
        ],
        letters=["A1", "B1", "C1", "A2", "B2", "C2"],
    )

    add_legend(fig)

    print(
        f"Per-trial cells (status x recording):\n"
        f"  presence:  {len(paired_pres)} cells "
        f"({(paired_pres['status'] == 0).sum()} loser / "
        f"{(paired_pres['status'] == 1).sum()} winner)\n"
        f"  intensity: {len(paired_int)} cells "
        f"({(paired_int['status'] == 0).sum()} loser / "
        f"{(paired_int['status'] == 1).sum()} winner)\n"
        f"  combined:  {len(paired_comb)} cells "
        f"({(paired_comb['status'] == 0).sum()} loser / "
        f"{(paired_comb['status'] == 1).sum()} winner)"
    )

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".svg"), dpi=300)
    print(f"Saved {OUT_FIG} and {OUT_FIG.with_suffix('.svg')}")

    plt.show()


if __name__ == "__main__":
    main()
