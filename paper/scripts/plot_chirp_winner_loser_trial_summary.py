# Ported from chirp_competition d040123 (src/plot_chirp_winner_loser_trial_summary.py).
# Changes: inputs from $PAPER_OUT/data/processed, figure to paper_config.figure(); seeded jitter.
import os
import json
from pathlib import Path

import arviz as az
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from paper_config import PROCESSED, SEED, figure
from style import (
    letter_subplots,
    loser_color,
    set_light_style,
    winner_color,
)

set_light_style()

bad_recordings = [
    "2020-05-21-10_00",
    "2020-05-25-10_00",
    "2020-05-28-10_00",
    "2020-06-10-10_00",
]

filename = Path(__file__).stem
# Model reported in the manuscript: trial-level random intercept only (v3).
MODEL_DIR = str(PROCESSED / "stats_bayesian_glmm_chirps_status_v3")
targetpath = figure(filename)
formats = ["png", "pdf", "svg"]

fm_colors = {"m": "w", "f": "grey"}
fm_alphas = {"m": 0.2, "f": 0.8}

ridgeplot_scatter_kwargs = dict(
    marker="o",
    edgecolor="k",
    facecolor="w",
    s=7,
    zorder=10,
    lw=0.5,
)
ridgeplot_line_kwargs = dict(lw=0.5, alpha=1, color="k")


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
def setup_plot():
    cm = 1 / 2.54
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.5 * cm, 10 * cm),
        dpi=300,
        height_ratios=[1, 0.25],
    )
    return fig, axes.flatten()


# ---------------------------------------------------------------------------
# Top panel: paired scatter + posterior KDEs + boxplots
# ---------------------------------------------------------------------------
def plot_data(ax):
    df = pd.read_csv(PROCESSED / "trial_summary_long.csv", index_col=0)
    pred_df = pd.read_csv(f"{MODEL_DIR}/predictions.csv")
    # v3 stores labels; map them to the codes used in trial_summary_long.csv
    pred_df["status"] = pred_df["status"].map({"loser": 0, "winner": 1})
    pred_df["sex"] = pred_df["sex"].map({"female": "f", "male": "m"})

    df = df[(df["draw"] == 0) & (~df["recording"].isin(bad_recordings))]

    chirps_win = df[df["status"] == 1]["chirps"].values
    chirps_lose = df[df["status"] == 0]["chirps"].values
    sex_win = df[df["status"] == 1]["sex"].values
    sex_lose = df[df["status"] == 0]["sex"].values
    winlose = np.vstack([chirps_win, chirps_lose]).T

    med_win = np.median(chirps_win)
    med_lose = np.median(chirps_lose)
    print(f"Median chirps win: {med_win}")
    print(f"Median chirps lose: {med_lose}")
    print(f"Median factor: {med_lose / med_win:.2f}")

    # paired scatter with connecting lines
    jitter = np.random.default_rng(SEED).uniform(-0.05, 0.05, len(winlose))
    for i, wl in enumerate(winlose):
        slope = np.diff(wl)
        positions = [0.25 + jitter[i], 0.75 + jitter[i]]
        for j, sex in enumerate([sex_win[i], sex_lose[i]]):
            ax.scatter(
                positions[j],
                wl[j],
                marker="o",
                edgecolor="k",
                facecolor=fm_colors[sex],
                s=6,
                lw=0.5,
                zorder=1000 if slope < 0 else -10,
            )
        ax.plot(positions, wl, color="lightgrey", lw=0.5, zorder=-20)

    # posterior KDE violins
    status_cfg = [
        (1, 0, -1, winner_color),
        (0, 1, 1, loser_color),
    ]
    widthscaler = 0.3

    for stat, pos, sign, col in status_cfg:
        # first pass: find max density for normalization
        max_density = 0
        for sex in ("m", "f"):
            data = pred_df[(pred_df["status"] == stat) & (pred_df["sex"] == sex)]
            mu = data["mu"].values
            lo, hi = data["hdi_2.5"].unique()[0], data["hdi_97.5"].unique()[0]
            mu = mu[(mu > lo) & (mu < hi)]
            kde = gaussian_kde(mu, bw_method=0.1)
            dist_x = np.linspace(lo, hi, 1000)
            max_density = max(max_density, np.max(kde(dist_x)))

        # second pass: plot
        for sex in ("m", "f"):
            data = pred_df[(pred_df["status"] == stat) & (pred_df["sex"] == sex)]
            mu = data["mu"].values
            lo, hi = data["hdi_2.5"].unique()[0], data["hdi_97.5"].unique()[0]
            mu = mu[(mu > lo) & (mu < hi)]

            dist_x = np.linspace(lo, hi, 1000)
            kde = gaussian_kde(mu, bw_method=0.1)
            dist_y = kde(dist_x) / max_density * widthscaler

            ax.fill_betweenx(
                dist_x,
                np.zeros_like(dist_x) + pos,
                dist_y * sign + pos,
                color=col,
                alpha=fm_alphas[sex],
                lw=0,
            )
            lpx = np.pad(dist_y * sign + pos, (1, 1), constant_values=pos)
            lpy = np.pad(dist_x, (1, 1), mode="edge")
            ax.plot(lpx, lpy, color="k", lw=0.5)
            ax.plot(np.zeros_like(dist_x) + pos, dist_x, color="k", lw=0.5)

    # boxplots
    ax.boxplot(
        [chirps_win, chirps_lose],
        positions=[0.15, 0.85],
        widths=0.05,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="k", lw=0.5),
        whiskerprops=dict(color="k", lw=0.5),
        capprops=dict(color="k", lw=0.5),
        medianprops=dict(color="k", lw=0.5),
        flierprops=dict(markersize=0),
    )

    n_pairs = len(df) // 2
    ax.text(
        0.03,
        0.98,
        f"n = {n_pairs} competitions",
        ha="left",
        va="top",
        fontsize=8,
        transform=ax.transAxes,
    )

    # legend: data-point markers + KDE opacity by sex
    female_marker = plt.Line2D(
        [],
        [],
        marker="o",
        color="none",
        markerfacecolor="grey",
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=3,
        lw=0,
        label="$\\female$ observed",
    )
    male_marker = plt.Line2D(
        [],
        [],
        marker="o",
        color="none",
        markerfacecolor="w",
        markeredgewidth=0.5,
        markeredgecolor="k",
        markersize=3,
        lw=0,
        label="$\\male$ observed",
    )
    female_kde = mpatches.Patch(
        facecolor="grey",
        alpha=fm_alphas["f"],
        edgecolor="k",
        lw=0.5,
        label="$\\female$ posterior",
    )
    male_kde = mpatches.Patch(
        facecolor="grey",
        alpha=fm_alphas["m"],
        edgecolor="k",
        lw=0.5,
        label="$\\male$ posterior",
    )
    ax.legend(
        handles=[
            female_marker,
            male_marker,
            female_kde,
            male_kde,
        ],
        loc="upper left",
        fontsize=6,
        frameon=False,
        handlelength=1,
        handleheight=1,
        bbox_to_anchor=(0.0, 0.92),
        ncol=1,
    )

    ax.set_xticklabels(["Winner", "Loser"])
    ax.set_ylabel("Chirp count")
    ax.set_xlabel("")


# ---------------------------------------------------------------------------
# Bottom panel: forest plot of posterior effect sizes
# ---------------------------------------------------------------------------
def _compute_kde(data, bw=0.1, scaler=0.5, samples=1000, density_thresh=0.02):
    kde = gaussian_kde(data, bw_method=bw)
    x = np.linspace(np.min(data), np.max(data), samples)
    y = kde(x)
    y = y / np.max(y) * scaler
    mask = y > density_thresh
    return x[mask], y[mask]


def _compute_hdi(data, hdi_prob=0.95):
    return np.median(data), az.hdi(data, hdi_prob=hdi_prob)


def plot_forest(ax):
    with open(f"{MODEL_DIR}/summary.json") as f:
        json.load(f)

    posterior = az.from_netcdf(f"{MODEL_DIR}/posterior.nc")
    posterior = az.concat(posterior, dim="chain")

    post_sex = np.exp(posterior.posterior["sex"].to_numpy().flatten())
    post_size = np.exp(posterior.posterior["size_z"].to_numpy().flatten())

    da_status = posterior.posterior["status"]
    status_dim = [d for d in da_status.dims if d not in ("chain", "draw")][0]
    post_status = np.exp(-da_status.sel({status_dim: "winner"}).to_numpy().flatten())

    scaler = 0.4
    kde_kwargs = dict(lw=0, alpha=0.5, color="grey")
    var_names = ["Status[loser]", "Sex[male]", "Size"]
    posteriors = [post_status, post_sex, post_size]
    ypos = [0, 1, 2]

    for y, draws in zip(ypos, posteriors):
        kx, ky = _compute_kde(draws, scaler=scaler, samples=1000)
        ax.fill_between(
            kx,
            np.zeros_like(kx) + y,
            ky + y,
            **kde_kwargs,
            label="Posterior PDF" if y == 0 else None,
        )
        mu, hdi = _compute_hdi(draws)
        ax.scatter(
            [mu],
            [y - 0.2],
            **ridgeplot_scatter_kwargs,
            label="Posterior median" if y == 0 else None,
        )
        ax.plot(
            hdi,
            [y - 0.2, y - 0.2],
            **ridgeplot_line_kwargs,
            label=r"95\% HDI" if y == 0 else None,
        )

    ax.axvline(1, c="darkgrey", lw=0.5, ls="--", zorder=-100, label="No effect")
    ax.set_yticks(ypos)
    ax.set_yticklabels(var_names)
    ax.set_ylabel("")
    ax.set_xlabel("Multiplicative effect on expected chirp count")
    ax.set_ylim(-0.5, len(ypos) - 1 + scaler)
    ax.legend(
        loc="upper right",
        fontsize=6,
        frameon=False,
        handlelength=1,
        bbox_to_anchor=(0.9, 1.1),
        ncol=2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig, axes = setup_plot()
    plot_data(axes[0])
    plot_forest(axes[1])

    letter_subplots(xoffset=[-0.3, -0.3], yoffset=[0.98, 0.95])

    sns.despine(trim=True, offset=5)
    plt.subplots_adjust(left=0.28, right=1, top=0.99, bottom=0.13, hspace=0.3)

    for fmt in formats:
        fig.savefig(targetpath.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
