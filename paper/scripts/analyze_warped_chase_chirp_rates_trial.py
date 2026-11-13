"""
Analyze the loser chirp rate in time-warped chase windows.

Each chase (agonistic onset, offset) pair defines a normalized coordinate
u = (t - t_on) / D, where D = t_off - t_on. The analysis window runs
from u = -1 to u = 2 so the display splits into pre-chase, during-chase,
and post-chase thirds of equal width in u.

Rate is computed per chase by Gaussian-KDE of its normalized chirp lags
and division by D (a normalized bin of width du is du*D seconds of real
time, so density-in-u / D = rate-in-Hz). Per-chase rates are averaged
across chases, with trial-level leave-out resampling matching
analyze_event_time_rates_trial.py.

Null model: each chase window is shifted by an independent draw from
U(-J, J) with D preserved, then chirps are re-normalized against the
shifted window. This is the direct analogue of the `jittered_centers`
null in analyze_event_time_rates_trial.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from analyze_event_time_rates_trial import (
    _leave_out_indices,
    gaussian_kde,
    is_good_trial,
    load_data,
)
from logger import get_logger

logger = get_logger(__name__)


# --- Analysis parameters ------------------------------------------------
# Analysis window in normalized chase-duration units (pre-chase, during-
# chase, and post-chase thirds).
WARP_WINDOW = (-1.0, 2.0)
# Grid resolution in normalized units.
WARP_GRID_DU = 0.005
# KDE bandwidth in normalized units.
WARP_SIGMA_U = 0.02
# Minimum chase duration (s). Chases below this are dropped because their
# per-chase rates blow up numerically (1 chirp / tiny D = huge Hz).
MIN_CHASE_DURATION_S = 0.5
# Active phase (s). Chase onsets outside this range are dropped; mirrors
# analyze_event_time_rates_trial.py.
ACTIVE_PHASE_S = (0, 3 * 60 * 60)
# Leave-out fraction for the bootstrap at the TRIAL level.
LEAVE_OUT_FRAC = 0.1
N_RESAMPLES = 100
# Jitter amplitude (s) for the null: the entire chase window is shifted
# by U(-J, J). Matches CENTER_JITTER_AMPLITUDE_S in the main module.
JITTER_AMPLITUDE_S = 120


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def extract_warped_chirp_times(chase_windows, chirp_times, u_lo, u_hi):
    """Return per-chase normalized chirp lags and chase durations.

    Parameters
    ----------
    chase_windows : array-like of shape (M, 2)
        (t_on, t_off) pairs in seconds.
    chirp_times : array-like
        All chirp times of the trial, in seconds.
    u_lo, u_hi : float
        Normalized window bounds (inclusive).

    Returns
    -------
    per_chase_u : list of np.ndarray
        Normalized chirp lags per chase.
    per_chase_D : list of float
        Chase durations in seconds.
    """
    chirp_times = np.asarray(chirp_times)
    per_chase_u = []
    per_chase_D = []
    for t_on, t_off in chase_windows:
        D = t_off - t_on
        if D <= 0:
            continue
        t_min = t_on + u_lo * D
        t_max = t_on + u_hi * D
        mask = (chirp_times >= t_min) & (chirp_times <= t_max)
        u = (chirp_times[mask] - t_on) / D
        per_chase_u.append(u)
        per_chase_D.append(float(D))
    return per_chase_u, per_chase_D


def per_chase_rate(u_grid, u_chase, D, sigma_u):
    """KDE of normalized chirp lags, converted to chirps/s.

    Density in normalized units has a du-integral equal to the chirp
    count. Dividing by D rescales to real-time rate (chirps/s), because
    a normalized bin of width du corresponds to du*D seconds.
    """
    if len(u_chase) == 0:
        return np.zeros_like(u_grid, dtype=float)
    density = gaussian_kde(u_grid, np.asarray(u_chase), sigma=sigma_u)
    return density / D


# ---------------------------------------------------------------------------
# Bootstraps
# ---------------------------------------------------------------------------
def observed_warped_rate_bootstrap(
    per_trial_chase_u,
    per_trial_chase_D,
    u_grid,
    sigma_u,
    n_bootstraps,
    leave_out_frac,
):
    """Trial-level leave-out bootstrap of the mean per-chase warped rate.

    Resamples at the trial level to respect the independence assumption
    (chases nested within trials). Each iteration drops a fraction of
    trials and averages over all chases in the kept trials.

    Returns
    -------
    bootstrapped : np.ndarray, shape (n_bootstraps, len(u_grid))
        Bootstrapped mean rate curves, chirps/s.
    per_chase_rates : np.ndarray, shape (n_chases, len(u_grid))
        Per-chase rate curves, chirps/s.
    per_chase_trial_idx : np.ndarray, shape (n_chases,)
        Trial index for each chase (same order as per_chase_rates).
    """
    n_trials = len(per_trial_chase_u)

    # Precompute per-chase rate curves.
    per_chase_rates = []
    per_chase_trial_idx = []
    for t_idx in range(n_trials):
        for u_chase, D in zip(per_trial_chase_u[t_idx], per_trial_chase_D[t_idx]):
            per_chase_rates.append(per_chase_rate(u_grid, u_chase, D, sigma_u))
            per_chase_trial_idx.append(t_idx)

    if not per_chase_rates:
        return (
            np.zeros((n_bootstraps, len(u_grid))),
            np.zeros((0, len(u_grid))),
            np.array([], dtype=int),
        )

    per_chase_rates = np.array(per_chase_rates)
    per_chase_trial_idx = np.array(per_chase_trial_idx)

    bootstrapped = np.zeros((n_bootstraps, len(u_grid)))
    for b in range(n_bootstraps):
        logger.info(f"Observed warped bootstrap {b + 1}/{n_bootstraps}")
        keep_trials = _leave_out_indices(n_trials, leave_out_frac)
        mask = np.isin(per_chase_trial_idx, keep_trials)
        if not mask.any():
            continue
        bootstrapped[b] = per_chase_rates[mask].mean(axis=0)

    return bootstrapped, per_chase_rates, per_chase_trial_idx


def null_warped_rate_bootstrap(
    per_trial_chase_windows,
    per_trial_chirps,
    u_grid,
    sigma_u,
    u_lo,
    u_hi,
    jitter_amplitude_s,
    n_bootstraps,
    leave_out_frac,
):
    """Jittered-window null with trial-level leave-out.

    Each iteration: drop a fraction of trials, then for every surviving
    chase shift (t_on, t_off) by an independent U(-J, J) draw (D fixed),
    re-extract normalized chirps against the shifted window, compute
    per-chase rates, and average.
    """
    n_trials = len(per_trial_chase_windows)
    bootstrapped = np.zeros((n_bootstraps, len(u_grid)))
    for b in range(n_bootstraps):
        logger.info(f"Null warped bootstrap {b + 1}/{n_bootstraps}")
        keep_trials = _leave_out_indices(n_trials, leave_out_frac)

        iter_chase_rates = []
        for t_idx in keep_trials:
            windows = np.asarray(per_trial_chase_windows[t_idx])
            if len(windows) == 0:
                continue
            chirps = per_trial_chirps[t_idx]
            eps = np.random.uniform(
                -jitter_amplitude_s, jitter_amplitude_s, size=len(windows)
            )
            jittered = windows + eps[:, None]
            u_list, D_list = extract_warped_chirp_times(jittered, chirps, u_lo, u_hi)
            for u_chase, D in zip(u_list, D_list):
                iter_chase_rates.append(per_chase_rate(u_grid, u_chase, D, sigma_u))

        if not iter_chase_rates:
            continue
        bootstrapped[b] = np.mean(iter_chase_rates, axis=0)

    return bootstrapped


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(path: Path):
    trial_summary = pd.read_csv(path / "trial_summary.csv")

    per_trial_chase_windows = []
    per_trial_lchirps = []
    per_trial_identifiers = []

    for i in range(len(trial_summary)):
        trial_row = trial_summary.iloc[i]
        trial_path = path / trial_row["recording"]

        # We need BORIS events (agonistic on/off) for this analysis, so
        # do NOT skip them.
        if not is_good_trial(trial_path, trial_row, skip_ag_events=False):
            continue

        logger.info(f"Loading data for {trial_row['recording']}")
        (
            _contact_t,
            ag_on_t,
            ag_off_t,
            loser_chirp_times,
            _winner_chirp_times,
            _loser_rise_times,
            _winner_rise_times,
        ) = load_data(trial_path, trial_row, skip_ag_events=False)

        ag_on_t = np.asarray(ag_on_t)
        ag_off_t = np.asarray(ag_off_t)
        if len(ag_on_t) == 0:
            per_trial_chase_windows.append(np.empty((0, 2)))
            per_trial_lchirps.append(np.asarray(loser_chirp_times))
            per_trial_identifiers.append(trial_row["recording"])
            continue

        # Active-phase filter on the onset + minimum-duration filter.
        start, end = ACTIVE_PHASE_S
        durations = ag_off_t - ag_on_t
        mask = (
            (ag_on_t >= start) & (ag_on_t < end) & (durations >= MIN_CHASE_DURATION_S)
        )
        windows = np.stack([ag_on_t[mask], ag_off_t[mask]], axis=1)

        per_trial_chase_windows.append(windows)
        per_trial_lchirps.append(np.asarray(loser_chirp_times))
        per_trial_identifiers.append(trial_row["recording"])

    # Drop trials with no usable chases.
    keep = [i for i, w in enumerate(per_trial_chase_windows) if len(w) > 0]
    if len(keep) < len(per_trial_chase_windows):
        logger.info(
            f"Dropping {len(per_trial_chase_windows) - len(keep)} trials with "
            f"no chases passing filters."
        )
    per_trial_chase_windows = [per_trial_chase_windows[i] for i in keep]
    per_trial_lchirps = [per_trial_lchirps[i] for i in keep]
    per_trial_identifiers = [per_trial_identifiers[i] for i in keep]

    n_trials = len(per_trial_chase_windows)
    n_chases = int(sum(len(w) for w in per_trial_chase_windows))
    logger.info(f"Retained {n_trials} trials with {n_chases} chases.")

    # Build the normalized grid.
    u_lo, u_hi = WARP_WINDOW
    u_grid = np.arange(u_lo, u_hi + WARP_GRID_DU, WARP_GRID_DU)

    # Per-chase normalized chirp lags (observed).
    per_trial_chase_u = []
    per_trial_chase_D = []
    for windows, chirps in zip(per_trial_chase_windows, per_trial_lchirps):
        u_list, D_list = extract_warped_chirp_times(windows, chirps, u_lo, u_hi)
        per_trial_chase_u.append(u_list)
        per_trial_chase_D.append(D_list)

    logger.info("Starting observed bootstrap ------------------------")
    (
        observed_rate,
        per_chase_rates,
        per_chase_trial_idx,
    ) = observed_warped_rate_bootstrap(
        per_trial_chase_u,
        per_trial_chase_D,
        u_grid,
        WARP_SIGMA_U,
        N_RESAMPLES,
        LEAVE_OUT_FRAC,
    )

    logger.info("Starting null bootstrap ------------------------")
    null_rate = null_warped_rate_bootstrap(
        per_trial_chase_windows,
        per_trial_lchirps,
        u_grid,
        WARP_SIGMA_U,
        u_lo,
        u_hi,
        JITTER_AMPLITUDE_S,
        N_RESAMPLES,
        LEAVE_OUT_FRAC,
    )

    # Per-trial rate = mean across that trial's per-chase rates. One row
    # per trial, matching the semantics of `per_trial_rates` in the main
    # figure (one row per independent unit of analysis).
    per_trial_rates = np.zeros((n_trials, len(u_grid)))
    for t in range(n_trials):
        mask = per_chase_trial_idx == t
        if mask.any():
            per_trial_rates[t] = per_chase_rates[mask].mean(axis=0)

    # --- Save ----------------------------------------------------------
    datapath = Path("data/processed/event_time_rates")
    datapath.mkdir(parents=True, exist_ok=True)

    flat_D = [D for D_list in per_trial_chase_D for D in D_list]
    flat_u = [u.tolist() for u_list in per_trial_chase_u for u in u_list]

    data_json = {
        "description": "lchirps warped in chase windows (u = (t - t_on) / D)",
        "warp_window": list(WARP_WINDOW),
        "sigma_u": WARP_SIGMA_U,
        "min_chase_duration_s": MIN_CHASE_DURATION_S,
        "jitter_amplitude_s": JITTER_AMPLITUDE_S,
        "u_grid": u_grid.tolist(),
        "jackknifed_eventrate": observed_rate.tolist(),
        "bootstrapped_baseline_jittered": null_rate.tolist(),
        "per_trial_rates": per_trial_rates.tolist(),
        "per_chase_rates": per_chase_rates.tolist(),
        "per_chase_trial_idx": per_chase_trial_idx.tolist(),
        "per_chase_durations": flat_D,
        "per_chase_normalized_chirp_times": flat_u,
        "trial_identifiers": per_trial_identifiers,
        "n_trials": n_trials,
        "n_chases": n_chases,
    }

    with open(datapath / "lchirps@warped_chase.json", "w") as f:
        json.dump(data_json, f)

    # --- Minimal sanity plot ------------------------------------------
    # Top: observed + null rate curves with 95% bands, dashed verticals
    # at u=0 (onset) and u=1 (offset).
    # Bottom: per-trial rate heatmap (one row per trial, row-peak
    # normalized), mimicking the heatmap panel in the main figure.
    obs_q = np.percentile(observed_rate, [2.5, 50, 97.5], axis=0) * 60
    null_q = np.percentile(null_rate, [2.5, 50, 97.5], axis=0) * 60

    fig, (ax_rate, ax_hm) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )

    # --- Rate panel ---
    ax_rate.fill_between(
        u_grid, null_q[0], null_q[2], color="tab:green", alpha=0.3, lw=0
    )
    ax_rate.plot(
        u_grid,
        null_q[1],
        color="tab:green",
        label=f"null: jittered window (J={JITTER_AMPLITUDE_S}s)",
    )
    ax_rate.fill_between(u_grid, obs_q[0], obs_q[2], color="tab:blue", alpha=0.3, lw=0)
    ax_rate.plot(u_grid, obs_q[1], color="tab:blue", label="observed")
    ax_rate.axvline(0, c="k", lw=0.5, ls="--")
    ax_rate.axvline(1, c="k", lw=0.5, ls="--")
    ax_rate.set_xlim(u_lo, u_hi)
    ax_rate.legend(frameon=False, fontsize="small", loc="upper right")
    ax_rate.set_title(
        f"Loser chirps in time-warped chase windows  "
        f"({n_trials} trials, {n_chases} chases, "
        f"D ≥ {MIN_CHASE_DURATION_S}s, σ_u={WARP_SIGMA_U})"
    )
    ax_rate.set_ylabel("Chirp rate (chirps/min)")

    # --- Heatmap panel ---
    # Row-peak normalization so temporal profiles are comparable across
    # trials with different absolute rates (matches the main figure).
    hm = per_trial_rates / (per_trial_rates.max(axis=1, keepdims=True) + 1e-12)
    cmap = sns.color_palette("mako", as_cmap=True)
    n_zero = int(np.sum(per_trial_rates.sum(axis=1) == 0))
    ax_hm.imshow(
        hm,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[u_lo, u_hi, 0, n_trials],
        interpolation="none",
    )
    ax_hm.axvline(0, c="w", lw=0.5, ls="--", alpha=0.6)
    ax_hm.axvline(1, c="w", lw=0.5, ls="--", alpha=0.6)
    ax_hm.set_ylabel("Trial")
    ax_hm.set_xlabel("Normalized chase time  (0 = onset, 1 = offset)")
    if n_zero:
        ax_hm.text(
            0.98,
            0.02,
            f"{n_zero}/{n_trials} trials with no chirps",
            transform=ax_hm.transAxes,
            ha="right",
            va="bottom",
            color="w",
            fontsize="x-small",
        )

    plt.tight_layout()
    plt.savefig(datapath / "lchirps@warped_chase.png")
    plt.show()


if __name__ == "__main__":
    typer.run(main)
