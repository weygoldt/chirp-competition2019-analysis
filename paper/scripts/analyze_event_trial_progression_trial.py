"""
Analysis of temporal allocation of events in trials.
Uses a single trial as the statistically independent unit of analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
import random
from scipy.stats import gaussian_kde

from analyze_event_time_rates_trial import load_and_convert_boris_events
from analyze_event_time_rates_trial import gaussian_kde as self_gaussian_kde
from logger import get_logger

female_color, male_color = "#e74c3c", "#3498db"
logger = get_logger(__name__)


# Bandwidth of the Gaussian kernel used to smooth event trains into
# trial-wise rate functions (see kernel_convolve). Units: seconds.
KERNEL_SIGMA_S = 180

# Number of resampling iterations used for both the shuffle null
# (compute_null_cumulative_rates) and the trial bootstrap
# (compute_observed_cis).
N_RESAMPLES = 1000


def scotts_bandwidth_factor(data):
    """Compute Scott's bandwidth factor for Gaussian KDE."""
    n = len(data)
    if n < 2:
        return 1.0  # Default factor for insufficient data
    std_dev = np.std(data, ddof=1)
    factor = n ** (-1 / 5) * std_dev
    return factor


def shuffle_within_interval(events, interval):
    start, end = interval
    num_events = len(events)
    # Generate random event times uniformly within the interval
    shuffled_events = np.random.uniform(start, end, num_events)
    return shuffled_events


def events_in_phase(event_times, active_phase):
    """Restrict each trial's events to the active phase.

    Events outside the phase would otherwise leak into it through the kernel
    tails and inflate the event count used for the uniform null. Trials
    without any event inside the phase are dropped.
    """
    start, end = active_phase
    clipped = [np.asarray(e)[(np.asarray(e) >= start) & (np.asarray(e) <= end)] for e in event_times]
    return [e for e in clipped if len(e) > 0]


def normalized_cumsum(rate, time_points):
    """Cumulative integral of `rate`, rescaled to end at 1 (NaN if all zero)."""
    cumulative_rate = np.cumsum(rate) * (time_points[1] - time_points[0])
    if cumulative_rate[-1] <= 0:
        return np.full_like(cumulative_rate, np.nan)
    return cumulative_rate / cumulative_rate[-1]


def kernel_convolve(events, time_points, sigma=KERNEL_SIGMA_S):
    """Smooth an event train onto `time_points` with a Gaussian kernel.

    Parameters
    ----------
    events : 1d array-like
        Event times (seconds).
    time_points : 1d array-like
        Time grid on which to evaluate the rate (seconds).
    sigma : float
        Gaussian kernel bandwidth in seconds. Defaults to KERNEL_SIGMA_S.
    """
    rate = self_gaussian_kde(time_points, events, sigma=sigma)
    return rate


def compute_null_cumulative_rates(event_times, active_phase, num_shuffles, time_points):
    null_cumulative_rates = []
    event_times = events_in_phase(event_times, active_phase)

    for _ in range(num_shuffles):
        shuffled_relative_rates = []

        for trial_events in event_times:
            # Shuffle chirp times within the active phase
            shuffled_events = shuffle_within_interval(trial_events, active_phase)

            # Compute relative event rate with kernel convolution
            if len(trial_events) > 1:
                shuffled_rate = kernel_convolve(shuffled_events, time_points)
            else:
                shuffled_rate = np.zeros_like(time_points)

            # Normalize the rate to ensure it sums to 1 over the active phase
            # global_rate = len(trial_events) / (active_phase[1] - active_phase[0])
            # normalized_rate = shuffled_rate / global_rate
            # normalized_rate = shuffled_rate / np.sum(shuffled_rate)
            normalized_rate = shuffled_rate / (np.sum(shuffled_rate) + 1e-6)

            # Store the normalized rate
            shuffled_relative_rates.append(normalized_rate)

        # Average relative rates across trials
        median_shuffled_rate = np.median(shuffled_relative_rates, axis=0)

        norm_cumulative_rate = normalized_cumsum(median_shuffled_rate, time_points)

        # Store the cumulative rate for this shuffle
        null_cumulative_rates.append(norm_cumulative_rate)

    # Convert list to array for easier manipulation
    null_cumulative_rates = np.array(null_cumulative_rates)

    return null_cumulative_rates


def compute_observed_cis(event_times, active_phase, time_points, num_bootstraps):
    bootstrap_cumulative_rates = []
    event_times = events_in_phase(event_times, active_phase)

    for _ in range(num_bootstraps):
        # Resample trials with replacement
        resampled_trials = random.choices(event_times, k=len(event_times))

        relative_rates = []
        for trial_events in resampled_trials:
            # Compute relative event rate for the resampled trial
            if len(trial_events) > 1:
                trial_rate = kernel_convolve(trial_events, time_points)
            else:
                trial_rate = np.zeros_like(time_points)

            # global_rate = len(trial_events) / (active_phase[1] - active_phase[0])
            # normalized_rate = trial_rate / global_rate
            normalized_rate = trial_rate / (np.sum(trial_rate) + 1e-6)
            relative_rates.append(normalized_rate)

        # Median across resampled trials (matches aggregator used for the null)
        median_rate = np.median(relative_rates, axis=0)

        norm_cumulative_rate = normalized_cumsum(median_rate, time_points)

        # Store cumulative rate for this bootstrap sample
        bootstrap_cumulative_rates.append(norm_cumulative_rate)

    # Convert list to array for easier manipulation
    bootstrap_cumulative_rates = np.array(bootstrap_cumulative_rates)

    return bootstrap_cumulative_rates


def relative_rate_progression(all_event_t, title=""):
    epsilon = 1e-6
    stop_t_all = 6 * 60 * 60
    stop_t_active = 3 * 60 * 60
    snippet_len = 5 * 60

    snippet_starts = np.arange(0, stop_t_all, snippet_len)
    all_snippet_ratios = []
    all_all_snippet_kdes = []
    all_active_snippet_kdes = []
    all_active_snippet_kdes = []
    conv_t_all = np.arange(0, stop_t_all, 1.0, dtype=float)
    conv_t_active = np.arange(0, stop_t_active, 1.0, dtype=float)

    trial_event_count = []

    for event_t in all_event_t:
        # Why does till divide the inactive time by the snippet length????
        # expected_snippet_count = len(event_t[event_t <= stop_t]) / (
        #     stop_t / snippet_len
        # )

        n_events_all = len(event_t[event_t <= stop_t_all])
        n_events_active = len(event_t[event_t <= stop_t_active])

        """
        if len(event_t) == 0:
        if n_events < 2:
            # kde_array = np.zeros_like(conv_t)
            # all_snippet_kdes.append(kde_array)
            continue
        """

        all_event_rate = n_events_all / stop_t_all
        active_event_rate = n_events_active / stop_t_active

        # print(f"Expected_snippet_count: {global_event_rate}")

        snippet_ratios = []
        for s0 in snippet_starts:
            snippet_count = len(
                event_t[(event_t >= s0) & (event_t < (s0 + snippet_len))]
            )
            event_rate = snippet_count / snippet_len
            snippet_ratios.append(event_rate / (all_event_rate + epsilon))

        all_snippet_ratios.append(snippet_ratios)

        if n_events_all < 2:
            all_kde_array = np.zeros_like(conv_t_all)
        else:
            # event rate estimate by self-build kde function
            # all_kde_array = gaussian_kde(
            #     conv_t_all, event_t[event_t <= stop_t_all], bandwidth=snippet_len
            # )
            # active_kde_array = gaussian_kde(
            #     conv_t_active, event_t[event_t <= stop_t_active], bandwidth=snippet_len
            # )
            # event rate estimate by scaling PDF by the number of events
            all_kde_array = kernel_convolve(event_t[event_t <= stop_t_all], conv_t_all)

        if n_events_active < 2:
            active_kde_array = np.zeros_like(conv_t_active)
        else:
            active_kde_array = kernel_convolve(
                event_t[event_t <= stop_t_active], conv_t_active
            )

        # Why am I doing this???
        # Normalize by number of events to get event rate
        # all_kde_array *= n_events_all
        # active_kde_array *= n_events_active

        # normalize by global event rate
        all_kde_array /= all_event_rate + epsilon  # normalize rate by global rate
        active_kde_array /= active_event_rate + epsilon

        # normalize by sum
        # all_kde_array /= np.sum(all_kde_array) + epsilon
        # active_kde_array /= np.sum(active_kde_array) + epsilon

        # print(f"KDE mean: {np.mean(kde_array)}")

        all_all_snippet_kdes.append(all_kde_array)
        all_active_snippet_kdes.append(active_kde_array)
        trial_event_count.append(len(event_t))

    all_snippet_ratio = np.array(all_snippet_ratios)
    all_all_snippet_kdes = np.array(all_all_snippet_kdes)
    all_active_snippet_kdes = np.array(all_active_snippet_kdes)

    fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])

    # plot_t = np.repeat(snippet_starts, 2)
    # plot_t[1::2] += snippet_len
    # plot_t = snippet_starts + snippet_len / 2

    # for event_ratios in all_snippet_ratios:
    #     plot_ratios = np.repeat(event_ratios, 2)
    #     ax.plot(plot_t / 3600, plot_ratios, color="grey", lw=1, alpha=0.5)
    #     ax.plot(snippet_starts + snippet_len / 2, event_ratios)
    #     ax.plot(plot_t / 3600, event_ratios, color="grey", lw=1, alpha=0.5)
    # mean_ratio = np.mean(all_snippet_ratios, axis=0)
    # plot_mean_ratio = np.repeat(mean_ratio, 2)
    # ax.plot(plot_t / 3600, plot_mean_ratio, color="k", lw=3)
    # ax.plot(plot_t / 3600, np.ones_like(plot_t), linestyle="dotted", lw=2, color="k")
    # ax.plot(plot_t / 3600, mean_ratio, color="k", lw=3)

    # for kde_array in all_snippet_kdes:
    #     plt.plot(conv_t / 3600, kde_array, color="lightgrey", lw=1, alpha=1)
    #     plt.title(f"{title} KDE")
    #     plt.show()

    # all_active_snippet_kdes = np.cumsum(all_active_snippet_kdes, axis=1)
    all_kde_qrts = np.percentile(all_all_snippet_kdes, [25, 50, 75], axis=0)
    active_kde_qrts = np.percentile(all_active_snippet_kdes, [25, 50, 75], axis=0)

    # mean_ratio = np.mean(all_snippet_kdes, axis=0)
    # std_ratio = np.std(all_snippet_kdes, axis=0)
    ax.plot(conv_t_active / 3600, active_kde_qrts[1], color="k", lw=4)
    ax.fill_between(
        conv_t_active / 3600,
        active_kde_qrts[0],
        active_kde_qrts[2],
        color="k",
        alpha=0.2,
    )
    ax.plot(
        conv_t_active / 3600,
        np.ones_like(conv_t_active),
        linestyle="dotted",
        lw=2,
        color="k",
    )

    ax.set_xlabel("time [h]", fontsize=12)
    ax.set_ylabel("norm. event rate", fontsize=12)
    ax.set_title(title)
    ax.tick_params(labelsize=10)

    # ax.set_xlim(0, 3)
    # ax.set_ylim(0, 5)

    plt.savefig(f"figures/event_meta/{title}_progression.png", dpi=300)
    plt.show()
    plt.close()
    # exit()

    # x = np.hstack(all_snippet_ratio)
    # y = np.hstack(np.tile(snippet_starts, (all_snippet_ratio.shape[0], 1)))

    # r, p = scp.pearsonr(x, y)

    # collect kde ratios and binned ratios to save for plotting
    # ensure json serializability
    # all_snippet_ratios = all_snippet_ratios
    # plot_t = plot_t.tolist()
    all_all_snippet_kdes = [x.tolist() for x in all_all_snippet_kdes]
    all_active_snippet_kdes = [x.tolist() for x in all_active_snippet_kdes]
    conv_t_all = conv_t_all.tolist()
    conv_t_active = conv_t_active.tolist()

    data = dict(
        # binned_ratios=all_snippet_ratios,
        # binned_times=plot_t,
        kde_ratios=all_all_snippet_kdes,
        kde_times=conv_t_all,
        active_kde_ratios=all_active_snippet_kdes,
        active_kde_times=conv_t_active,
        event_count=trial_event_count,
    )

    return data


def analyze(event_times, active_phase, num_shuffles, time_points, title, out_dir):
    print(f"Analyzing {title}")
    baseline = compute_null_cumulative_rates(
        event_times,
        active_phase,
        num_shuffles,
        time_points,
    )
    observed = compute_observed_cis(
        event_times,
        active_phase,
        time_points,
        num_shuffles,
    )

    data = relative_rate_progression(event_times, title=title)

    observed = [x.tolist() for x in observed]
    baseline = [x.tolist() for x in baseline]

    data["cumdist_observed"] = observed
    data["cumdist_baseline"] = baseline
    data["cumdist_time"] = time_points.tolist()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{title}.json", "w") as f:
        json.dump(data, f)


def main(base_path, annotated_only=False):
    logger.info("Start event time correlation analysis")
    if annotated_only:
        logger.info(
            "Running in annotated-only mode: "
            "only trials with BORIS annotations are included."
        )
    logger.info("Setting up directories")
    path = Path("figures/event_meta")
    path.mkdir(parents=True, exist_ok=True)
    path = Path("figures/event_time_corr")
    path.mkdir(parents=True, exist_ok=True)

    subdir = (
        "trial_event_progressions_annotated_subset"
        if annotated_only
        else "trial_event_progressions_all_fish"
    )
    out_dir = Path("data/processed") / subdir

    # Fixed seeds so that the null and bootstrap bands are reproducible.
    np.random.seed(42)
    random.seed(42)

    logger.info("Loading data")
    trial_summary = pd.read_csv(
        os.path.join(base_path, "trial_summary.csv"), index_col=0
    )
    chirp_notes = pd.read_csv(os.path.join(base_path, "chirp_notes.csv"), index_col=0)
    trial_mask = chirp_notes["good"] == 1

    all_rise_times_lose = []
    all_rise_times_win = []
    all_chirp_times_lose = []
    all_chirp_times_win = []
    win_sex = []
    lose_sex = []

    all_contact_t = []
    all_ag_on_t = []
    all_ag_off_t = []

    # get all subdirectories in base_path
    trials_on_disk = Path(base_path).glob("*/")
    trials_on_disk = [str(x.name) for x in trials_on_disk if x.is_dir()]
    # print(trials_on_disk)

    bad_recordings = [
        "2020-05-21-10_00",
        "2020-05-25-10_00",
        "2020-05-28-10_00",
        "2020-06-10-10_00",
    ]

    logger.info("Iterating over trials and collecting event times")
    for index, trial in trial_summary.iterrows():
        if trial["recording"] in bad_recordings:
            continue
        if trial["recording"] not in trials_on_disk:
            continue

        # print(index, len(trial_summary))
        got_boris = False

        trial_path = os.path.join(base_path, trial["recording"])

        if trial["group"] < 3:
            continue
        if trial["draw"] == 1:
            continue
        if os.path.exists(os.path.join(trial_path, "led_idxs.csv")):
            got_boris = True
        if os.path.exists(os.path.join(trial_path, "LED_frames.npy")):
            got_boris = True

        # if trial_mask[index]:
        #     continue

        ids = np.load(os.path.join(trial_path, "analysis", "ids.npy"))
        times = np.load(os.path.join(trial_path, "times.npy"))
        sorter = -1 if trial["win_ID"] != ids[0] else 1

        ### event times --> BORIS behavior
        if got_boris:
            (
                contact_t_GRID,
                ag_on_off_t_GRID,
                led_idx,
                led_frames,
            ) = load_and_convert_boris_events(trial_path, trial["recording"], sr=20_000)
            all_contact_t.append(contact_t_GRID)
            all_ag_on_t.append(ag_on_off_t_GRID[:, 0])
            all_ag_off_t.append(ag_on_off_t_GRID[:, 1])
        else:
            if annotated_only:
                # Paired-subset mode: drop trials without BORIS annotations so
                # that all event types come from the same set of trials.
                continue
            all_contact_t.append(np.array([]))
            all_ag_on_t.append(np.array([]))
            all_ag_off_t.append(np.array([]))

        if not os.path.exists(os.path.join(trial_path, "chirp_times_rcnn.npy")):
            continue
        chirp_t = np.load(os.path.join(trial_path, "chirp_times_rcnn.npy"))
        chirp_ids = np.load(os.path.join(trial_path, "chirp_ids_rcnn.npy"))
        chirp_times = [
            chirp_t[chirp_ids == trial["win_ID"]],
            chirp_t[chirp_ids == trial["lose_ID"]],
        ]

        all_chirp_times_lose.append(chirp_times[1])
        all_chirp_times_win.append(chirp_times[0])

        win_sex.append(trial["sex_win"])
        lose_sex.append(trial["sex_lose"])

    # remove empty entries
    all_chirp_times_lose = [x for x in all_chirp_times_lose if len(x) > 0]
    all_chirp_times_win = [x for x in all_chirp_times_win if len(x) > 0]
    all_contact_t = [x for x in all_contact_t if len(x) > 0]
    all_ag_on_t = [x for x in all_ag_on_t if len(x) > 0]

    print(f"Lenght of loser chirp times: {len(all_chirp_times_lose)}")
    print(f"Lenght of winner chirp times: {len(all_chirp_times_win)}")
    print(f"Lenght of contact times: {len(all_contact_t)}")
    print(f"Lenght of chasing times: {len(all_ag_on_t)}")

    # relative_rate_progression(all_chirp_times_lose, title=r"chirp$_{lose}$")
    # relative_rate_progression(all_chirp_times_win, title=r"chirp$_{win}$")
    # relative_rate_progression(all_contact_t, title=r"contact")
    # relative_rate_progression(all_ag_on_t, title=r"chasing")

    stop_t_active = 3 * 60 * 60
    event_times = all_contact_t
    active_phase = (0, stop_t_active)
    num_shuffles = N_RESAMPLES
    time_points = np.linspace(0, stop_t_active, 100)

    analyze(
        all_chirp_times_lose,
        active_phase,
        num_shuffles,
        time_points,
        title=r"chirp$_{lose}$",
        out_dir=out_dir,
    )
    analyze(
        all_chirp_times_win,
        active_phase,
        num_shuffles,
        time_points,
        title=r"chirp$_{win}$",
        out_dir=out_dir,
    )
    analyze(
        all_contact_t,
        active_phase,
        num_shuffles,
        time_points,
        title=r"contact",
        out_dir=out_dir,
    )
    analyze(
        all_ag_on_t,
        active_phase,
        num_shuffles,
        time_points,
        title=r"chasing",
        out_dir=out_dir,
    )

    # Diagnostic plot: null and observed bands for the last event type above.
    baseline = compute_null_cumulative_rates(
        event_times,
        active_phase,
        num_shuffles,
        time_points,
    )
    observed = compute_observed_cis(
        event_times,
        active_phase,
        time_points,
        num_shuffles,
    )

    print(np.shape(baseline))

    baseline_qrts = np.percentile(baseline, [2.5, 50, 97.5], axis=0)
    observed_qrts = np.percentile(observed, [2.5, 50, 97.5], axis=0)

    fig, ax = plt.subplots()
    ax.plot(baseline_qrts[0])
    ax.fill_between(
        np.arange(len(baseline_qrts[0])),
        baseline_qrts[0],
        baseline_qrts[2],
        alpha=0.2,
    )
    ax.plot(observed_qrts[0])
    ax.fill_between(
        np.arange(len(observed_qrts[0])),
        observed_qrts[0],
        observed_qrts[2],
        alpha=0.2,
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyse temporal allocation of events (chirps, contacts, chasing) "
            "across trials. Default mode runs on all available trials; "
            "non-BORIS-annotated trials contribute empty contact/chasing "
            "arrays. Use --annotated-only to restrict to the subset of trials "
            "with BORIS annotations for a fully paired comparison across "
            "event types."
        )
    )
    parser.add_argument(
        "base_path",
        help="Path to the directory containing trial_summary.csv and per-trial subdirectories.",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help=(
            "Restrict analysis to BORIS-annotated trials. Writes output to "
            "data/processed/trial_event_progressions_annotated_subset/."
        ),
    )
    args = parser.parse_args()
    main(args.base_path, annotated_only=args.annotated_only)
