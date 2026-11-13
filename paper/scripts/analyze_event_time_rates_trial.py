"""
Analyze the chirp (or other event) rate around agonistic time points.
Assumes a single trial is a statistically independent unit.
Results are saved as json files.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as scs
import typer
from rich.progress import track
from IPython import embed

from logger import get_logger

logger = get_logger(__name__)

# For these, the EODfs are too close
bad_recordings = [
    "2020-05-21-10_00",
    "2020-05-25-10_00",
    "2020-05-28-10_00",
    "2020-06-10-10_00",
]

# --- Analysis parameters ------------------------------------------------
# Evaluation window for the event-centred rate estimate (seconds).
MAX_CONV_DT_S = 30
# Grid resolution for rate evaluation (seconds).
CONV_T_DT_S = 0.1
# Active phase of each trial (seconds from recording onset). Null
# pseudo-centres are drawn uniformly from this interval and observed
# centres are restricted to it.
ACTIVE_PHASE_S = (0, 3 * 60 * 60)
# Fraction of trials discarded in each leave-out resampling iteration
# (applied to both the observed and the null estimator).
LEAVE_OUT_FRAC = 0.1
# Number of resampling iterations for both observed and null.
N_RESAMPLES = 1000
# Jitter amplitude (seconds) for the "jittered_centers" null mode. In that
# mode each real centre time is perturbed by an independent draw from
# U(-J, J). J should be large enough to destroy alignment on the scale of
# the evaluation window (MAX_CONV_DT_S) but small enough to preserve the
# trial-scale temporal density of centres.
# CENTER_JITTER_AMPLITUDE_S = 300
CENTER_JITTER_AMPLITUDE_S = 120

SKIP_AG_EVENTS = True  # Make true to focus on full dataset for chirps only

# Preferred null mode per centring pair for the main figure. Both null
# modes are computed and stored for every pair; this mapping only records
# which one is considered primary for display. For chirp-on-chirp centring,
# the jittered-centre null isolates fine-timescale locking from
# co-clustering; for event-based centrings (agonistic/contact), the
# uniform null gives the expected rate at arbitrary times in the active
# phase. Any pair not listed here defaults to "uniform".
PREFERRED_NULL = {
    ("wchirps", "lchirps"): "jittered",
    ("lchirps", "wchirps"): "jittered",
}

# Back-compat aliases (still read by main()).
max_conv_dt = MAX_CONV_DT_S
conv_t_dt = CONV_T_DT_S
jack_pct = 1.0 - LEAVE_OUT_FRAC
maxtime = ACTIVE_PHASE_S[1]
n_iters = N_RESAMPLES

# KDE kernel for the short-timescale event-centred rate. Narrow bandwidth
# because we care about locking on the order of seconds.
gamma_kde_params = {
    "shape": 1,
    "scale": 1.2,
    "tolerance": 0.5,
}
gaussian_kde_params = {
    "sigma": 0.3,
}


@dataclass
class CompetitionTimes:
    """Data class to store all event times for a competition."""

    contacts: list
    agon: list
    agoff: list
    lchirps: list
    wchirps: list
    lrises: list
    wrises: list
    wsex: list
    lsex: list
    identifiers: list


def load_boris(trial_path, recording):
    """Load behavioral data from BORIS csv file."""
    boris_file = "-".join(recording.split("-")[:3]) + ".csv"
    data = pd.read_csv(os.path.join(trial_path, boris_file))
    times = data["Start (s)"]
    behavior = data["Behavior"]

    t_ag_on = times[behavior == 0]
    t_ag_off = times[behavior == 1]

    t_ag_on_off = []
    for t in t_ag_on:
        t1 = np.array(t_ag_off)[t_ag_off > t]
        if len(t1) >= 1:
            t_ag_on_off.append(np.array([t, t1[0]]))
    t_ag_on_off = np.array(t_ag_on_off)

    t_contact = times[behavior == 2]
    t_contact = t_contact.to_numpy()
    fps = data["FPS"][0]

    return times, behavior, t_ag_on_off, t_contact, fps


def converte_video_frames_to_grid_idx(event_frames, led_frames, led_idx):
    event_idx_grid = (event_frames - led_frames[0]) / (
        led_frames[-1] - led_frames[0]
    ) * (led_idx[-1] - led_idx[0]) + led_idx[0]
    return event_idx_grid


def load_and_convert_boris_events(trial_path, recording, sr):
    """Load and convert BORIS events to grid recording time."""
    # idx in grid-recording
    led_idx = (
        pd.read_csv(os.path.join(trial_path, "led_idxs.csv"), header=None)
        .iloc[:, 0]
        .to_numpy()
    )

    # frames where LED gets switched on
    led_frames = np.load(os.path.join(trial_path, "LED_frames.npy"))

    times, behavior, t_ag_on_off, t_contact, video_FPS = load_boris(
        trial_path, recording
    )

    contact_frame = np.array(np.round(t_contact * video_FPS), dtype=int)
    ag_on_off_frame = np.array(np.round(t_ag_on_off * video_FPS), dtype=int)

    # led_t_GRID = led_idx / sr
    contact_t_GRID = (
        converte_video_frames_to_grid_idx(contact_frame, led_frames, led_idx) / sr
    )
    ag_on_off_t_GRID = (
        converte_video_frames_to_grid_idx(ag_on_off_frame, led_frames, led_idx) / sr
    )

    return contact_t_GRID, ag_on_off_t_GRID, led_idx, led_frames


def is_good_trial(trial_path: Path, trial_row: pd.Series, skip_ag_events=False) -> bool:
    if trial_row["recording"] in bad_recordings:
        logger.info(f"Skipping {trial_row['recording']}, in bad_recordings")
        return False

    if trial_row["draw"] == 1:
        logger.info(f"Skipping {trial_row['recording']}, draw == 1")
        return False

    # """
    # if we just want chirp echo responses this is not needed, otherwise
    # we need to check if the chirp times are present

    if not skip_ag_events:
        if trial_row["group"] < 5:
            logger.info(f"Skipping {trial_row['recording']}, group < 5")
            return False

        if not Path(trial_path / "LED_frames.npy").exists():
            logger.info(
                f"Skipping {trial_row['recording']}, LED_frames.npy does not exist"
            )
            return False

        if not Path(trial_path / "LED_frames.npy").exists():
            logger.info(
                f"Skipping {trial_row['recording']}, LED_frames.npy does not exist"
            )
            return False
    # """

    return True


def load_data(trial_path: Path, trial_row: pd.Series, skip_ag_events=False):
    ids = np.load(trial_path / "analysis/ids.npy")
    times = np.load(trial_path / "times.npy")
    sorter = -1 if trial_row["win_ID"] != ids[0] else 1

    if not skip_ag_events:
        logger.info("Loading and converting BORIS events.")
        contact_t, ag_on_off_t, led_idx, led_frames = load_and_convert_boris_events(
            trial_path, trial_row["recording"], sr=20_000
        )
        ag_on_t = ag_on_off_t[:, 0]
        ag_off_t = ag_on_off_t[:, 1]
    else:
        logger.info("Skipping BORIS events.")
        contact_t = []
        ag_on_t = []
        ag_off_t = []

    logger.info("Loading and converting communication events.")

    chirp_t = np.load(trial_path / "chirp_times_rcnn.npy")
    chirp_ids = np.load(trial_path / "chirp_ids_rcnn.npy")
    loser_chirp_times = chirp_t[chirp_ids == trial_row["lose_ID"]]
    winner_chirp_times = chirp_t[chirp_ids == trial_row["win_ID"]]

    rise_idx = np.load(trial_path / "analysis/rise_idx.npy")[::sorter]
    rise_idx_int = [
        np.array(rise_idx[i][~np.isnan(rise_idx[i])], dtype=int)
        for i in range(len(rise_idx))
    ]
    rise_times = [times[rise_idx_int[0]], times[rise_idx_int[1]]]
    winner_rise_times = rise_times[0]
    loser_rise_times = rise_times[1]

    return (
        contact_t,
        ag_on_t,
        ag_off_t,
        loser_chirp_times,
        winner_chirp_times,
        loser_rise_times,
        winner_rise_times,
    )


def event_centered_times(center_event_times, sourrounding_event_times, max_dt=np.inf):
    centered_times = []
    for cevent_t in center_event_times:
        ct = np.array(sourrounding_event_times - cevent_t)
        centered_times.append(ct[np.abs(ct) < max_dt])

    return np.array(centered_times)


def filter_timestamps_outside_ranges(timestamps, starts, stops, after_event_pad=5):
    """Filter timestamps outside of the given ranges.

    This was used to discard all chirps inside chasing events for the
    estimate of the baseline rate. But the resulting rate was very
    low because most chirps are somewhat associated with chasing events
    so it is currently not in use.
    """
    stops += after_event_pad
    filtered_timestamps = []
    # Iterate over each range
    remove_time = 0
    for i, (start, stop) in enumerate(zip(starts, stops)):
        # Handle first invalid interval
        if i == 0:
            valid_timestamps = timestamps[timestamps < start]
        # Handle the middle intervals
        elif i < len(starts) - 1:
            # get the previous start and stop
            prev_start = starts[i - 1]
            prev_stop = stops[i - 1]
            prev_duration = prev_stop - prev_start
            remove_time += prev_duration

            # get the valid timestamps between the previous and
            # the current invalid interval
            valid_timestamps = timestamps[
                (timestamps > prev_stop) & (timestamps < start)
            ]

            # subtract the duration of the previous interval from the
            # valid timestamps to not alter the rate due to long
            # pauses in the invalid intervals
            valid_timestamps -= remove_time
        # Handle the last invalid interval
        else:
            # get the previous start and stop
            prev_start = starts[i - 1]
            prev_stop = stops[i - 1]
            prev_duration = prev_stop - prev_start
            remove_time += prev_duration

            # get the valid timestamps after the last invalid interval
            valid_timestamps = timestamps[timestamps > stop]

            # subtract the duration of the previous interval from the
            # valid timestamps to not alter the rate due to long
            # pauses in the invalid intervals
            valid_timestamps -= remove_time

            # now also get the timestamps after the last invalid interval
            last_interval_duration = stop - start
            remove_time += last_interval_duration
            last_timestamps = timestamps[timestamps > stop]
            last_timestamps -= remove_time

            # concatenate the valid timestamps and the timestamps after
            # the last invalid interval
            valid_timestamps = np.concatenate([valid_timestamps, last_timestamps])

        filtered_timestamps.append(valid_timestamps)
    filtered_timestamps = np.concatenate(filtered_timestamps)

    return filtered_timestamps, remove_time


def gamma_kde(x, timestamps, shape=1.2, scale=1, tolerance=0.2):
    # Initialize the KDE with zeros, same shape as x
    kde = np.zeros_like(x, dtype=float)
    dur = np.max(x) - np.min(x)
    pad = dur * tolerance
    minx = np.min(x) - pad
    maxx = np.max(x) + pad

    relevant_timestamps = timestamps[(timestamps >= minx) & (timestamps <= maxx)]

    # Normalize the Gamma distribution to have an integral of 1
    def gamma_kernel(t, shape, scale):
        result = np.zeros_like(t, dtype=float)
        positive_indices = t > 0
        norm_factor = scs.gamma(shape) * scale**shape
        result[positive_indices] = (
            t[positive_indices] ** (shape - 1) * np.exp(-t[positive_indices] / scale)
        ) / norm_factor
        return result

    # Evaluate the gamma KDE at each point in x for each timestamp
    for timestamp in relevant_timestamps:
        kde += gamma_kernel(x - timestamp, shape, scale)

    return kde


def gaussian_kde(x, timestamps, sigma=0.3):
    kde = np.zeros_like(x, dtype=float)
    dur = np.max(x) - np.min(x)
    pad = dur * 0.2
    minx = np.min(x) - pad
    maxx = np.max(x) + pad

    relevant_timestamps = timestamps[(timestamps >= minx) & (timestamps <= maxx)]

    def gaussian_kernel(t, sigma):
        result = np.zeros_like(t, dtype=float)
        result = np.exp(-0.5 * (t / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return result

    for timestamp in relevant_timestamps:
        kde += gaussian_kernel(x - timestamp, sigma)

    return kde


def _leave_out_indices(n_trials, leave_out_frac):
    """Pick which trials to discard in a single leave-out resample.

    Uses floor(leave_out_frac * n_trials) with a lower bound of 1, matching
    the original implementation's discard count. Unlike the original, draws
    without replacement so exactly `n_discard` distinct trials are dropped.
    """
    n_discard = max(1, int(n_trials * leave_out_frac))
    discard = np.random.choice(n_trials, n_discard, replace=False)
    return np.setdiff1d(np.arange(n_trials), discard)


def observed_rate_bootstrap(
    center_event_times,
    surrounding_event_times,
    conv_time,
    n_bootstraps=N_RESAMPLES,
    leave_out_frac=LEAVE_OUT_FRAC,
):
    """Observed event-centred rate with leave-out resampling.

    In each iteration, discard a fraction of trials, pool lags from the
    remaining trials, evaluate a Gaussian KDE on ``conv_time``, and
    normalise by the total number of centring events across kept trials.
    Also returns per-trial rate curves (no resampling) for the heatmap
    panel.

    Returns
    -------
    bootstrapped_rates : np.ndarray, shape (n_bootstraps, len(conv_time))
        Rate curves in events per second per centring event.
    per_trial_rates : list of np.ndarray
        Per-trial rate curves, same units.
    per_trial_lags : list of np.ndarray
        Pooled surrounding-event lags for each trial, used as input to the
        eventplot display.
    """
    n_trials = len(center_event_times)
    n_centerings = np.array([len(c) for c in center_event_times], dtype=int)

    # Pre-compute per-trial lag pools (input to both the heatmap and every
    # leave-out iteration).
    per_trial_lags = []
    for i in range(n_trials):
        if n_centerings[i] == 0:
            per_trial_lags.append(np.array([]))
            continue
        lags = np.concatenate(
            [np.asarray(surrounding_event_times[i]) - c for c in center_event_times[i]]
        )
        per_trial_lags.append(lags)

    bootstrapped_rates = np.zeros((n_bootstraps, len(conv_time)))
    for b in range(n_bootstraps):
        logger.info(f"Observed bootstrap iteration {b + 1}/{n_bootstraps}")
        keep = _leave_out_indices(n_trials, leave_out_frac)

        pooled = [per_trial_lags[i] for i in keep if len(per_trial_lags[i]) > 0]
        kept_n = int(np.sum(n_centerings[keep]))
        if not pooled or kept_n == 0:
            continue
        pooled = np.concatenate(pooled)

        kde = gaussian_kde(conv_time, pooled, **gaussian_kde_params)
        kde /= kept_n
        bootstrapped_rates[b] = kde

    # Per-trial rates (no resampling, for heatmap).
    per_trial_rates = []
    for i in range(n_trials):
        if n_centerings[i] == 0:
            per_trial_rates.append(np.zeros_like(conv_time))
            continue
        kde = gaussian_kde(conv_time, per_trial_lags[i], **gaussian_kde_params)
        kde /= n_centerings[i]
        per_trial_rates.append(kde)

    return bootstrapped_rates, per_trial_rates, per_trial_lags


def null_rate_bootstrap(
    center_event_times,
    surrounding_event_times,
    active_phase,
    conv_time,
    n_bootstraps=N_RESAMPLES,
    leave_out_frac=LEAVE_OUT_FRAC,
    null_mode="uniform",
    jitter_amplitude_s=CENTER_JITTER_AMPLITUDE_S,
):
    """Monte Carlo null event-centred rate.

    In each iteration, discard a fraction of trials (matched to the
    observed estimator), then construct pseudo-centres according to
    ``null_mode``, compute surrounding-event lags relative to those
    pseudo-centres, pool across kept trials, evaluate a Gaussian KDE on
    ``conv_time``, and normalise by the total number of pseudo-centres.

    Parameters
    ----------
    null_mode : {"uniform", "jittered_centers"}
        ``"uniform"`` draws pseudo-centres uniformly in ``active_phase``
        with the same count as the observed centres. Appropriate when the
        centring stream is rare relative to the surrounding stream or when
        a flat baseline is the null of interest.
        ``"jittered_centers"`` takes each real centre and perturbs it by
        an independent draw from U(-J, J) with J = ``jitter_amplitude_s``.
        This preserves the temporal clustering of centres while destroying
        alignment on the ``jitter_amplitude_s`` timescale, isolating
        fine-timescale locking from co-clustering. Appropriate when both
        streams are themselves clustered in time (e.g. chirp-on-chirp).
    jitter_amplitude_s : float
        Jitter amplitude J for ``null_mode="jittered_centers"``. Ignored
        otherwise.

    Returns
    -------
    np.ndarray, shape (n_bootstraps, len(conv_time))
        Null rate curves in events per second per centring event.
    """
    if null_mode not in ("uniform", "jittered_centers"):
        raise ValueError(
            f"null_mode must be 'uniform' or 'jittered_centers', got {null_mode!r}"
        )

    n_trials = len(center_event_times)
    n_centerings = np.array([len(c) for c in center_event_times], dtype=int)
    start, end = active_phase

    bootstrapped_rates = np.zeros((n_bootstraps, len(conv_time)))
    for b in range(n_bootstraps):
        logger.info(f"Null bootstrap ({null_mode}) iteration {b + 1}/{n_bootstraps}")
        keep = _leave_out_indices(n_trials, leave_out_frac)

        pooled_lags = []
        for i in keep:
            if n_centerings[i] == 0:
                continue
            if null_mode == "uniform":
                pseudo_centers = np.random.uniform(start, end, n_centerings[i])
            else:  # "jittered_centers"
                pseudo_centers = np.asarray(center_event_times[i]) + np.random.uniform(
                    -jitter_amplitude_s,
                    jitter_amplitude_s,
                    n_centerings[i],
                )
            surrounds_i = np.asarray(surrounding_event_times[i])
            for pc in pseudo_centers:
                pooled_lags.append(surrounds_i - pc)

        kept_n = int(np.sum(n_centerings[keep]))
        if not pooled_lags or kept_n == 0:
            continue
        pooled_lags = np.concatenate(pooled_lags)

        kde = gaussian_kde(conv_time, pooled_lags, **gaussian_kde_params)
        kde /= kept_n
        bootstrapped_rates[b] = kde

    return bootstrapped_rates


# ---------------------------------------------------------------------------
# Legacy function names kept for any downstream code that still imports them.
# Implemented as thin wrappers around the unified resamplers above.
# ---------------------------------------------------------------------------
def baselinerate_bootstrap_(*args, **kwargs):
    raise NotImplementedError(
        "baselinerate_bootstrap_ has been replaced by null_rate_bootstrap; "
        "see module docstring."
    )


def eventrate_jackknife_trial_per_population(*args, **kwargs):
    raise NotImplementedError(
        "eventrate_jackknife_trial_per_population has been replaced by "
        "observed_rate_bootstrap; see module docstring."
    )


def main(path: Path):
    trial_summary = pd.read_csv(path / "trial_summary.csv")

    data = CompetitionTimes(
        contacts=[],
        agon=[],
        agoff=[],
        lchirps=[],
        wchirps=[],
        lrises=[],
        wrises=[],
        wsex=[],
        lsex=[],
        identifiers=[],
    )

    for i in range(len(trial_summary)):
        trial_row = trial_summary.iloc[i]
        trial_path = path / trial_row["recording"]

        if not is_good_trial(trial_path, trial_row, skip_ag_events=SKIP_AG_EVENTS):
            continue

        logger.info(f"Loading data for {trial_row['recording']}")
        (
            contact_t,
            ag_on_t,
            ag_off_t,
            loser_chirp_times,
            winner_chirp_times,
            loser_rise_times,
            winner_rise_times,
        ) = load_data(
            trial_path, trial_row, skip_ag_events=SKIP_AG_EVENTS
        )  # skip_ag_events=True if just chirp echo responses are needed

        data.contacts.append(contact_t)
        data.agon.append(ag_on_t)
        data.agoff.append(ag_off_t)
        data.lchirps.append(loser_chirp_times)
        data.wchirps.append(winner_chirp_times)
        data.lrises.append(loser_rise_times)
        data.wrises.append(winner_rise_times)
        data.wsex.append(trial_row["sex_win"])
        data.lsex.append(trial_row["sex_lose"])
        data.identifiers.append(trial_row["recording"])

    logger.info("Finished collecting data, starting analysis ------------------------")

    centerings = [
        # ("agoff", "lchirps"),
        # ("agoff", "wchirps"),
        # ("agon", "lchirps"),
        # ("agon", "wchirps"),
        # ("contacts", "lchirps"),
        # ("contacts", "wchirps"),
        ("wchirps", "lchirps"),
        ("lchirps", "wchirps"),
        # ("agoff", "lrises"),
        # ("agoff", "wrises"),
        # ("agon", "lrises"),
        # ("agon", "wrises"),
        # ("contacts", "lrises"),
        # ("contacts", "wrises"),
        # ("lrises", "lchirps"),
        # ("wrises", "wchirps"),
        # ("wrises", "lrises"),
        # ("lrises", "wrises"),
    ]

    centered_data = {}

    for c1, c2 in centerings:
        logger.info(f"Centering {c2} around {c1}")
        n_trials = len(getattr(data, c1))

        center_event_times = getattr(data, c1)
        sourrounding_event_times = getattr(data, c2)
        trial_identifiers = list(data.identifiers)

        # Restrict centres to the active phase so observed and null share the
        # same centring domain; surroundings are left unfiltered because the
        # narrow KDE (sigma=0.3 s) only picks up lags within a few seconds of
        # conv_time anyway.
        start, end = ACTIVE_PHASE_S
        center_event_times = [
            np.asarray(c)[(np.asarray(c) >= start) & (np.asarray(c) < end)]
            for c in center_event_times
        ]

        # Drop trials with no centres (nothing to centre around).
        keep_indices = [i for i in range(n_trials) if len(center_event_times[i]) > 0]
        if len(keep_indices) < n_trials:
            logger.info(
                f"Dropping {n_trials - len(keep_indices)} trials without "
                f"{c1} centres in the active phase."
            )
        center_event_times = [center_event_times[i] for i in keep_indices]
        sourrounding_event_times = [sourrounding_event_times[i] for i in keep_indices]
        trial_identifiers = [trial_identifiers[i] for i in keep_indices]
        n_trials = len(keep_indices)

        centered_data[(c1, c2)] = {
            "centers": center_event_times,
            "surrounds": sourrounding_event_times,
            "identifiers": trial_identifiers,
        }

        logger.info(
            f"Finished collecting {c2} around {c1}: "
            f"{n_trials} trials, "
            f"{sum(len(c) for c in center_event_times)} total centres."
        )

    logger.info("Starting bootstrapping ------------------------")

    conv_time = np.arange(-MAX_CONV_DT_S, MAX_CONV_DT_S, CONV_T_DT_S)

    for (c1, c2), packet in centered_data.items():
        logger.info(f"Resampling: {c2} around {c1}")

        centers = packet["centers"]
        surrounds = packet["surrounds"]
        identifiers = packet["identifiers"]

        # Observed rate (leave-out resampling) + per-trial lag pools and rates.
        (
            jackknifed_eventrate,
            per_trial_rates,
            per_trial_lags,
        ) = observed_rate_bootstrap(
            centers,
            surrounds,
            conv_time,
            n_bootstraps=N_RESAMPLES,
            leave_out_frac=LEAVE_OUT_FRAC,
        )

        # Null rates: both a uniform pseudo-centre null and a
        # jittered-real-centre null are computed for every pair so the
        # supplementary figure can compare them; the main figure uses the
        # one flagged in PREFERRED_NULL.
        # bootstrapped_baseline_uniform = null_rate_bootstrap(
        #     centers,
        #     surrounds,
        #     active_phase=ACTIVE_PHASE_S,
        #     conv_time=conv_time,
        #     n_bootstraps=N_RESAMPLES,
        #     leave_out_frac=LEAVE_OUT_FRAC,
        #     null_mode="uniform",
        # )
        bootstrapped_baseline_jittered = null_rate_bootstrap(
            centers,
            surrounds,
            active_phase=ACTIVE_PHASE_S,
            conv_time=conv_time,
            n_bootstraps=N_RESAMPLES,
            leave_out_frac=LEAVE_OUT_FRAC,
            null_mode="jittered_centers",
            jitter_amplitude_s=CENTER_JITTER_AMPLITUDE_S,
        )

        preferred_null = PREFERRED_NULL.get((c1, c2), "uniform")

        data_json = {
            "description": f"{c2} centered around {c1}",
            "centered_times": [x.tolist() for x in per_trial_lags],
            "conv_time": conv_time.tolist(),
            "jackknifed_eventrate": jackknifed_eventrate.tolist(),
            # "bootstrapped_baseline_uniform": bootstrapped_baseline_uniform.tolist(),
            "bootstrapped_baseline_jittered": bootstrapped_baseline_jittered.tolist(),
            "preferred_null": preferred_null,
            "jitter_amplitude_s": CENTER_JITTER_AMPLITUDE_S,
            "per_trial_rates": [x.tolist() for x in per_trial_rates],
            "trial_identifiers": identifiers,
        }

        datapath = Path("data/processed/event_time_rates")
        datapath.mkdir(parents=True, exist_ok=True)
        with open(datapath / f"{c2}@{c1}.json", "w") as f:
            json.dump(data_json, f)

        jk_quarts = np.percentile(jackknifed_eventrate, [2.5, 50, 97.5], axis=0)
        # bt_uniform_quarts = np.percentile(
        #     bootstrapped_baseline_uniform, [2.5, 50, 97.5], axis=0
        # )
        bt_jittered_quarts = np.percentile(
            bootstrapped_baseline_jittered, [2.5, 50, 97.5], axis=0
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
        ax.plot(conv_time, jk_quarts[1], color="tab:blue", label="observed")
        ax.fill_between(
            conv_time,
            jk_quarts[0],
            jk_quarts[2],
            color="tab:blue",
            alpha=0.3,
            lw=0,
        )
        # ax.plot(
        #     conv_time, bt_uniform_quarts[1], color="tab:orange", label="null: uniform"
        # )
        # ax.fill_between(
        #     conv_time,
        #     bt_uniform_quarts[0],
        #     bt_uniform_quarts[2],
        #     color="tab:orange",
        #     alpha=0.3,
        #     lw=0,
        # )
        ax.plot(
            conv_time,
            bt_jittered_quarts[1],
            color="tab:green",
            label=f"null: jittered (J={CENTER_JITTER_AMPLITUDE_S}s)",
        )
        ax.fill_between(
            conv_time,
            bt_jittered_quarts[0],
            bt_jittered_quarts[2],
            color="tab:green",
            alpha=0.3,
            lw=0,
        )
        ax.legend(frameon=False, fontsize="small")
        ax.set_title(f"{c2} centered around {c1} (preferred null: {preferred_null})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Event rate (Hz)")

        plt.savefig(datapath / f"{c2}@{c1}.png")
        plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
