# Ported from chirp_competition d040123 (src/summarize_all_timing_distributions.py).
# Changes: chirps via chirps.load_chirps (PAPER_DETECTIONS); data_path defaults to DATA_ROOT;
# JSONs under PROCESSED.
"""
This script extracts data to summarize inter-chirp intervals, chasing-durations, etc. to plot distributions of each
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from analyze_event_time_rates_trial import load_and_convert_boris_events
from chirps import load_chirps
from paper_config import DATA_ROOT, processed_dir

# plt.style.use(["src/base.mplstyle", "src/light_background.mplstyle"])
app = typer.Typer(pretty_exceptions_show_locals=False)

np.random.seed(42)

BAD_RECORDINGS = [
    "2020-05-21-10_00",
    "2020-05-25-10_00",
    "2020-05-28-10_00",
    "2020-06-10-10_00",
]


def is_bad_trial(trial_row: pd.Series) -> bool:
    """
    Check whether a trial should be skipped entirely (chirps and behavior).

    Parameters
    ----------
    trial_row : pd.Series
        A single row from the trial summary DataFrame.

    Returns
    -------
    bool
        True if the trial should be skipped, False otherwise.
    """
    if trial_row["draw"] == 1:
        return True
    if trial_row["recording"] in BAD_RECORDINGS:
        return True
    return False


def has_behavior_annotations(trial_row: pd.Series, data_path: Path) -> bool:
    """
    Check whether the BORIS behavioral annotations (chasing, contact) of a trial
    can be synchronized to the grid recording via the video LED.

    Parameters
    ----------
    trial_row : pd.Series
        A single row from the trial summary DataFrame.
    data_path : Path
        Base path where trial folders and files exist.

    Returns
    -------
    bool
        True if chasing and contact events can be used, False otherwise.
    """
    trial_folder = data_path / trial_row["recording"]

    if trial_row["group"] < 5:
        return False
    if not (trial_folder / "led_idxs.csv").exists():
        return False
    if not (trial_folder / "LED_frames.npy").exists():
        return False
    return True


def extract_data_for_trial(
    trial_row: pd.Series,
    data_path: Path,
    with_behavior: bool = True,
    max_recording_time: float = 3 * 60 * 60,
) -> dict:
    """
    Extract and compute all chasing-related data for a single trial.

    Parameters
    ----------
    trial_row : pd.Series
        One row from the trial_summary DataFrame.
    data_path : Path
        Base path where the trial data is located.
    with_behavior : bool, optional
        Whether to extract chasing and contact events. If False, only the
        inter-chirp intervals are computed and the behavioral lists are empty.
    max_recording_time : float, optional
        Hard-coded maximum trial length in seconds, by default 3 hours.

    Returns
    -------
    rows : list of dict
        Each dict is a row containing chasing event data for this trial.
    """

    trial_folder = data_path / trial_row["recording"]

    # Load events and chirps
    if with_behavior:
        contact_times, chasing_on_off_times, _, _ = load_and_convert_boris_events(
            trial_folder, trial_row["recording"], sr=20000
        )
    else:
        contact_times = np.array([])
        chasing_on_off_times = np.empty((0, 2))
    chirp_times, chirp_ids = load_chirps(trial_folder)

    loser_chirp_times = chirp_times[chirp_ids == trial_row["lose_ID"]]
    winner_chirp_times = chirp_times[chirp_ids == trial_row["win_ID"]]

    loser_chirp_times = loser_chirp_times[loser_chirp_times < max_recording_time]
    winner_chirp_times = winner_chirp_times[winner_chirp_times < max_recording_time]
    loser_chirp_times = np.sort(loser_chirp_times)
    winner_chirp_times = np.sort(winner_chirp_times)

    chasing_onsets = chasing_on_off_times[:, 0]
    chasing_offsets = chasing_on_off_times[:, 1]

    # Filter chasing events beyond the end of the active phase
    onset_mask = chasing_onsets < max_recording_time
    chasing_onsets = chasing_onsets[onset_mask]
    chasing_offsets = chasing_offsets[onset_mask]
    offset_mask = chasing_offsets < max_recording_time
    chasing_onsets = chasing_onsets[offset_mask]
    chasing_offsets = chasing_offsets[offset_mask]
    contact_times = contact_times[contact_times < max_recording_time]

    # Check if any mismatch between chasing on and offsets
    if len(chasing_onsets) != len(chasing_offsets):
        raise ValueError("Mismatch between chasing on and offsets detected.")

    # Check if any negative chasing durations
    chasing_durations = chasing_offsets - chasing_onsets
    inter_chasing_intervals = chasing_onsets[1:] - chasing_offsets[:-1]
    inter_contact_intervals = contact_times[1:] - contact_times[:-1]
    winner_inter_chirp_intervals = winner_chirp_times[1:] - winner_chirp_times[:-1]
    loser_inter_chirp_intervals = loser_chirp_times[1:] - loser_chirp_times[:-1]

    if np.any(chasing_durations < 0):
        raise ValueError("Negative chasing durations detected.")
    if np.any(inter_chasing_intervals < 0):
        # fig, ax = plt.subplots(figsize=(20, 5), constrained_layout=True)
        # for on, off in zip(chasing_onsets, chasing_offsets):
        #     ax.axvspan(on, off, color="blue", alpha=0.2)
        # for con in contact_times:
        #     ax.axvline(con, color="red", linestyle="--")
        # plt.show()

        # Remove the negative ici
        inter_chasing_intervals = inter_chasing_intervals[inter_chasing_intervals >= 0]

        print(
            "WARNING: Negative inter-chasing intervals detected and removed. But I checked this and this is fine because it is just a small chasing event completely overlapping with another, larger one, so ok to remove."
        )
    if np.any(inter_contact_intervals < 0):
        raise ValueError("Negative inter-contact intervals detected.")
    if np.any(winner_inter_chirp_intervals < 0):
        raise ValueError("Negative inter-winner chirp intervals detected.")
    if np.any(loser_inter_chirp_intervals < 0):
        raise ValueError("Negative inter-loser chirp intervals detected.")

    data = {
        "trial": trial_row["recording"],
        "has_behavior": with_behavior,
        "chasing_durations": chasing_durations.tolist(),
        "inter_chasing_intervals": inter_chasing_intervals.tolist(),
        "inter_contact_intervals": inter_contact_intervals.tolist(),
        "winner_inter_chirp_intervals": winner_inter_chirp_intervals.tolist(),
        "loser_inter_chirp_intervals": loser_inter_chirp_intervals.tolist(),
    }
    return data


def get_data(
    data_path: Path,
    randomize: bool = False,
    max_recording_time: float = 3 * 60 * 60,
) -> pd.DataFrame:
    """
    Create a DataFrame with summary statistics on chasing and chirps from all trials.

    Parameters
    ----------
    data_path : Path
        Folder where 'trial_summary.csv' and trial subfolders reside.
    randomize : bool, optional
        If True, chirp times will be randomized in each trial. Default False.
    max_recording_time : float, optional
        Hard-coded maximum trial length in seconds. Default 3 hours.

    Returns
    -------
    pd.DataFrame
        Rows represent individual chasing events. Columns contain chirp counts and intervals.
    """
    summary_df = pd.read_csv(data_path / "trial_summary.csv", index_col=0)

    all_rows = []
    skip_count = 0

    for _, trial_row in summary_df.iterrows():
        print(f"Processing trial {trial_row['recording']}")
        if is_bad_trial(trial_row):
            skip_count += 1
            continue

        # Chirps are available for all trials, chasing and contacts only for
        # trials that can be synchronized to the video
        with_behavior = has_behavior_annotations(trial_row, data_path)
        if not with_behavior:
            print("  No video sync, extracting chirps only")

        data = extract_data_for_trial(
            trial_row=trial_row,
            data_path=data_path,
            with_behavior=with_behavior,
            max_recording_time=max_recording_time,
        )
        all_rows.append(data)

    return all_rows


@app.command()
def main(data_path: Path = typer.Argument(DATA_ROOT)):
    """
    Main entry point. Generates random permutations of the data, then a non-random set.
    """
    filename = "agonistic_behavior_timing"
    save_dir = processed_dir()

    data_dict = get_data(data_path, randomize=False)

    # embed()
    # exit()

    with open(save_dir / f"{filename}.json", "w") as f:
        json.dump(data_dict, f)

    # Subset of trials with behavioral annotations, so that chirp and behavior
    # distributions come from the same trials
    annotated_subset = [d for d in data_dict if d["has_behavior"]]
    with open(save_dir / f"{filename}_annotated_subset.json", "w") as f:
        json.dump(annotated_subset, f)


if __name__ == "__main__":
    app()
