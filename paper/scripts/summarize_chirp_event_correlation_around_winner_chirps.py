import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.progress import track


app = typer.Typer(pretty_exceptions_show_locals=False)

BAD_RECORDINGS = [
    "2020-05-21-10_00",
    "2020-05-25-10_00",
    "2020-05-28-10_00",
    "2020-06-10-10_00",
]


def is_bad_trial(trial_row: pd.Series) -> bool:
    """
    Return True if the trial should be skipped.

    Same selection as `is_good_trial(..., skip_ag_events=True)` in
    `src_paper/analyze_event_time_rates_trial.py` (Fig. 6 A): drop draws and
    the recordings with too-close EODfs. The echo analysis only uses the
    chirp times, so trials without video-annotated behavior (groups < 5,
    missing `led_idxs.csv` / `LED_frames.npy`) are kept.

    Parameters
    ----------
    trial_row : pd.Series
        A single row from the trial summary DataFrame.

    Returns
    -------
    bool
        True if the trial should be skipped, False otherwise.
    """
    if trial_row["recording"] in BAD_RECORDINGS:
        return True
    if trial_row["draw"] == 1:
        return True
    return False


def randomize_chirp_times(
    chirp_times: np.ndarray, max_recording_time: float
) -> np.ndarray:
    """
    Randomly re-distribute chirp times within [0, max_recording_time].

    Parameters
    ----------
    chirp_times : np.ndarray
        Original array of chirp times (seconds).
    max_recording_time : float
        Maximum time for the experiment or recording.

    Returns
    -------
    np.ndarray
        Randomized chirp times, same shape as input.
    """
    return np.random.uniform(0, max_recording_time, size=len(chirp_times))


# Half-width of the uniform jitter applied to the sender's chirps in the
# surrogate data (seconds); same value as for the event-triggered rate nulls.
JITTER_S = 120.0


def jitter_chirp_times(
    chirp_times: np.ndarray, max_recording_time: float, jitter: float = JITTER_S
) -> np.ndarray:
    """
    Shift each chirp by an independent offset drawn from U(-jitter, jitter).

    Unlike a uniform redraw over the whole active phase, this keeps the
    clustering of chirps in fight episodes on time scales longer than the
    jitter and only destroys the fine-scale alignment to the other fish's
    chirps. Shifted chirps falling outside [0, max_recording_time] are
    redrawn so that every window lies inside the analysis period.
    """
    out = chirp_times + np.random.uniform(-jitter, jitter, size=len(chirp_times))
    bad = (out < 0) | (out > max_recording_time)
    while np.any(bad):
        out[bad] = chirp_times[bad] + np.random.uniform(-jitter, jitter, size=bad.sum())
        bad = (out < 0) | (out > max_recording_time)
    return out


def count_chirps_in_interval(chirp_times: np.ndarray, start: float, end: float) -> int:
    """
    Count how many chirps occur within a time interval [start, end].

    Parameters
    ----------
    chirp_times : np.ndarray
        All chirp times (seconds).
    start : float
        Start of interval (seconds).
    end : float
        End of interval (seconds).

    Returns
    -------
    int
        Number of chirps in the specified interval.
    """
    return int(np.sum((chirp_times >= start) & (chirp_times <= end)))


def extract_data_for_trial(
    trial_row: pd.Series,
    data_path: Path,
    randomize: bool,
    max_recording_time: float = 3 * 60 * 60,
) -> list[dict]:
    time_window = (10, 10)

    trial_folder = data_path / trial_row["recording"]

    chirp_times = np.load(trial_folder / "chirp_times_rcnn.npy")
    chirp_ids = np.load(trial_folder / "chirp_ids_rcnn.npy")

    loser_chirp_times = chirp_times[chirp_ids == trial_row["lose_ID"]]
    winner_chirp_times = chirp_times[chirp_ids == trial_row["win_ID"]]

    loser_chirp_times = loser_chirp_times[loser_chirp_times < max_recording_time]
    winner_chirp_times = winner_chirp_times[winner_chirp_times < max_recording_time]

    # Surrogate data: jitter the sender's (winner) chirps, keep the receiver's
    if randomize:
        winner_chirp_times = jitter_chirp_times(winner_chirp_times, max_recording_time)

    rows = []
    for i in range(len(winner_chirp_times)):
        winner_chirp = winner_chirp_times[i]

        row = {}
        row["recording"] = trial_row["recording"]
        row["winner_id"] = trial_row["win_ID"]
        row["loser_id"] = trial_row["lose_ID"]
        row["winner_sex"] = trial_row["sex_win"]
        row["loser_sex"] = trial_row["sex_lose"]
        row["winner_experience"] = trial_row["exp_win"]
        row["loser_experience"] = trial_row["exp_lose"]

        # Count loser chirps before winner chirp
        start = winner_chirp - time_window[1]
        end = winner_chirp
        row[f"loser_chirps_{time_window[1]}s_before"] = count_chirps_in_interval(
            loser_chirp_times, start, end
        )

        # Count loser chirps after winner chirp
        start = winner_chirp
        end = winner_chirp + time_window[1]
        row[f"loser_chirps_{time_window[1]}s_after"] = count_chirps_in_interval(
            loser_chirp_times, start, end
        )

        # Around winner chirp
        start = winner_chirp - time_window[1]
        end = winner_chirp + time_window[1]
        row[f"loser_chirps_{time_window[1] * 2}s_around"] = count_chirps_in_interval(
            loser_chirp_times, start, end
        )

        rows.append(row)

    return rows


def get_data(
    data_path: Path,
    randomize: bool = False,
    max_recording_time: float = 3 * 60 * 60,
) -> pd.DataFrame:
    summary_df = pd.read_csv(data_path / "trial_summary.csv", index_col=0)

    all_rows = []
    skip_count = 0
    chasing_count = 0

    for _, trial_row in summary_df.iterrows():
        if is_bad_trial(trial_row):
            skip_count += 1
            continue

        # Collect all chasing data from the current trial
        chasing_rows = extract_data_for_trial(
            trial_row=trial_row,
            data_path=data_path,
            randomize=randomize,
            max_recording_time=max_recording_time,
        )
        all_rows.extend(chasing_rows)
        chasing_count += len(chasing_rows)

    result_df = pd.DataFrame(all_rows)

    return result_df


@app.command()
def main(data_path: Path):
    """
    Main entry point. Generates random permutations of the data, then a non-random set.
    """
    filename = "loser_chirps_around_winner_chirps"
    save_dir = Path("data/processed/chirp_event_correlations/data/winner_chirps")

    if save_dir.exists() and any(save_dir.glob("*.csv")):
        print("Removing existing files.")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_permutations = 1000
    np.random.seed(42)
    for i in track(range(num_permutations), description="Permuting data"):
        perm_df = get_data(data_path, randomize=True)
        perm_df.to_csv(save_dir / f"{filename}_perm_{i}.csv", index=False)

    final_df = get_data(data_path, randomize=False)
    final_df.to_csv(save_dir / f"{filename}_obs.csv", index=False)


if __name__ == "__main__":
    app()
