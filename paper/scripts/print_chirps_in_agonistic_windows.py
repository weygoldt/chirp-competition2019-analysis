# Ported from chirp_competition d040123 (src_paper/print_chirps_in_agonistic_windows.py).
# Changes: chirps via chirps.load_chirps (PAPER_DETECTIONS); data root default DATA_ROOT;
# per-trial table under $PAPER_OUT/data/processed (named after --max-shift, so the two
# runs of run_all.sh keep their own); the printed numbers also go to write_stats.
"""
This script counts the winner and loser chirps that are associated with agonistic
behavior and those that are unrelated to it. A chirp counts as agonistic if it
falls into either of these windows, everything else is unrelated:
  * Chasing window : [onset - chasing_pre, offset + chasing_post]
  * Contact window : [contact - contact_window, contact + contact_window]
Overlapping windows are merged, window sizes are set via command line options.

Because long windows would capture many chirps by chance alone, the fraction of
the recording covered by the windows is saved as well. This is the fraction of
chirps expected in the windows if chirps were emitted uniformly in time.

Whether chirps fall into the windows more or less often than expected from the
time they cover is tested with a permutation test. In each permutation, the chirp
train of each fish is circularly shifted by a random offset relative to the
behavior of its trial. This keeps the temporal structure of both the chirps
(e.g. bursts) and the agonistic windows, and only breaks their alignment. The
test statistic is the median across trials of the fraction of chirps in the
windows minus the fraction of time the windows cover.

Trial filtering (bad recordings, draws, group < 5, missing LED files) is reused
from `summarize_submissive_chirp_effect_on_dominant.is_bad_trial`.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from analyze_event_time_rates_trial import load_and_convert_boris_events
from chirps import load_chirps
from manuscript_stats import write_stats
from paper_config import DATA_ROOT, processed
from summarize_submissive_chirp_effect_on_dominant import is_bad_trial

app = typer.Typer(pretty_exceptions_show_locals=False)


def merge_intervals(intervals: np.ndarray) -> np.ndarray:
    """
    Merge overlapping intervals.

    Parameters
    ----------
    intervals : np.ndarray
        Array of shape (n, 2) with start and stop times.

    Returns
    -------
    np.ndarray
        Sorted, non-overlapping intervals of shape (m, 2).
    """
    if len(intervals) == 0:
        return np.empty((0, 2))

    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = [intervals[0].copy()]
    for start, stop in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append(np.array([start, stop]))
    return np.array(merged)


def in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """
    Check which times fall into a set of sorted, non-overlapping intervals.

    Parameters
    ----------
    times : np.ndarray
        Event times.
    intervals : np.ndarray
        Merged intervals of shape (m, 2), as returned by `merge_intervals`.

    Returns
    -------
    np.ndarray
        Boolean mask, True where a time lies within an interval.
    """
    if len(intervals) == 0:
        return np.zeros(len(times), dtype=bool)

    # Index of the last interval starting before or at each time
    idx = np.searchsorted(intervals[:, 0], times, side="right") - 1
    valid = idx >= 0
    mask = np.zeros(len(times), dtype=bool)
    mask[valid] = times[valid] <= intervals[idx[valid], 1]
    return mask


def extract_data_for_trial(
    trial_row: pd.Series,
    data_path: Path,
    chasing_pre: float,
    chasing_post: float,
    contact_window: float,
    max_recording_time: float,
) -> tuple[dict, dict]:
    """
    Count the winner and loser chirps inside agonistic windows of a single trial.

    Parameters
    ----------
    trial_row : pd.Series
        One row from the trial_summary DataFrame.
    data_path : Path
        Base path where the trial data is located.
    chasing_pre : float
        Seconds before chasing onset included in the window.
    chasing_post : float
        Seconds after chasing offset included in the window.
    contact_window : float
        Seconds before and after a contact included in the window.
    max_recording_time : float
        Maximum trial length in seconds.

    Returns
    -------
    dict
        Chirp counts, fractions and window coverage for this trial.
    dict
        Merged agonistic windows and chirp times of winner and loser, used for
        the permutation test.
    """
    trial_folder = data_path / trial_row["recording"]

    # Load events and chirps
    contact_times, chasing_on_off_times, _, _ = load_and_convert_boris_events(
        trial_folder, trial_row["recording"], sr=20000
    )
    chirp_times, chirp_ids = load_chirps(trial_folder)

    chirp_mask = chirp_times < max_recording_time
    chirp_times = chirp_times[chirp_mask]
    chirp_ids = chirp_ids[chirp_mask]
    winner_chirp_times = chirp_times[chirp_ids == trial_row["win_ID"]]
    loser_chirp_times = chirp_times[chirp_ids == trial_row["lose_ID"]]

    # Only keep events starting within the recording window, windows extending
    # beyond it are clipped below
    chasing_on_off_times = chasing_on_off_times[
        chasing_on_off_times[:, 0] < max_recording_time
    ]
    contact_times = contact_times[contact_times < max_recording_time]

    if np.any(chasing_on_off_times[:, 1] < chasing_on_off_times[:, 0]):
        raise ValueError("Negative chasing durations detected.")

    chasing_windows = np.column_stack(
        [
            chasing_on_off_times[:, 0] - chasing_pre,
            chasing_on_off_times[:, 1] + chasing_post,
        ]
    )
    contact_windows = np.column_stack(
        [contact_times - contact_window, contact_times + contact_window]
    )

    chasing_windows = merge_intervals(
        np.clip(chasing_windows, 0, max_recording_time)
    )
    contact_windows = merge_intervals(
        np.clip(contact_windows, 0, max_recording_time)
    )
    agonistic_windows = merge_intervals(
        np.concatenate([chasing_windows, contact_windows])
    )

    data = {
        "trial": trial_row["recording"],
        "n_chasing": len(chasing_on_off_times),
        "n_contact": len(contact_times),
        "coverage_chasing": np.sum(np.diff(chasing_windows, axis=1))
        / max_recording_time,
        "coverage_contact": np.sum(np.diff(contact_windows, axis=1))
        / max_recording_time,
        "coverage_agonistic": np.sum(np.diff(agonistic_windows, axis=1))
        / max_recording_time,
    }

    for fish, times in [("winner", winner_chirp_times), ("loser", loser_chirp_times)]:
        n_chasing = np.sum(in_intervals(times, chasing_windows))
        n_contact = np.sum(in_intervals(times, contact_windows))
        n_agonistic = np.sum(in_intervals(times, agonistic_windows))
        n_total = len(times)
        data[f"{fish}_n_chirps"] = n_total
        data[f"{fish}_n_chasing"] = n_chasing
        data[f"{fish}_n_contact"] = n_contact
        data[f"{fish}_n_agonistic"] = n_agonistic
        data[f"{fish}_n_unrelated"] = n_total - n_agonistic
        data[f"{fish}_frac_agonistic"] = (
            n_agonistic / n_total if n_total > 0 else np.nan
        )
        data[f"{fish}_frac_unrelated"] = (
            (n_total - n_agonistic) / n_total if n_total > 0 else np.nan
        )

    events = {
        "agonistic_windows": agonistic_windows,
        "winner": winner_chirp_times,
        "loser": loser_chirp_times,
    }
    return data, events


def get_data(
    data_path: Path,
    chasing_pre: float,
    chasing_post: float,
    contact_window: float,
    max_recording_time: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Collect chirp counts in agonistic windows for all annotated trials.

    Parameters
    ----------
    data_path : Path
        Folder where 'trial_summary.csv' and trial subfolders reside.
    chasing_pre : float
        Seconds before chasing onset included in the window.
    chasing_post : float
        Seconds after chasing offset included in the window.
    contact_window : float
        Seconds before and after a contact included in the window.
    max_recording_time : float
        Maximum trial length in seconds.

    Returns
    -------
    pd.DataFrame
        One row per trial.
    list of dict
        Agonistic windows and chirp times per trial, in the same order.
    """
    summary_df = pd.read_csv(data_path / "trial_summary.csv", index_col=0)

    all_rows = []
    all_events = []
    skip_count = 0
    for _, trial_row in summary_df.iterrows():
        if is_bad_trial(trial_row, data_path):
            skip_count += 1
            continue

        data, events = extract_data_for_trial(
            trial_row=trial_row,
            data_path=data_path,
            chasing_pre=chasing_pre,
            chasing_post=chasing_post,
            contact_window=contact_window,
            max_recording_time=max_recording_time,
        )
        all_rows.append(data)
        all_events.append(events)

    print(f"Included {len(all_rows)} trials, skipped {skip_count}.")
    return pd.DataFrame(all_rows), all_events


def shifted_fractions(
    times: np.ndarray,
    intervals: np.ndarray,
    n_permutations: int,
    max_recording_time: float,
    max_shift: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fraction of circularly shifted chirps inside the agonistic windows.

    Parameters
    ----------
    times : np.ndarray
        Chirp times of one fish.
    intervals : np.ndarray
        Merged agonistic windows of the trial.
    n_permutations : int
        Number of random shifts.
    max_recording_time : float
        Trial length in seconds, the period of the circular shift.
    max_shift : float or None
        If given, shifts are drawn from [-max_shift, max_shift] seconds instead
        of the full trial length, which preserves slow trends across the trial.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Fraction of chirps in the windows for each permutation.
    """
    if max_shift is None:
        shifts = rng.uniform(0, max_recording_time, size=n_permutations)
    else:
        shifts = rng.uniform(-max_shift, max_shift, size=n_permutations)

    shifted = (times[None, :] + shifts[:, None]) % max_recording_time
    mask = in_intervals(shifted.ravel(), intervals).reshape(shifted.shape)
    return mask.mean(axis=1)


def two_sided_p(observed: float, null: np.ndarray) -> float:
    """Two-sided permutation p-value, including the observed value."""
    p_greater = (np.sum(null >= observed) + 1) / (len(null) + 1)
    p_less = (np.sum(null <= observed) + 1) / (len(null) + 1)
    return min(1.0, 2 * min(p_greater, p_less))


def permutation_test(
    df: pd.DataFrame,
    all_events: list[dict],
    n_permutations: int,
    max_recording_time: float,
    max_shift: float | None,
    seed: int,
) -> dict:
    """
    Test whether chirps fall into agonistic windows more or less often than
    expected from the fraction of time the windows cover.

    Adds the mean shifted fraction and a per-trial p-value to `df` for each fish.

    Parameters
    ----------
    df : pd.DataFrame
        Per-trial table as returned by `get_data`.
    all_events : list of dict
        Agonistic windows and chirp times per trial, as returned by `get_data`.
    n_permutations : int
        Number of permutations.
    max_recording_time : float
        Trial length in seconds.
    max_shift : float or None
        Maximum shift in seconds, None for shifts across the full trial.
    seed : int
        Seed of the random number generator.

    Returns
    -------
    dict
        Observed statistics, null distributions and p-values per fish.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for fish in ["winner", "loser"]:
        null_fracs = np.full((len(df), n_permutations), np.nan)
        null_counts = np.zeros((len(df), n_permutations))
        for i, events in enumerate(all_events):
            if len(events[fish]) == 0:
                continue
            null_fracs[i] = shifted_fractions(
                events[fish],
                events["agonistic_windows"],
                n_permutations,
                max_recording_time,
                max_shift,
                rng,
            )
            null_counts[i] = null_fracs[i] * len(events[fish])

        has_chirps = df[f"{fish}_n_chirps"].to_numpy() > 0
        observed_frac = df[f"{fish}_frac_agonistic"].to_numpy()
        coverage = df["coverage_agonistic"].to_numpy()

        df[f"{fish}_null_frac_agonistic"] = [
            null_fracs[i].mean() if has_chirps[i] else np.nan for i in range(len(df))
        ]
        df[f"{fish}_p"] = [
            two_sided_p(observed_frac[i], null_fracs[i]) if has_chirps[i] else np.nan
            for i in range(len(df))
        ]

        # Median excess across trials with chirps
        observed_excess = np.median(observed_frac[has_chirps] - coverage[has_chirps])
        null_excess = np.median(
            null_fracs[has_chirps] - coverage[has_chirps, None], axis=0
        )

        # Pooled fraction, dominated by trials with many chirps
        n_total = df[f"{fish}_n_chirps"].sum()
        observed_pooled = df[f"{fish}_n_agonistic"].sum() / n_total
        null_pooled = null_counts.sum(axis=0) / n_total

        results[fish] = {
            "n_trials": int(has_chirps.sum()),
            "n_significant": int((df[f"{fish}_p"] < 0.05).sum()),
            "observed_excess": observed_excess,
            "null_excess": null_excess,
            "p_excess": two_sided_p(observed_excess, null_excess),
            "observed_pooled": observed_pooled,
            "null_pooled": null_pooled,
            "p_pooled": two_sided_p(observed_pooled, null_pooled),
        }

    return results


def median_iqr(values: pd.Series) -> str:
    """Format the median and interquartile range of fractions across trials."""
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{median:.1%} [IQR {q1:.1%}-{q3:.1%}]"


def print_trial_table(df: pd.DataFrame) -> None:
    """Print chirp counts and fractions for each trial."""
    columns = ["trial", "n_chasing", "n_contact", "coverage_agonistic"]
    for fish in ["winner", "loser"]:
        columns += [
            f"{fish}_n_chirps",
            f"{fish}_n_agonistic",
            f"{fish}_n_unrelated",
            f"{fish}_frac_agonistic",
            f"{fish}_null_frac_agonistic",
            f"{fish}_p",
        ]
    with pd.option_context(
        "display.max_rows", None, "display.width", 250, "display.precision", 3
    ):
        print(df[columns].to_string(index=False))


def print_summary(df: pd.DataFrame) -> None:
    """Print pooled and per-trial fractions of chirps in agonistic windows."""
    print(f"\n{len(df)} trials with behavioral annotations")
    print(
        f"Time covered by agonistic windows: "
        f"{median_iqr(df['coverage_agonistic'])} "
        f"(chasing: {median_iqr(df['coverage_chasing'])}, "
        f"contact: {median_iqr(df['coverage_contact'])})"
    )
    for fish in ["winner", "loser"]:
        n_total = df[f"{fish}_n_chirps"].sum()
        pooled = {
            key: df[f"{fish}_n_{key}"].sum() / n_total
            for key in ["agonistic", "unrelated", "chasing", "contact"]
        }
        print(
            f"\n{fish.capitalize()} chirps: {n_total} "
            f"({df[f'{fish}_frac_agonistic'].notna().sum()} trials with chirps)"
        )
        for key in ["agonistic", "unrelated"]:
            frac = df[f"{fish}_frac_{key}"]
            print(
                f"  {key:<9}: {df[f'{fish}_n_{key}'].sum()}, "
                f"pooled {pooled[key]:.1%}, "
                f"per trial {median_iqr(frac)}, "
                f"range {frac.min():.1%}-{frac.max():.1%}"
            )
        print(
            f"  pooled agonistic by window type: chasing {pooled['chasing']:.1%}, "
            f"contact {pooled['contact']:.1%}"
        )


def print_permutation_test(
    results: dict, n_permutations: int, max_shift: float | None
) -> None:
    """Print the results of the permutation test."""
    shift_range = "full trial" if max_shift is None else f"+-{max_shift:g} s"
    print(
        f"\nPermutation test: {n_permutations} circular shifts of each chirp train "
        f"({shift_range})"
    )
    for fish, res in results.items():
        null_q = np.quantile(res["null_excess"], [0.025, 0.975])
        pooled_q = np.quantile(res["null_pooled"], [0.025, 0.975])
        print(f"\n{fish.capitalize()} chirps ({res['n_trials']} trials with chirps)")
        print(
            f"  median excess over coverage: {res['observed_excess']:+.1%}, "
            f"null 95% [{null_q[0]:+.1%}, {null_q[1]:+.1%}], "
            f"p = {res['p_excess']:.4f}"
        )
        print(
            f"  pooled fraction agonistic  : {res['observed_pooled']:.1%}, "
            f"null 95% [{pooled_q[0]:.1%}, {pooled_q[1]:.1%}], "
            f"p = {res['p_pooled']:.4f}"
        )
        print(
            f"  trials with p < 0.05       : {res['n_significant']} of "
            f"{res['n_trials']}"
        )


def median_iqr_values(values: pd.Series) -> dict:
    """Median and interquartile range of a per-trial fraction, as `median_iqr` prints them."""
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return {"median": median, "iqr": [q1, q3]}


def stats_values(df: pd.DataFrame, results: dict, n_permutations: int) -> dict:
    """The numbers of `print_summary` and `print_permutation_test`, for write_stats."""
    values = {
        "n_trials": len(df),
        "n_permutations": n_permutations,
        "coverage": {
            key: median_iqr_values(df[f"coverage_{key}"])
            for key in ["agonistic", "chasing", "contact"]
        },
    }
    for fish, res in results.items():
        n_total = df[f"{fish}_n_chirps"].sum()
        per_trial = {}
        for key in ["agonistic", "unrelated"]:
            frac = df[f"{fish}_frac_{key}"]
            per_trial[key] = {**median_iqr_values(frac), "range": [frac.min(), frac.max()]}
        values[fish] = {
            "n_chirps": n_total,
            "n_trials_with_chirps": res["n_trials"],
            "pooled": {
                key: df[f"{fish}_n_{key}"].sum() / n_total
                for key in ["agonistic", "unrelated", "chasing", "contact"]
            },
            "per_trial": per_trial,
            "excess": {
                "observed": res["observed_excess"],
                "null_95": np.quantile(res["null_excess"], [0.025, 0.975]),
                "p": res["p_excess"],
            },
            "pooled_test": {
                "observed": res["observed_pooled"],
                "null_95": np.quantile(res["null_pooled"], [0.025, 0.975]),
                "p": res["p_pooled"],
            },
            "n_trials_p_below_0_05": res["n_significant"],
        }
    return values


@app.command()
def main(
    data_path: Path = DATA_ROOT,
    chasing_pre: float = 10.0,
    chasing_post: float = 10.0,
    contact_window: float = 10.0,
    max_recording_time: float = 3 * 60 * 60,
    output_csv: Optional[Path] = None,  # noqa: UP045, typer needs Optional
    n_permutations: int = 10000,
    max_shift: Optional[float] = None,  # noqa: UP045, typer needs Optional
    seed: int = 42,
):
    """
    Main entry point. Saves one row per trial and prints a summary.
    """
    # The permutation columns depend on the shift range, so each range has its
    # own table and stats namespace.
    suffix = "" if max_shift is None else f"_maxshift_{max_shift:g}s"
    if output_csv is None:
        output_csv = processed(f"chirps_in_agonistic_windows{suffix}.csv")
    df, all_events = get_data(
        data_path,
        chasing_pre=chasing_pre,
        chasing_post=chasing_post,
        contact_window=contact_window,
        max_recording_time=max_recording_time,
    )
    results = permutation_test(
        df,
        all_events,
        n_permutations=n_permutations,
        max_recording_time=max_recording_time,
        max_shift=max_shift,
        seed=seed,
    )
    df.to_csv(output_csv, index=False)
    print(f"Per-trial table written to: {output_csv}\n")

    print_trial_table(df)
    print_summary(df)
    print_permutation_test(results, n_permutations, max_shift)
    write_stats(f"agonistic_windows{suffix}", stats_values(df, results, n_permutations))


if __name__ == "__main__":
    app()
