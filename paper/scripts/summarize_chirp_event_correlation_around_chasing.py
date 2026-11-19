# Ported from chirp_competition d040123 (src_paper/summarize_chirp_event_correlation_around_chasing.py).
# Changes: chirps via chirps.load_chirps (PAPER_DETECTIONS); data root defaults to paper_config.DATA_ROOT;
# output under $PAPER_OUT (it still clears its own output directory first). Seeds unchanged. The loading
# of a trial's chases and chirps is factored out (trial_events; same order of random draws), so that
# window_overlap can count how far the windows overlap; that and the chase counts go to write_stats
# ("chase_windows"). The window comes from offset_window.py (was the constant -3, +13 s, "m3p13"); its
# columns, the output directory chirp_event_correlations_offset_window and the stats are named after it.
"""
Build per-chase rows of chirp counts in the offset window around each
chase offset (``offset_window.py``: [offset + left, offset + right],
left < 0 < right, from the loser chirp-rate peak of the primary run),
for both winner and loser chirps. Used downstream by the hurdle-model
permutation test that asks whether chirp rates in the offset window
exceed what would be expected if chirps were randomly distributed across
the recording.

For every chase event we report chirp counts in:

  * during chase                  [onset, offset]
  * offset window                 [offset + left, offset + right]

The offset window is NOT clipped by adjacent chases here -- every row
has the same exposure (right - left s) -- which is what lets the
downstream hurdle-null test compare observed and shuffled chirp counts
on equal footing. (The clipped variant lives in the receiver-effect /
coupling pipeline, where exposure variability matters for a different
scientific question.)

The `randomize=True` branch replaces each fish's chirp times with
uniform draws on [0, t_max], where `t_max` is the latest event time in
the recording (capped at 3 h).

Reuses helpers from `summarize_submissive_chirp_effect_on_dominant.py`:
`is_bad_trial`, `randomize_chirp_times`, `count_chirps_in_interval`,
`_merge_overlapping_chases`. Those handle the validity rules, the safe
random regeneration, the empty-interval guard, and the
overlapping-chase merge that would otherwise create negative inter-chase
intervals.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.progress import track

from analyze_event_time_rates_trial import load_and_convert_boris_events
from chirps import load_chirps
from manuscript_stats import write_stats
from offset_window import offset_window
from paper_config import DATA_ROOT, PROCESSED
from summarize_submissive_chirp_effect_on_dominant import (
    BAD_RECORDINGS,  # noqa: F401  -- re-exported for visibility
    _merge_overlapping_chases,
    count_chirps_in_interval,
    is_bad_trial,
    randomize_chirp_times,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_T_MAX_CAP = 3 * 60 * 60  # seconds; cap at the 3-h active phase

# The offset window (offset_window.py) has the same definition as in the
# receiver-effect / coupling analyses, but here it is NOT clipped by
# adjacent chases -- every row gets the same exposure. Its columns:
WINDOW_LABEL = "offset_window"


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------
def trial_events(
    trial_row: pd.Series,
    data_path: Path,
    t_max_cap: float = DEFAULT_T_MAX_CAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chase onsets and offsets (before merging) and loser and winner chirp
    times of one trial, all within the active phase."""
    trial_folder = data_path / trial_row["recording"]

    _, chasing_on_off_times, _, _ = load_and_convert_boris_events(
        trial_folder, trial_row["recording"], sr=20000
    )
    chirp_times, chirp_ids = load_chirps(trial_folder)

    if np.isnan(chirp_times).any():
        raise ValueError(
            f"{trial_row['recording']}: NaN values in the chirp times "
            f"({int(np.isnan(chirp_times).sum())} of {len(chirp_times)})."
        )

    loser_chirp_times = chirp_times[chirp_ids == trial_row["lose_ID"]]
    winner_chirp_times = chirp_times[chirp_ids == trial_row["win_ID"]]

    loser_chirp_times = loser_chirp_times[loser_chirp_times < t_max_cap]
    winner_chirp_times = winner_chirp_times[winner_chirp_times < t_max_cap]

    chasing_onsets = chasing_on_off_times[:, 0]
    chasing_offsets = chasing_on_off_times[:, 1]

    # Drop chasing events past the active-phase cap.
    in_window = (chasing_onsets < t_max_cap) & (chasing_offsets < t_max_cap)
    return (chasing_onsets[in_window], chasing_offsets[in_window],
            loser_chirp_times, winner_chirp_times)


def extract_data_for_trial(
    trial_row: pd.Series,
    data_path: Path,
    randomize: bool,
    rng: np.random.Generator,
    window: tuple[float, float],
    t_max_cap: float = DEFAULT_T_MAX_CAP,
) -> list[dict]:
    """Return one dict per chase event in the trial; `window` is the offset
    window (left, right) in s relative to the chase offset."""
    left, right = window
    chasing_onsets, chasing_offsets, loser_chirp_times, winner_chirp_times = trial_events(
        trial_row, data_path, t_max_cap
    )

    if randomize:
        loser_chirp_times = randomize_chirp_times(loser_chirp_times, t_max_cap, rng)
        winner_chirp_times = randomize_chirp_times(winner_chirp_times, t_max_cap, rng)

    # Same merge logic as the receiver-effect script: keeps onsets
    # monotonic, removes overlapping pairs, fails loud on bad data.
    chasing_onsets, chasing_offsets = _merge_overlapping_chases(
        chasing_onsets, chasing_offsets, trial_row["recording"]
    )

    rows: list[dict] = []
    for chase_onset, chase_offset in zip(chasing_onsets, chasing_offsets):
        row: dict = {
            "recording": trial_row["recording"],
            "winner_id": trial_row["win_ID"],
            "loser_id": trial_row["lose_ID"],
            "winner_sex": trial_row["sex_win"],
            "loser_sex": trial_row["sex_lose"],
            "winner_experience": trial_row["exp_win"],
            "loser_experience": trial_row["exp_lose"],
            "chasing_duration": float(chase_offset - chase_onset),
        }

        # During the chase itself.
        row["n_loser_chirps_during_chasing"] = count_chirps_in_interval(
            loser_chirp_times, chase_onset, chase_offset
        )
        row["n_winner_chirps_during_chasing"] = count_chirps_in_interval(
            winner_chirp_times, chase_onset, chase_offset
        )

        # ------------------------------------------------------------------
        # Offset window: [offset + left, offset + right], unclipped.
        # ------------------------------------------------------------------
        m_lo = chase_offset + left
        m_hi = chase_offset + right
        row[f"n_loser_chirps_{WINDOW_LABEL}"] = count_chirps_in_interval(
            loser_chirp_times, m_lo, m_hi
        )
        row[f"n_winner_chirps_{WINDOW_LABEL}"] = count_chirps_in_interval(
            winner_chirp_times, m_lo, m_hi
        )
        row[f"{WINDOW_LABEL}_width"] = float(m_hi - m_lo)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Trial loop
# ---------------------------------------------------------------------------
def get_data(
    data_path: Path,
    window: tuple[float, float],
    randomize: bool = False,
    rng: np.random.Generator | None = None,
    t_max_cap: float = DEFAULT_T_MAX_CAP,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    summary_df = pd.read_csv(data_path / "trial_summary.csv", index_col=0)

    all_rows: list[dict] = []
    for _, trial_row in summary_df.iterrows():
        if is_bad_trial(trial_row, data_path):
            continue

        chasing_rows = extract_data_for_trial(
            trial_row=trial_row,
            data_path=data_path,
            randomize=randomize,
            rng=rng,
            window=window,
            t_max_cap=t_max_cap,
        )
        all_rows.extend(chasing_rows)

    return pd.DataFrame(all_rows)


def window_overlap(
    data_path: Path, window: tuple[float, float], t_max_cap: float = DEFAULT_T_MAX_CAP
) -> dict:
    """How far the offset windows of the observed chases overlap (audit item O4
    of the old analysis): merged chases, consecutive offsets of a trial closer
    than the window width, and loser chirps that fall in more than one window."""
    summary_df = pd.read_csv(data_path / "trial_summary.csv", index_col=0)
    left, right = window
    width = right - left
    n = dict(chases_before_merging=0, chases=0, offset_gaps=0, offset_gaps_below_width=0,
             loser_chirps=0, loser_chirps_in_a_window=0, loser_chirps_in_several_windows=0)
    for _, trial_row in summary_df.iterrows():
        if is_bad_trial(trial_row, data_path):
            continue
        onsets, offsets, loser, _ = trial_events(trial_row, data_path, t_max_cap)
        n["chases_before_merging"] += len(onsets)
        _, offsets = _merge_overlapping_chases(onsets, offsets, trial_row["recording"])
        n["chases"] += len(offsets)
        gaps = np.diff(offsets)
        n["offset_gaps"] += len(gaps)
        n["offset_gaps_below_width"] += int((gaps < width).sum())
        # windows per chirp, with the bounds of count_chirps_in_interval
        k = ((loser[:, None] >= offsets[None, :] + left)
             & (loser[:, None] <= offsets[None, :] + right)).sum(axis=1)
        n["loser_chirps"] += len(loser)
        n["loser_chirps_in_a_window"] += int((k >= 1).sum())
        n["loser_chirps_in_several_windows"] += int((k >= 2).sum())
    return {
        **n,
        "window_width_s": width,
        "share_offset_gaps_below_width": n["offset_gaps_below_width"] / n["offset_gaps"],
        "share_windowed_loser_chirps_in_several": (
            n["loser_chirps_in_several_windows"] / n["loser_chirps_in_a_window"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    data_path: Path = DATA_ROOT,
    num_permutations: int = 1000,
    seed: int = 42,
):
    """Generate `num_permutations` shuffled CSVs and one observed CSV."""

    filename = "chirps_around_chasing"
    save_dir = PROCESSED / "chirp_event_correlations_offset_window" / "data" / "chasing"
    window = offset_window()
    print(f"Offset window: {window[0]:+g} to {window[1]:+g} s around chase offsets")

    if save_dir.exists() and any(save_dir.glob("*.csv")):
        print(f"Removing existing CSVs under {save_dir}.")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_rng = np.random.default_rng(seed)

    for i in track(range(num_permutations), description="Permuting"):
        # Each permutation gets its own substream so seeds are reproducible
        # but permutations are independent.
        perm_rng = np.random.default_rng(base_rng.integers(0, 2**63 - 1))
        perm_df = get_data(data_path, window, randomize=True, rng=perm_rng)
        perm_df.to_csv(save_dir / f"{filename}_perm_{i}.csv", index=False)

    obs_rng = np.random.default_rng(base_rng.integers(0, 2**63 - 1))
    final_df = get_data(data_path, window, randomize=False, rng=obs_rng)
    final_df.to_csv(save_dir / f"{filename}_obs.csv", index=False)

    print(f"\nWrote {num_permutations} permutations + 1 observed file to {save_dir}.")
    print(f"Observed file shape: {final_df.shape}")

    overlap = window_overlap(data_path, window)
    if overlap["chases"] != len(final_df):
        raise RuntimeError(f"{overlap['chases']} chases in the overlap count, {len(final_df)} rows")
    print(overlap)
    write_stats("chase_windows", {"n_trials": final_df["recording"].nunique(), **overlap})


if __name__ == "__main__":
    app()
