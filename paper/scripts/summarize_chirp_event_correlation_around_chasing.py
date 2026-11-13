"""
Build per-chase rows of chirp counts in the fixed [-3, +13] s window
around each chase offset, for both winner and loser chirps. Used
downstream by the hurdle-model permutation test that asks whether chirp
rates in the m3p13 window exceed what would be expected if chirps were
randomly distributed across the recording.

For every chase event we report chirp counts in:

  * during chase                  [onset, offset]
  * m3p13 window around offset    [offset - 3, offset + 13]

The m3p13 window is NOT clipped by adjacent chases here -- every row has
the same 16 s exposure -- which is what lets the downstream hurdle-null
test compare observed and shuffled chirp counts on equal footing. (The
clipped variant lives in the receiver-effect / coupling pipeline, where
exposure variability matters for a different scientific question.)

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

# m3p13 window around the chase offset. Same definition as the
# receiver-effect / coupling analyses, but here NOT clipped by adjacent
# chases -- every row gets the same 16 s exposure.
M3P13_PRE = 3.0
M3P13_POST = 13.0
M3P13_LABEL = "fixed_m3p13"


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------
def extract_data_for_trial(
    trial_row: pd.Series,
    data_path: Path,
    randomize: bool,
    rng: np.random.Generator,
    t_max_cap: float = DEFAULT_T_MAX_CAP,
) -> list[dict]:
    """Return one dict per chase event in the trial."""
    trial_folder = data_path / trial_row["recording"]

    _, chasing_on_off_times, _, _ = load_and_convert_boris_events(
        trial_folder, trial_row["recording"], sr=20000
    )
    chirp_times = np.load(trial_folder / "chirp_times_rcnn.npy")
    chirp_ids = np.load(trial_folder / "chirp_ids_rcnn.npy")

    if np.isnan(chirp_times).any():
        raise ValueError(
            f"{trial_row['recording']}: NaN values in chirp_times_rcnn.npy "
            f"({int(np.isnan(chirp_times).sum())} of {len(chirp_times)})."
        )

    loser_chirp_times = chirp_times[chirp_ids == trial_row["lose_ID"]]
    winner_chirp_times = chirp_times[chirp_ids == trial_row["win_ID"]]

    loser_chirp_times = loser_chirp_times[loser_chirp_times < t_max_cap]
    winner_chirp_times = winner_chirp_times[winner_chirp_times < t_max_cap]

    if randomize:
        loser_chirp_times = randomize_chirp_times(loser_chirp_times, t_max_cap, rng)
        winner_chirp_times = randomize_chirp_times(winner_chirp_times, t_max_cap, rng)

    chasing_onsets = chasing_on_off_times[:, 0]
    chasing_offsets = chasing_on_off_times[:, 1]

    # Drop chasing events past the active-phase cap.
    in_window = (chasing_onsets < t_max_cap) & (chasing_offsets < t_max_cap)
    chasing_onsets = chasing_onsets[in_window]
    chasing_offsets = chasing_offsets[in_window]

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
        # m3p13 window around offset: [offset - 3, offset + 13], unclipped.
        # ------------------------------------------------------------------
        m_lo = chase_offset - M3P13_PRE
        m_hi = chase_offset + M3P13_POST
        row[f"n_loser_chirps_{M3P13_LABEL}"] = count_chirps_in_interval(
            loser_chirp_times, m_lo, m_hi
        )
        row[f"n_winner_chirps_{M3P13_LABEL}"] = count_chirps_in_interval(
            winner_chirp_times, m_lo, m_hi
        )
        row[f"{M3P13_LABEL}_window_width"] = float(m_hi - m_lo)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Trial loop
# ---------------------------------------------------------------------------
def get_data(
    data_path: Path,
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
            t_max_cap=t_max_cap,
        )
        all_rows.extend(chasing_rows)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    data_path: Path,
    num_permutations: int = 1000,
    seed: int = 42,
):
    """Generate `num_permutations` shuffled CSVs and one observed CSV."""

    filename = "chirps_around_chasing"
    save_dir = Path("data/processed/chirp_event_correlations_m3p13/data/chasing")

    if save_dir.exists() and any(save_dir.glob("*.csv")):
        print(f"Removing existing CSVs under {save_dir}.")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_rng = np.random.default_rng(seed)

    for i in track(range(num_permutations), description="Permuting"):
        # Each permutation gets its own substream so seeds are reproducible
        # but permutations are independent.
        perm_rng = np.random.default_rng(base_rng.integers(0, 2**63 - 1))
        perm_df = get_data(data_path, randomize=True, rng=perm_rng)
        perm_df.to_csv(save_dir / f"{filename}_perm_{i}.csv", index=False)

    obs_rng = np.random.default_rng(base_rng.integers(0, 2**63 - 1))
    final_df = get_data(data_path, randomize=False, rng=obs_rng)
    final_df.to_csv(save_dir / f"{filename}_obs.csv", index=False)

    print(f"\nWrote {num_permutations} permutations + 1 observed file to {save_dir}.")
    print(f"Observed file shape: {final_df.shape}")


if __name__ == "__main__":
    app()
