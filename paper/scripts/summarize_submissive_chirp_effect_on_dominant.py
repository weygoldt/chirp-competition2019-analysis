"""
Build per-chase-event summary rows from BORIS annotations and RCNN chirp
detections.

For each chase event we count loser and winner chirps in a fixed
[-3, +13] s window around the chase offset. The window is clipped to
keep it inside the interval between the current chase's onset and the
next chase's onset: the left clip to `chase_onset` excludes the
previous inter-chase interval when the chase is very short (duration
< 3 s); the right clip to `next_onset` excludes any part of the next
chase. These two clips are deliberately asymmetric --- the left side
can still extend into the current chase, but neither side ever
overlaps a neighbouring chase event.

The fixed window is additionally decomposed into two halves at the
chase offset, with a third post-offset variant that ignores the
next-chase clip:

  chase_tail                : [max(offset - 3, onset), offset]
                              Width depends on D only. Captures chirps
                              during the last 3 s of the chase.
  post_offset_clipped       : [offset, min(offset + 13, next_onset)]
                              Width depends on next-ICI only. Captures
                              chirps after the chase, never overlaps
                              the next chase.
  post_offset_unclipped     : [offset, offset + 13]
                              Width always 13 s. Captures chirps after
                              the chase; may overlap the next chase's
                              pre-onset period in dense bouts. Paired
                              with post_offset_clipped to bracket the
                              window-length coupling in either
                              direction.

Also records `time_to_next_chasing`, `time_since_previous_chasing`, and
`next_chasing_duration` as bout-structure covariates.

The `randomize=True` branch replaces each fish's chirp times with
uniform draws on [0, t_max], where `t_max` is the latest observed event
in the recording (capped at 3 h).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from analyze_event_time_rates_trial import load_and_convert_boris_events

app = typer.Typer(pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BAD_RECORDINGS = [
    "2020-05-21-10_00",
    "2020-05-25-10_00",
    "2020-05-28-10_00",
    "2020-06-10-10_00",
]

# Fixed [-3, +13] s window around the chase offset.
FIXED_LEFT = 3.0
FIXED_RIGHT = 13.0

# Upper bound for randomization support and for chase/chirp filtering
# when no shorter per-recording bound is available.
DEFAULT_T_MAX_CAP = 3 * 60 * 60  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_bad_trial(trial_row: pd.Series, data_path: Path) -> bool:
    trial_folder = data_path / trial_row["recording"]
    if trial_row["group"] < 5:
        return True
    if not (trial_folder / "led_idxs.csv").exists():
        return True
    if not (trial_folder / "LED_frames.npy").exists():
        return True
    if trial_row["draw"] == 1:
        return True
    if trial_row["recording"] in BAD_RECORDINGS:
        return True
    return False


def randomize_chirp_times(
    chirp_times: np.ndarray, t_max: float, rng: np.random.Generator
) -> np.ndarray:
    """Replace chirp times with uniform draws on [0, t_max]."""
    return rng.uniform(0.0, t_max, size=len(chirp_times))


def count_chirps_in_interval(chirp_times: np.ndarray, start: float, end: float) -> int:
    if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
        return 0
    return int(np.sum((chirp_times >= start) & (chirp_times <= end)))


def _merge_overlapping_chases(
    onsets: np.ndarray, offsets: np.ndarray, recording: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(onsets) != len(offsets):
        raise ValueError(
            f"{recording}: chasing onset/offset length mismatch "
            f"({len(onsets)} vs {len(offsets)})."
        )
    if len(onsets) == 0:
        return onsets, offsets
    if np.any(np.diff(onsets) < 0):
        raise ValueError(f"{recording}: chasing onsets not monotonically increasing.")
    if np.any(offsets - onsets < 0):
        raise ValueError(f"{recording}: negative chasing durations detected.")

    merged_on = [float(onsets[0])]
    merged_off = [float(offsets[0])]
    n_merged = 0
    for on, off in zip(onsets[1:], offsets[1:]):
        on, off = float(on), float(off)
        if on <= merged_off[-1]:
            new_off = max(merged_off[-1], off)
            print(
                f"[merge] {recording}: overlapping chase "
                f"[{on:.3f}, {off:.3f}] absorbed into "
                f"[{merged_on[-1]:.3f}, {new_off:.3f}]."
            )
            merged_off[-1] = new_off
            n_merged += 1
        else:
            merged_on.append(on)
            merged_off.append(off)

    if n_merged:
        print(
            f"[merge] {recording}: {n_merged} overlap(s) merged; "
            f"{len(onsets)} -> {len(merged_on)} chases."
        )

    return np.asarray(merged_on), np.asarray(merged_off)


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------
def extract_chasing_data_for_trial(
    trial_row: pd.Series,
    data_path: Path,
    randomize: bool,
    rng: np.random.Generator,
    t_max_cap: float = DEFAULT_T_MAX_CAP,
) -> tuple[list[dict], int, int]:
    """Return (rows, n_chasing, n_contact) for one trial."""
    trial_folder = data_path / trial_row["recording"]

    contact_times, chasing_on_off_times, _, _ = load_and_convert_boris_events(
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

    observed_max = max(
        float(chirp_times.max()) if len(chirp_times) else 0.0,
        float(chasing_on_off_times[:, 1].max()) if chasing_on_off_times.size else 0.0,
        float(contact_times.max()) if len(contact_times) else 0.0,
    )
    t_max = min(observed_max, t_max_cap) if observed_max > 0 else t_max_cap

    loser_chirp_times = loser_chirp_times[loser_chirp_times < t_max]
    winner_chirp_times = winner_chirp_times[winner_chirp_times < t_max]

    if randomize:
        loser_chirp_times = randomize_chirp_times(loser_chirp_times, t_max, rng)
        winner_chirp_times = randomize_chirp_times(winner_chirp_times, t_max, rng)

    chasing_onsets = chasing_on_off_times[:, 0]
    chasing_offsets = chasing_on_off_times[:, 1]

    keep = (chasing_onsets < t_max) & (chasing_offsets < t_max)
    chasing_onsets = chasing_onsets[keep]
    chasing_offsets = chasing_offsets[keep]
    chasing_onsets, chasing_offsets = _merge_overlapping_chases(
        chasing_onsets, chasing_offsets, trial_row["recording"]
    )

    contact_times = contact_times[contact_times < t_max]

    rows: list[dict] = []
    for i in range(len(chasing_onsets)):
        chase_onset = chasing_onsets[i]
        chase_offset = chasing_offsets[i]

        prev_offset = chasing_offsets[i - 1] if i > 0 else -np.inf
        next_onset = (
            chasing_onsets[i + 1] if (i + 1) < len(chasing_onsets) else np.inf
        )

        # --- full fixed [-3, +13] window (asymmetric clip) -----------------
        fixed_start = max(chase_offset - FIXED_LEFT, chase_onset)
        fixed_end = min(chase_offset + FIXED_RIGHT, next_onset)
        fixed_duration = max(fixed_end - fixed_start, 0.0)

        n_loser_fixed = count_chirps_in_interval(
            loser_chirp_times, fixed_start, fixed_end
        )
        n_winner_fixed = count_chirps_in_interval(
            winner_chirp_times, fixed_start, fixed_end
        )

        # --- decomposition windows -----------------------------------------
        # Chase-tail [max(offset - 3, onset), offset]: width depends on D.
        tail_start = max(chase_offset - FIXED_LEFT, chase_onset)
        tail_end = chase_offset
        tail_duration = max(tail_end - tail_start, 0.0)
        n_loser_tail = count_chirps_in_interval(
            loser_chirp_times, tail_start, tail_end
        )
        n_winner_tail = count_chirps_in_interval(
            winner_chirp_times, tail_start, tail_end
        )

        # Post-offset clipped [offset, min(offset + 13, next_onset)]:
        # width depends on next-ICI; never overlaps the next chase.
        post_clip_end = min(chase_offset + FIXED_RIGHT, next_onset)
        post_clip_duration = max(post_clip_end - chase_offset, 0.0)
        n_loser_post_clip = count_chirps_in_interval(
            loser_chirp_times, chase_offset, post_clip_end
        )
        n_winner_post_clip = count_chirps_in_interval(
            winner_chirp_times, chase_offset, post_clip_end
        )

        # Post-offset unclipped [offset, offset + 13]: fixed 13 s width,
        # may overlap the next chase's pre-onset period. Paired with the
        # clipped variant to bracket window-length coupling in the
        # opposite direction.
        post_unclip_end = chase_offset + FIXED_RIGHT
        post_unclip_duration = FIXED_RIGHT
        n_loser_post_unclip = count_chirps_in_interval(
            loser_chirp_times, chase_offset, post_unclip_end
        )
        n_winner_post_unclip = count_chirps_in_interval(
            winner_chirp_times, chase_offset, post_unclip_end
        )

        # --- chirps during the chase itself --------------------------------
        n_loser_in = count_chirps_in_interval(
            loser_chirp_times, chase_onset, chase_offset
        )
        n_winner_in = count_chirps_in_interval(
            winner_chirp_times, chase_onset, chase_offset
        )

        rows.append(
            {
                "recording": trial_row["recording"],
                "winner_id": trial_row["win_ID"],
                "loser_id": trial_row["lose_ID"],
                "winner_sex": trial_row["sex_win"],
                "loser_sex": trial_row["sex_lose"],
                "winner_experience": trial_row["exp_win"],
                "loser_experience": trial_row["exp_lose"],
                "chasing_duration": chase_offset - chase_onset,
                # in-chase
                "n_loser_chirps_in_chasing": n_loser_in,
                "n_winner_chirps_in_chasing": n_winner_in,
                "loser_chirped_in_chasing": n_loser_in > 0,
                "winner_chirped_in_chasing": n_winner_in > 0,
                # full fixed [-3, +13] window
                "fixed_window_start": fixed_start,
                "fixed_window_end": fixed_end,
                "fixed_window_duration": fixed_duration,
                "n_loser_chirps_fixed_m3p13": n_loser_fixed,
                "n_winner_chirps_fixed_m3p13": n_winner_fixed,
                "loser_chirp_rate_fixed_m3p13": (
                    n_loser_fixed / fixed_duration if fixed_duration > 0 else np.nan
                ),
                "winner_chirp_rate_fixed_m3p13": (
                    n_winner_fixed / fixed_duration if fixed_duration > 0 else np.nan
                ),
                "loser_chirped_fixed_m3p13": n_loser_fixed > 0,
                "winner_chirped_fixed_m3p13": n_winner_fixed > 0,
                # chase-tail half (width depends on D only)
                "chase_tail_duration": tail_duration,
                "n_loser_chirps_chase_tail": n_loser_tail,
                "n_winner_chirps_chase_tail": n_winner_tail,
                "loser_chirped_chase_tail": n_loser_tail > 0,
                "winner_chirped_chase_tail": n_winner_tail > 0,
                # post-offset, clipped to next chase (width depends on next-ICI)
                "post_offset_clipped_duration": post_clip_duration,
                "n_loser_chirps_post_offset_clipped": n_loser_post_clip,
                "n_winner_chirps_post_offset_clipped": n_winner_post_clip,
                "loser_chirped_post_offset_clipped": n_loser_post_clip > 0,
                "winner_chirped_post_offset_clipped": n_winner_post_clip > 0,
                # post-offset, unclipped (fixed 13 s width)
                "post_offset_unclipped_duration": post_unclip_duration,
                "n_loser_chirps_post_offset_unclipped": n_loser_post_unclip,
                "n_winner_chirps_post_offset_unclipped": n_winner_post_unclip,
                "loser_chirped_post_offset_unclipped": n_loser_post_unclip > 0,
                "winner_chirped_post_offset_unclipped": n_winner_post_unclip > 0,
                # bout-structure covariates
                "time_to_next_chasing": (
                    next_onset - chase_offset if np.isfinite(next_onset) else np.nan
                ),
                "time_since_previous_chasing": (
                    chase_onset - prev_offset if np.isfinite(prev_offset) else np.nan
                ),
                "next_chasing_duration": (
                    chasing_offsets[i + 1] - chasing_onsets[i + 1]
                    if (i + 1) < len(chasing_onsets)
                    else np.nan
                ),
            }
        )

    return rows, len(chasing_onsets), len(contact_times)


# ---------------------------------------------------------------------------
# Trial aggregation
# ---------------------------------------------------------------------------
def get_data(
    data_path: Path,
    randomize: bool = False,
    t_max_cap: float = DEFAULT_T_MAX_CAP,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    summary_df = pd.read_csv(data_path / "trial_summary.csv", index_col=0)

    all_rows: list[dict] = []
    n_used = n_skipped = total_chasing = total_contact = 0

    for _, trial_row in summary_df.iterrows():
        if is_bad_trial(trial_row, data_path):
            n_skipped += 1
            continue

        rows, n_chasing, n_contact = extract_chasing_data_for_trial(
            trial_row=trial_row,
            data_path=data_path,
            randomize=randomize,
            rng=rng,
            t_max_cap=t_max_cap,
        )
        all_rows.extend(rows)
        total_chasing += n_chasing
        total_contact += n_contact
        n_used += 1

    df = pd.DataFrame(all_rows)
    print(
        f"  trials: used={n_used}, skipped={n_skipped}; "
        f"chases={total_chasing}, contacts={total_contact}; "
        f"rows={len(df)}"
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@app.command()
def main(
    data_path: Path,
    num_permutations: int = 300,
    seed: int = 42,
    save_dir: Path = Path(
        "data/processed/chirp_effect_on_dominant_adaptive_window/data"
    ),
) -> None:
    """Generate permutation baselines and one observed dataset."""
    filename = "chirp_effect_on_dominant"

    if save_dir.exists() and any(save_dir.glob("*.csv")):
        print(f"Removing existing CSVs in {save_dir}.")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for i in range(num_permutations):
        print(f"Permutation {i + 1}/{num_permutations}")
        perm_df = get_data(data_path, randomize=True, rng=rng)
        perm_df.to_csv(save_dir / f"{filename}_perm_{i}.csv", index=False)

    print("Observed run")
    final_df = get_data(data_path, randomize=False, rng=rng)
    final_df.to_csv(save_dir / f"{filename}_obs.csv", index=False)


if __name__ == "__main__":
    app()
