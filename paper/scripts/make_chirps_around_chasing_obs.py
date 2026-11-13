"""Reconstructed writer of ``chirp_event_correlations/data/chasing/chirps_around_chasing_obs.csv``.

New step (reconstruction). The before/after-chasing-offset logistic
regression (``stats_bayesian_logreg_loser_chirps_before_vs_after_chasing_offset.py``)
reads this file, but the script version that wrote it in the old analysis
was never committed (audit item O5 of the old repository's
``docs/known_issues.md``); the committed ``src/summarize_chirp_event_correlation_around_chasing.py``
neither merges overlapping chases nor writes the ``*_20s_around_chasing_on``
columns, so it is not that writer.

The old file has the 1,958 rows and the 13 columns of the m3p13 observation
file (``summarize_chirp_event_correlation_around_chasing.py``, same chase
handling: chases and chirps before 3 h, overlapping chases merged) plus 12
columns: chirps of the loser and of the winner in [x - 10, x] ("10s_before"),
[x, x + 10] ("10s_after") and their sum ("20s_around") for x = chase onset
and offset, counted with ``count_chirps_in_interval``. This script builds
exactly that. On the v1 detections (``PAPER_DETECTIONS=rcnn``) it reproduces
all 25 columns of the old file value for value (checked 2026-11-13 during
the port; the 12 reconstructed columns by a read-only check, the 13 shared
ones because they equal the old m3p13 file). Only the observed file is
written; the regression reads no permutations.

Run after ``summarize_chirp_event_correlation_around_chasing.py``.
"""

from __future__ import annotations

import pandas as pd

from analyze_event_time_rates_trial import load_and_convert_boris_events
from chirps import load_chirps
from paper_config import DATA_ROOT, PROCESSED, processed
from summarize_submissive_chirp_effect_on_dominant import (
    _merge_overlapping_chases,
    count_chirps_in_interval,
    is_bad_trial,
)

T_MAX_S = 3 * 60 * 60
WINDOW_S = 10
OLD_COLUMNS = [
    "recording", "winner_id", "loser_id", "winner_sex", "loser_sex",
    "winner_experience", "loser_experience", "chasing_duration",
    "n_loser_chirps_during_chasing", "n_winner_chirps_during_chasing",
    "n_loser_chirps_10s_before_chasing_on", "n_winner_chirps_10s_before_chasing_on",
    "n_loser_chirps_10s_after_chasing_on", "n_winner_chirps_10s_after_chasing_on",
    "n_loser_chirps_20s_around_chasing_on", "n_winner_chirps_20s_around_chasing_on",
    "n_loser_chirps_10s_before_chasing_off", "n_winner_chirps_10s_before_chasing_off",
    "n_loser_chirps_10s_after_chasing_off", "n_winner_chirps_10s_after_chasing_off",
    "n_loser_chirps_20s_around_chasing_off", "n_winner_chirps_20s_around_chasing_off",
    "n_loser_chirps_fixed_m3p13", "n_winner_chirps_fixed_m3p13", "fixed_m3p13_window_width",
]


def window_counts(source: str | None = None) -> pd.DataFrame:
    """The 12 before/after/around columns per merged chase (plus ``recording``)."""
    summary = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    rows = []
    for _, trial in summary.iterrows():
        if is_bad_trial(trial, DATA_ROOT):
            continue
        folder = DATA_ROOT / trial["recording"]
        _, on_off, _, _ = load_and_convert_boris_events(folder, trial["recording"], sr=20_000)
        times, ids = load_chirps(folder, source)
        chirps = {
            "loser": times[(ids == trial["lose_ID"]) & (times < T_MAX_S)],
            "winner": times[(ids == trial["win_ID"]) & (times < T_MAX_S)],
        }
        onsets, offsets = on_off[:, 0], on_off[:, 1]
        keep = (onsets < T_MAX_S) & (offsets < T_MAX_S)
        onsets, offsets = _merge_overlapping_chases(onsets[keep], offsets[keep], trial["recording"])
        for onset, offset in zip(onsets, offsets):
            row = {"recording": trial["recording"]}
            for who, c in chirps.items():
                for edge, x in (("on", onset), ("off", offset)):
                    before = count_chirps_in_interval(c, x - WINDOW_S, x)
                    after = count_chirps_in_interval(c, x, x + WINDOW_S)
                    row[f"n_{who}_chirps_10s_before_chasing_{edge}"] = before
                    row[f"n_{who}_chirps_10s_after_chasing_{edge}"] = after
                    row[f"n_{who}_chirps_20s_around_chasing_{edge}"] = before + after
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    m3p13 = pd.read_csv(
        PROCESSED / "chirp_event_correlations_m3p13" / "data" / "chasing"
        / "chirps_around_chasing_obs.csv"
    )
    extra = window_counts()
    if len(extra) != len(m3p13) or not (extra["recording"].to_numpy() == m3p13["recording"].to_numpy()).all():
        raise RuntimeError(
            f"chase rows do not line up with the m3p13 file ({len(extra)} vs {len(m3p13)})"
        )
    out = pd.concat([m3p13, extra.drop(columns="recording")], axis=1)[OLD_COLUMNS]
    path = processed("chirp_event_correlations", "data", "chasing", "chirps_around_chasing_obs.csv")
    out.to_csv(path, index=False)
    print(f"{len(out)} chases -> {path}")


if __name__ == "__main__":
    main()
