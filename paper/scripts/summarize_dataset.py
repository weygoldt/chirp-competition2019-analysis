"""The competition dataset in numbers: trials, recordings and annotated behaviour.

New step (no counterpart in the old repository, where the Methods counted
these numbers by hand). Nothing here depends on chirp detections; the chirp
counts are written by ``compare_detection_counts.py`` ("counts"). Writes
write_stats("dataset"):

- the trials of ``trial_summary.csv``: draws, the four excluded recordings
  (``BAD_RECORDINGS``) and the analysed trials (no draw, not excluded), per
  year and per sex combination (winner, loser);
- the duration and electrode count of the analysed recordings (headers of the
  raw grid recordings);
- the trials with annotated behaviour (BORIS; the selection of
  ``is_bad_trial``: group >= 5, LED files, no draw, not excluded) and, in them,
  the chases and contacts of the whole recording and of the first 3 h (the
  active phase the analyses use), and the intervals between consecutive chase
  onsets within the active phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from thunderlab.dataloader import DataLoader

from analyze_event_time_rates_trial import load_and_convert_boris_events
from manuscript_stats import write_stats
from paper_config import DATA_ROOT
from summarize_submissive_chirp_effect_on_dominant import BAD_RECORDINGS, is_bad_trial

ACTIVE_S = 3 * 3600


def recording_hours(recording: str) -> tuple[float, int]:
    """Duration (h) and electrode count of a raw grid recording."""
    with DataLoader(str(DATA_ROOT / recording)) as data:
        return len(data) / data.rate / 3600, data.channels


def annotated_events(trials: pd.DataFrame) -> dict:
    """Chases and contacts of the annotated trials, whole recording and active phase.

    A chase is in the active phase if it starts and ends before 3 h, as in
    ``summarize_all_timing_distributions.py``.
    """
    chases, contacts, onset_gaps = [], [], []
    for _, trial in trials.iterrows():
        contact_t, chase_on_off, _, _ = load_and_convert_boris_events(
            DATA_ROOT / trial["recording"], trial["recording"], sr=20_000)
        active = (chase_on_off[:, 0] < ACTIVE_S) & (chase_on_off[:, 1] < ACTIVE_S)
        chases.append((len(chase_on_off), int(active.sum())))
        contacts.append((len(contact_t), int((contact_t < ACTIVE_S).sum())))
        onset_gaps.append(np.diff(np.sort(chase_on_off[active, 0])))

    def counts(pairs: list[tuple[int, int]]) -> dict:
        whole, active = (sum(p) for p in zip(*pairs, strict=True))
        return {"whole": whole, "first_3h": active, "share_first_3h": active / whole}

    trial_medians = [np.median(g) for g in onset_gaps]
    return {
        "chases": counts(chases),
        "contacts": counts(contacts),
        "chase_onset_interval_s": {"median": np.median(np.concatenate(onset_gaps)),
                                   "trial_medians_range": [min(trial_medians),
                                                           max(trial_medians)]},
    }


def main() -> None:
    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    excluded = ts["recording"].isin(BAD_RECORDINGS)
    analysed = ts[(ts["draw"] == 0) & ~excluded]
    group5 = ts[ts["group"] >= 5]
    annotated = ts[[not is_bad_trial(row, DATA_ROOT) for _, row in ts.iterrows()]]

    hours, channels = zip(*(recording_hours(r) for r in analysed["recording"]), strict=True)
    sexes = (analysed["sex_win"] + analysed["sex_lose"]).value_counts()  # "fm": female winner
    stats = {
        "n_trials": len(ts),
        "n_draws": ts["draw"].sum(),
        "n_excluded": excluded.sum(),
        "n_excluded_not_draw": (excluded & (ts["draw"] == 0)).sum(),
        "n_analysed": len(analysed),
        "analysed_per_year": analysed["recording"].str[:4].value_counts().to_dict(),
        "analysed_sex_winner_loser": {"ff": sexes.get("ff", 0), "fm": sexes.get("fm", 0),
                                      "mf": sexes.get("mf", 0), "mm": sexes.get("mm", 0)},
        "recordings": {"hours_total": sum(hours), "hours_range": [min(hours), max(hours)],
                       "electrodes": sorted(set(channels))},
        "annotated": {
            "n_group_5": len(group5),
            "n_group_5_draws": group5["draw"].sum(),
            "n_group_5_excluded": group5["recording"].isin(BAD_RECORDINGS).sum(),
            "n_trials": len(annotated),
            **annotated_events(annotated),
        },
    }
    if sum(stats["analysed_sex_winner_loser"].values()) != len(analysed):
        raise ValueError(f"unexpected sex codes: {sorted(sexes.index)}")
    print(stats)
    write_stats("dataset", stats)


if __name__ == "__main__":
    main()
