"""The trial summary with chirp counts from the chosen detections.

New step (no counterpart in the old repository). ``trial_summary.csv`` in the
data root was written by ``src/summarize_trial_summary.py`` of the old
analysis, whose chirp counts ``chirps_win`` / ``chirps_lose`` are the number
of v1 (``rcnn``) chirps of the winner and the loser over the whole recording
(summarize_trial_summary.py:590-592; verified equal for all 43 trials). That
script cannot be rerun as committed (it imports a module that no longer
exists) and would overwrite the file in the data root. This step copies the
data root's ``trial_summary.csv`` unchanged except for those two columns,
which it recounts from ``PAPER_DETECTIONS``, and writes it to
``$PAPER_OUT/data/processed/trial_summary.csv``. Everything else in the
summary (sizes, EODf, rises, chase counts, draws) does not depend on chirp
detections.
"""

from __future__ import annotations

import pandas as pd

from chirps import load_chirps
from paper_config import DATA_ROOT, DETECTIONS, processed


def main() -> None:
    src = DATA_ROOT / "trial_summary.csv"
    df = pd.read_csv(src, index_col=0)
    old = df[["chirps_win", "chirps_lose"]].copy()
    for i, row in df.iterrows():
        t, ids = load_chirps(DATA_ROOT / row["recording"])
        df.loc[i, "chirps_win"] = int((ids == row["win_ID"]).sum())
        df.loc[i, "chirps_lose"] = int((ids == row["lose_ID"]).sum())
    out = processed("trial_summary.csv")
    df.to_csv(out)
    print(f"{src} -> {out} (chirp counts from {DETECTIONS})")
    print(
        f"chirps_win: {old.chirps_win.sum()} -> {df.chirps_win.sum()}; "
        f"chirps_lose: {old.chirps_lose.sum()} -> {df.chirps_lose.sum()} (all 43 trials)"
    )


if __name__ == "__main__":
    main()
