"""Chirp counts per trial and status: v1 (``rcnn``) against v2 (``rtdetr``).

Reads both legacy array pairs of every trial in ``trial_summary.csv`` and
writes ``$PAPER_OUT/report/detection_counts.csv`` (one row per trial) and
``detection_counts.md`` (totals and per-status summaries for the 33 analysed
trials). Counts are given for the whole recording and for the first 3 h
(the active phase the analyses use). The trial selection is the one of the
analysis scripts: drop draws and the four recordings with too-close EODfs
(``bad_recordings`` of ``analyze_event_time_rates_trial.py``).

Also checks whether ``trial_summary.csv``'s ``chirps_win`` / ``chirps_lose``
are the v1 counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chirps import load_chirps
from paper_config import DATA_ROOT, OUT

BAD_RECORDINGS = [
    "2020-05-21-10_00",
    "2020-05-25-10_00",
    "2020-05-28-10_00",
    "2020-06-10-10_00",
]
ACTIVE_S = 3 * 3600


def main() -> None:
    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv")
    rows = []
    for _, r in ts.iterrows():
        row = {
            "recording": r["recording"],
            "group": r["group"],
            "draw": int(r["draw"]),
            "bad_recording": r["recording"] in BAD_RECORDINGS,
            "win_ID": r["win_ID"],
            "lose_ID": r["lose_ID"],
            "summary_chirps_win": r["chirps_win"],
            "summary_chirps_lose": r["chirps_lose"],
        }
        for src, tag in (("rcnn", "v1"), ("rtdetr", "v2")):
            t, ids = load_chirps(DATA_ROOT / r["recording"], src)
            for status, fid in (("win", r["win_ID"]), ("lose", r["lose_ID"])):
                sel = ids == fid
                row[f"{tag}_{status}"] = int(sel.sum())
                row[f"{tag}_{status}_3h"] = int((sel & (t < ACTIVE_S)).sum())
            row[f"{tag}_other_ids"] = int((~np.isin(ids, [r["win_ID"], r["lose_ID"]])).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df["included"] = (df["draw"] == 0) & ~df["bad_recording"]

    out = OUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "detection_counts.csv", index=False)

    inc = df[df["included"]]
    lines = [
        "# Chirp counts per trial: v1 (rcnn) vs v2 (rtdetr)",
        "",
        f"Trials in trial_summary.csv: {len(df)}; analysed (no draw, not bad_recordings): {len(inc)}.",
        f"trial_summary chirps_win/lose equal v1 whole-recording counts in "
        f"{int(((df.summary_chirps_win == df.v1_win) & (df.summary_chirps_lose == df.v1_lose)).sum())}"
        f" of {len(df)} trials.",
        "",
        "| window | status | v1 total | v2 total | v2/v1 | per-trial median v1 | median v2 | median ratio (10-90 %) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for suffix, label in (("", "whole recording"), ("_3h", "first 3 h")):
        for status in ("win", "lose"):
            v1, v2 = inc[f"v1_{status}{suffix}"], inc[f"v2_{status}{suffix}"]
            ratio = (v2 / v1.replace(0, np.nan)).dropna()
            lines.append(
                f"| {label} | {'winner' if status == 'win' else 'loser'} | {v1.sum():,} | {v2.sum():,} | "
                f"{v2.sum() / v1.sum():.2f} | {v1.median():.0f} | {v2.median():.0f} | "
                f"{ratio.median():.2f} ({ratio.quantile(0.1):.2f}-{ratio.quantile(0.9):.2f}) |"
            )
    for tag in ("v1", "v2"):
        n_more = int((inc[f"{tag}_lose_3h"] > inc[f"{tag}_win_3h"]).sum())
        share = inc[f"{tag}_lose_3h"] / (inc[f"{tag}_lose_3h"] + inc[f"{tag}_win_3h"])
        lines.append(
            f"\n{tag}, first 3 h: losers chirp more than winners in {n_more} of {len(inc)} trials; "
            f"loser share median {share.median():.2f}."
        )
    zero = {tag: int(((inc[f"{tag}_win_3h"] == 0)).sum()) for tag in ("v1", "v2")}
    lines.append(f"\nWinners without chirps in the first 3 h: v1 {zero['v1']}, v2 {zero['v2']}.")
    (out / "detection_counts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
