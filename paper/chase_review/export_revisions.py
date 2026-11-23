"""Table of the reviewed chases: BORIS vs your onset / offset, one row per chase (the latest save).

    paper/chase_review/export_revisions.py NAME [--root DIR] [--out DIR]

Reads ``session.json`` and ``revisions.jsonl`` of session NAME and writes
``paper/output/chase_review/NAME/revisions.csv`` (derived data stays in the repository's
git-ignored output) and prints a summary: how many chases were reviewed, kept, moved or
rejected, and the median shift (yours minus BORIS, recording seconds) of onsets and
offsets, for all reviews and for blind ones (BORIS markers hidden).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SESSIONS = Path("/mnt/data1/chirp_v2/labels/chase_review_2019")
OUT = HERE.parent / "output" / "chase_review"

COLUMNS = ["chase_id", "recording", "k", "in_fig6", "not_a_chase",
           "boris_onset_frame", "boris_offset_frame", "boris_onset_rec_s", "boris_offset_rec_s",
           "onset_frame", "offset_frame", "onset_rec_s", "offset_rec_s",
           "d_onset_frames", "d_offset_frames", "d_onset_s", "d_offset_s",
           "onset_unsure", "offset_unsure", "blind", "chirps_hidden", "shares_offset", "note", "reviewer", "saved_at"]


def latest(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rev = json.loads(line)
                out[rev["chase_id"]] = rev
    return out


def revision_table(session: dict, revisions: dict[str, dict]) -> pd.DataFrame:
    lo, hi = session["meta"]["active_phase_s"]
    chases = {c["id"]: c for c in session["chases"]}
    rows = []
    for cid, r in revisions.items():
        c = chases[cid]
        row = {k: r.get(k) for k in COLUMNS if k in r}
        row["in_fig6"] = bool(lo <= c["onset_rec_s"] and c["offset_rec_s"] < hi)
        row["shares_offset"] = bool(c["shares_offset_with"])
        if r.get("onset_frame") is not None:
            row["d_onset_frames"] = r["onset_frame"] - c["onset_frame"]
            row["d_offset_frames"] = r["offset_frame"] - c["offset_frame"]
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["recording", "k"]).reset_index(drop=True)


def summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "no reviewed chases yet"
    chase = df[~df["not_a_chase"].astype(bool)]
    kept = chase[(chase["d_onset_frames"] == 0) & (chase["d_offset_frames"] == 0)]
    lines = [f"{len(df)} reviewed: {len(kept)} kept as in BORIS, {len(chase) - len(kept)} moved, "
             f"{int(df['not_a_chase'].astype(bool).sum())} not a chase; {int(df['blind'].astype(bool).sum())} blind, "
             f"{int(df['chirps_hidden'].astype(bool).sum())} with chirps hidden"]

    def stats(sel: pd.DataFrame, name: str) -> None:
        for end in ("onset", "offset"):
            d = sel[f"d_{end}_s"].astype(float).dropna().to_numpy()
            if len(d):
                q = np.percentile(d, [25, 50, 75])
                lines.append(f"  {name} {end}: n={len(d)}, yours - BORIS median {q[1]:+.2f} s "
                             f"[IQR {q[0]:+.2f}, {q[2]:+.2f}], later in {np.mean(d > 0):.0%}, earlier in {np.mean(d < 0):.0%}")

    stats(chase, "all  ")
    stats(chase[chase["blind"].astype(bool)], "blind")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--root", type=Path, default=SESSIONS)
    ap.add_argument("--out", type=Path, help=f"output directory (default {OUT}/NAME)")
    args = ap.parse_args()
    ses_dir = args.root / args.name
    session = json.loads((ses_dir / "session.json").read_text())
    df = revision_table(session, latest(ses_dir / "revisions.jsonl"))
    out = args.out or OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "revisions.csv", index=False)
    print(f"{out / 'revisions.csv'}: {len(df)} chases")
    print(summary(df))


if __name__ == "__main__":
    main()
