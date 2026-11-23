"""Tables of the reviewed chases: BORIS vs your onset / offset (the latest save of each review).

    paper/chase_review/export_revisions.py NAME [--root DIR] [--out DIR]

Reads ``session.json`` and ``revisions.jsonl`` of session NAME and writes, to
``paper/output/chase_review/NAME/`` (derived data stays in the repository's git-ignored output):

- ``revisions.csv``: reviews of one BORIS chase (``mode="target"``, BORIS markers shown, and the
  first blind reviews), one row per chase;
- ``window_matches.csv``: blind reviews of a whole window (``mode="window"``: every chase the
  reviewer saw), matched to the BORIS chases in that window by overlap (:func:`match_window`),
  one row per BORIS chase in the window and one per reviewer chase without a BORIS chase.

and prints a summary: the median shift (yours minus BORIS, recording seconds) of onsets and
offsets, with a bootstrap 95 % interval of the median.
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
WINDOW_COLUMNS = ["window_of", "recording", "status", "chase_id", "boris_in_window", "your_index",
                  "boris_onset_frame", "boris_offset_frame", "boris_onset_rec_s", "boris_offset_rec_s",
                  "onset_frame", "offset_frame", "onset_rec_s", "offset_rec_s",
                  "onset_censored", "offset_censored", "d_onset_s", "d_offset_s",
                  "onset_unsure", "offset_unsure", "in_fig6", "blind", "chirps_hidden", "reviewer", "saved_at"]


def latest(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rev = json.loads(line)
                out[rev["chase_id"]] = rev
    return out


def in_fig6(chase: dict, session: dict) -> bool:
    lo, hi = session["meta"]["active_phase_s"]
    return bool(lo <= chase["onset_rec_s"] and chase["offset_rec_s"] < hi)


def revision_table(session: dict, revisions: dict[str, dict]) -> pd.DataFrame:
    """Reviews of one BORIS chase (``mode`` "target" or absent), one row per chase."""
    chases = {c["id"]: c for c in session["chases"]}
    rows = []
    for cid, r in revisions.items():
        if r.get("mode", "target") != "target":
            continue
        c = chases[cid]
        row = {k: r.get(k) for k in COLUMNS if k in r}
        row["in_fig6"] = in_fig6(c, session)
        row["shares_offset"] = bool(c["shares_offset_with"])
        if r.get("onset_frame") is not None:
            row["d_onset_frames"] = r["onset_frame"] - c["onset_frame"]
            row["d_offset_frames"] = r["offset_frame"] - c["offset_frame"]
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["recording", "k"]).reset_index(drop=True)


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    """Frames two intervals [a0, a1], [b0, b1] share."""
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def match_window(rev: dict, session: dict) -> list[dict]:
    """Rows of one window review: the BORIS chases in the window against the reviewer's chases.

    A BORIS chase and a reviewer chase belong together when they overlap. Status per BORIS chase:
    ``matched`` (one reviewer chase overlaps it and that one overlaps no other BORIS chase),
    ``merged`` (its reviewer chase spans other BORIS chases too), ``split`` (several reviewer
    chases overlap it), ``boris_only`` (none). Reviewer chases that overlap no BORIS chase:
    ``yours_only``. Shifts (yours - BORIS) only for matched pairs, and only for ends that are
    inside the window on both sides: a reviewer end on the window's edge is censored (the chase
    ran past it), and so is a BORIS end outside the window.
    """
    w0, wn = rev["window"]["frame0"], rev["window"]["n"]
    w1 = w0 + wn - 1
    boris = [c for c in session["chases"] if c["recording"] == rev["recording"]
             and c["offset_frame"] >= w0 and c["onset_frame"] <= w1]
    yours = rev.get("chases") or []
    ov = np.array([[overlap(b["onset_frame"], b["offset_frame"], y["onset_frame"], y["offset_frame"]) for y in yours]
                   for b in boris], dtype=int).reshape(len(boris), len(yours))
    common = {"window_of": rev["chase_id"], "recording": rev["recording"], "blind": rev.get("blind"),
              "chirps_hidden": rev.get("chirps_hidden"), "reviewer": rev.get("reviewer"), "saved_at": rev.get("saved_at")}
    rows = []
    for i, b in enumerate(boris):
        js = np.flatnonzero(ov[i])
        row = {**common, "chase_id": b["id"], "boris_in_window": bool(b["onset_frame"] >= w0 and b["offset_frame"] <= w1),
               "boris_onset_frame": b["onset_frame"], "boris_offset_frame": b["offset_frame"],
               "boris_onset_rec_s": b["onset_rec_s"], "boris_offset_rec_s": b["offset_rec_s"], "in_fig6": in_fig6(b, session)}
        if len(js) == 0:
            row["status"] = "boris_only"
        elif len(js) > 1:
            row["status"] = "split"
        elif np.count_nonzero(ov[:, js[0]]) > 1:
            row["status"] = "merged"
        else:
            row["status"] = "matched"
        if len(js) == 1:
            y = yours[js[0]]
            row.update({"your_index": int(js[0]), "onset_frame": y["onset_frame"], "offset_frame": y["offset_frame"],
                        "onset_rec_s": y["onset_rec_s"], "offset_rec_s": y["offset_rec_s"],
                        "onset_unsure": y.get("onset_unsure"), "offset_unsure": y.get("offset_unsure"),
                        "onset_censored": bool(y.get("starts_before_window") or b["onset_frame"] < w0),
                        "offset_censored": bool(y.get("ends_after_window") or b["offset_frame"] > w1)})
            if row["status"] == "matched":
                if not row["onset_censored"]:
                    row["d_onset_s"] = y["onset_rec_s"] - b["onset_rec_s"]
                if not row["offset_censored"]:
                    row["d_offset_s"] = y["offset_rec_s"] - b["offset_rec_s"]
        rows.append(row)
    for j, y in enumerate(yours):
        if not ov[:, j].any():
            rows.append({**common, "status": "yours_only", "your_index": j, "onset_frame": y["onset_frame"],
                         "offset_frame": y["offset_frame"], "onset_rec_s": y["onset_rec_s"], "offset_rec_s": y["offset_rec_s"],
                         "onset_unsure": y.get("onset_unsure"), "offset_unsure": y.get("offset_unsure"),
                         "onset_censored": bool(y.get("starts_before_window")),
                         "offset_censored": bool(y.get("ends_after_window"))})
    return rows


def window_table(session: dict, revisions: dict[str, dict]) -> pd.DataFrame:
    """All window reviews matched to BORIS; a BORIS chase seen in several windows counts once
    (the window opened for it, else the latest)."""
    rows = []
    for r in revisions.values():
        if r.get("mode") == "window":
            rows += match_window(r, session)
    df = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    if df.empty:
        return df
    df["_own"] = df["chase_id"] == df["window_of"]
    df = df.sort_values(["_own", "saved_at"], ascending=False)
    has_id = df["chase_id"].notna()
    df = pd.concat([df[has_id].drop_duplicates("chase_id"), df[~has_id]]).drop(columns="_own")
    return df.sort_values(["recording", "boris_onset_frame", "onset_frame"]).reset_index(drop=True)


def describe(d: np.ndarray, name: str) -> str:
    d = d[np.isfinite(d)]
    if not len(d):
        return f"  {name}: none"
    q = np.percentile(d, [25, 50, 75])
    bm = np.median(np.random.default_rng(0).choice(d, (10000, len(d))), axis=1)
    return (f"  {name}: n={len(d)}, yours - BORIS median {q[1]:+.2f} s (95% CI {np.percentile(bm, 2.5):+.2f}, "
            f"{np.percentile(bm, 97.5):+.2f}) [IQR {q[0]:+.2f}, {q[2]:+.2f}], earlier in {np.sum(d < 0)}, later in {np.sum(d > 0)}")


def summary(df: pd.DataFrame, wf: pd.DataFrame | None = None) -> str:
    lines = []
    if df.empty:
        lines.append("one-chase reviews: none")
    else:
        chase = df[~df["not_a_chase"].astype(bool)]
        kept = chase[(chase["d_onset_frames"] == 0) & (chase["d_offset_frames"] == 0)]
        lines.append(f"one-chase reviews: {len(df)}: {len(kept)} kept as in BORIS, {len(chase) - len(kept)} moved, "
                     f"{int(df['not_a_chase'].astype(bool).sum())} not a chase; {int(df['blind'].astype(bool).sum())} blind, "
                     f"{int(df['chirps_hidden'].astype(bool).sum())} with chirps hidden")
        for sel, name in ((chase, "all  "), (chase[chase["blind"].astype(bool)], "blind")):
            for end in ("onset", "offset"):
                lines.append(describe(sel[f"d_{end}_s"].astype(float).to_numpy(), f"{name} {end}"))
    if wf is not None and not wf.empty:
        b = wf[wf["chase_id"].notna()]
        counts = b["status"].value_counts().to_dict()
        lines.append(f"window reviews: {wf['window_of'].nunique()} windows, {len(b)} BORIS chases "
                     f"({int(b['boris_in_window'].astype(bool).sum())} wholly inside a window): "
                     + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                     + f"; your chases without a BORIS chase: {int((wf['status'] == 'yours_only').sum())}")
        m = b[(b["status"] == "matched") & b["blind"].astype(bool)]
        for end in ("onset", "offset"):
            lines.append(describe(m[f"d_{end}_s"].astype(float).to_numpy(), f"blind window, matched {end}"))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--root", type=Path, default=SESSIONS)
    ap.add_argument("--out", type=Path, help=f"output directory (default {OUT}/NAME)")
    args = ap.parse_args()
    ses_dir = args.root / args.name
    session = json.loads((ses_dir / "session.json").read_text())
    revisions = latest(ses_dir / "revisions.jsonl")
    df = revision_table(session, revisions)
    wf = window_table(session, revisions)
    out = args.out or OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "revisions.csv", index=False)
    wf.to_csv(out / "window_matches.csv", index=False)
    print(f"{out / 'revisions.csv'}: {len(df)} chases; {out / 'window_matches.csv'}: {len(wf)} rows")
    print(summary(df, wf))


if __name__ == "__main__":
    main()
