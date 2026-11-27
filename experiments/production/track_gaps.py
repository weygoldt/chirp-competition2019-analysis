"""Where and how often the wavetracker tracks of the trial fish have gaps, and what that does to chirp assignment.

The v2 pipeline assigns a chirp to the nearest *live* track (chirpdetector
`pipeline/assign.py`): a track is live at the chirp's onset only between its
first and last detection and not inside a gap longer than ``max_gap_s``
(10 s), and only within ``tolerance_hz`` (20 Hz) of the predicted emitter
EODf. A chirp of fish X inside a gap of X's track is therefore left
unassigned, or given to the other fish if that fish lies within 20 Hz.

For every trial in ``trial_summary.csv`` and both trial fish (``win_ID``,
``lose_ID``) this script lists the track's gaps longer than ``max_gap_s``
(interior gaps and the stretches before the first / after the last
detection), the EODf just before and after each interior gap, and the chirps
(``chirp`` rows of ``chirps_rtdetr.parquet``) whose onset falls into a gap:
how they were assigned, and whether the track interpolated linearly across
the gap would lie within ``tolerance_hz`` of the chirp's ``emitter_hz`` (and
be the nearest fish).

Writes to OUT (default: the run directory's ``track_gaps/``):
``gaps.csv`` (one row per gap) and ``gap_chirps.csv`` (one row per chirp in a
gap; a chirp inside gaps of both fish appears twice). Findings of the
2026-11-14 run: ``README.md`` next to the outputs.

    python experiments/production/track_gaps.py [--data $CHIRP_REC_2019] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2019  # noqa: E402

MAX_GAP_S = 10.0
TOLERANCE_HZ = 20.0
ACTIVE_S = 3 * 3600
EDGE_S = 1.0  # baseline: median EODf over this long next to the gap
BAD_RECORDINGS = ["2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00"]


def track(rec_dir: Path, fish_id: float) -> tuple[np.ndarray, np.ndarray]:
    ident = np.load(rec_dir / "ident_v.npy")
    sel = ident == fish_id
    t = np.load(rec_dir / "times.npy")[np.load(rec_dir / "idx_v.npy")[sel].astype(int)]
    f = np.load(rec_dir / "fund_v.npy")[sel]
    order = np.argsort(t)
    return t[order], f[order]


def edge_freq(t: np.ndarray, f: np.ndarray, at: float, side: str) -> float:
    sel = (t >= at - EDGE_S) & (t <= at) if side == "before" else (t >= at) & (t <= at + EDGE_S)
    return float(np.nanmedian(f[sel])) if sel.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=REC_2019)
    ap.add_argument("--out", type=Path, default=DATA / "detections" / "competition2019_v2_20261113" / "track_gaps")
    ap.add_argument(
        "--suffix",
        default="",
        help="read the copies chirps_rtdetr<suffix>.parquet / _meta.json of the recording "
        "directories (default: the v2 production run's; _sw: the single-window run's, with "
        "--out $CHIRP_DATA/detections/competition2019_v3/track_gaps)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(args.data / "trial_summary.csv", index_col=0)
    gaps, gap_chirps = [], []
    for _, row in ts.iterrows():
        rec = row["recording"]
        rec_dir = args.data / rec
        meta = json.loads((rec_dir / f"chirps_rtdetr{args.suffix}_meta.json").read_text())
        duration = float(meta["inputs"]["duration_s"])
        analysed = (row["draw"] == 0) and rec not in BAD_RECORDINGS
        chirps = pd.read_parquet(rec_dir / f"chirps_rtdetr{args.suffix}.parquet")
        chirps = chirps[chirps["chirp"]]
        fish = {"winner": row["win_ID"], "loser": row["lose_ID"]}
        tracks = {k: track(rec_dir, v) for k, v in fish.items()}
        for status, fid in fish.items():
            t, f = tracks[status]
            other = "loser" if status == "winner" else "winner"
            ot, of = tracks[other]
            edges = []
            if len(t) == 0:
                edges.append(("no_track", 0.0, duration, np.nan, np.nan))
            else:
                if t[0] > MAX_GAP_S:
                    edges.append(("start", 0.0, float(t[0]), np.nan, edge_freq(t, f, t[0], "after")))
                d = np.diff(t)
                for i in np.flatnonzero(d > MAX_GAP_S):
                    a, b = float(t[i]), float(t[i + 1])
                    edges.append(("interior", a, b, edge_freq(t, f, a, "before"), edge_freq(t, f, b, "after")))
                if duration - t[-1] > MAX_GAP_S:
                    edges.append(("end", float(t[-1]), duration, edge_freq(t, f, t[-1], "before"), np.nan))
            for kind, a, b, fa, fb in edges:
                in_gap = chirps[(chirps["t_start_s"] > a) & (chirps["t_start_s"] < b)]
                # interpolated baseline of this fish inside an interior gap
                if kind == "interior" and np.isfinite(fa) and np.isfinite(fb):
                    base = fa + (fb - fa) * (in_gap["t_start_s"] - a) / (b - a)
                else:
                    base = pd.Series(np.nan, index=in_gap.index)
                # other fish's baseline at the onsets (nearest sample within MAX_GAP_S)
                if len(ot):
                    j = np.clip(np.searchsorted(ot, in_gap["t_start_s"]), 0, len(ot) - 1)
                    near = np.abs(ot[j] - in_gap["t_start_s"].to_numpy()) <= MAX_GAP_S
                    other_base = np.where(near, of[j], np.nan)
                else:
                    other_base = np.full(len(in_gap), np.nan)
                d_this = np.abs(in_gap["emitter_hz"] - base)
                d_other = np.abs(in_gap["emitter_hz"].to_numpy() - other_base)
                rescuable = (d_this <= TOLERANCE_HZ) & ~(d_other < d_this)
                for k, (_, c) in enumerate(in_gap.iterrows()):
                    gap_chirps.append({
                        "recording": rec, "analysed": analysed, "fish": status, "fish_id": fid,
                        "gap_kind": kind, "gap_start_s": a, "gap_end_s": b,
                        "t_start_s": c["t_start_s"], "in_first_3h": c["t_start_s"] < ACTIVE_S,
                        "emitter_hz": c["emitter_hz"], "assign_status": c["assign_status"],
                        "assigned_to": ("this" if c["track_id"] == fid else
                                        "other" if c["track_id"] == fish[other] else
                                        "none" if pd.isna(c["track_id"]) else "third"),
                        "interp_baseline_hz": base.iloc[k], "other_baseline_hz": other_base[k],
                        "interp_df_hz": d_this.iloc[k], "rescuable_by_interp": bool(rescuable.iloc[k]),
                    })
                gaps.append({
                    "recording": rec, "analysed": analysed, "fish": status, "fish_id": fid,
                    "kind": kind, "start_s": a, "end_s": b, "duration_s": b - a,
                    "overlap_first_3h_s": max(0.0, min(b, ACTIVE_S) - min(a, ACTIVE_S)),
                    "f_before_hz": fa, "f_after_hz": fb, "f_jump_hz": fb - fa,
                    "recording_s": duration, "chirps_in_gap": len(in_gap),
                    "chirps_unassigned": int(in_gap["track_id"].isna().sum()),
                    "chirps_to_other": int((in_gap["track_id"] == fish[other]).sum()),
                    "rescuable_by_interp": int(rescuable.sum()),
                })
    g, c = pd.DataFrame(gaps), pd.DataFrame(gap_chirps)
    g.to_csv(args.out / "gaps.csv", index=False)
    c.to_csv(args.out / "gap_chirps.csv", index=False)
    print(f"wrote {len(g)} gaps, {len(c)} gap chirps to {args.out}")


if __name__ == "__main__":
    main()
