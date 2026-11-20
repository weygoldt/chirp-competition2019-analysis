"""Two-fish trajectories (winner, loser) from the pose tracks and the electrode identities (author's go, 2026-11-20).

The pose tracks (``c1_chases.py``: confident full-length detections, anonymous
``track``) swap the fish when they touch; the electrode identities (branch
``link2022``, ``$CHIRP_DATA/video2022/identity/<run>/<rec>/observations_labelled.parquet``:
per detection ``label`` winner / loser / unknown, ``p_winner``) cut the tracks at
those swaps and name each piece. Here every detection takes its label; unknown ones
are dropped; if two detections of one frame carry the same label, the one with the
stronger evidence (``|p_winner - 0.5|``) is kept (counted). Each fish's detections
are cut into *runs* at gaps > ``MAX_GAP_S`` and their kinematics computed per run
(``chases.kinematics``: spine1 smoothed over 0.4 s, velocity, speed in BL/s).

    .venv/bin/python experiments/video2022/identity_tracks.py poses_r3

Output: ``$CHIRP_DATA/video2022/fish2/<run>/<rec>_detections.parquet`` (frame, t_s,
identity, p_winner, track, keypoints), ``<rec>_kinematics.parquet`` (identity, run, frame,
i, t_s, x, y, vx, vy, speed, sx, sy) and ``coverage.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1
import b3_predict as bp
import c1_chases as c1
import chases as ch
from paths import DATA

IDENT = DATA / "video2022" / "identity"
OUT = DATA / "video2022" / "fish2"
MAX_GAP_S = 0.5
FISH = ("winner", "loser")


def labelled(tr: pd.DataFrame, obs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The tracked detections with their identity (winner / loser only), at most one per identity and frame;
    and how many same-label duplicates were dropped."""
    d = tr.merge(obs[["frame", "track", "label", "p_winner"]], on=["frame", "track"], how="inner")
    d = d[d["label"].isin(FISH)].rename(columns={"label": "identity"})
    d = d.assign(strength=(d["p_winner"] - 0.5).abs()).sort_values(["frame", "identity", "strength"],
                                                                  ascending=[True, True, False])
    n = len(d)
    d = d.drop_duplicates(["frame", "identity"]).drop(columns="strength").reset_index(drop=True)
    return d, n - len(d)


def kinematics(d: pd.DataFrame, bl: float, stride: int, fps: float = c1.FPS) -> pd.DataFrame:
    """Per identity, runs of detections without gaps > MAX_GAP_S, and their kinematics (``chases.kinematics``)."""
    out = []
    gap = round(MAX_GAP_S * fps)
    for k, ident in enumerate(FISH):
        g = d[d["identity"] == ident].sort_values("frame")
        if g.empty:
            continue
        run = np.r_[0, np.cumsum(np.diff(g["frame"].to_numpy()) > gap)]
        g = g.assign(track=run * 2 + k)  # one pseudo track per run
        kin = ch.kinematics(g, bl, stride, fps)
        out.append(kin.assign(identity=ident, run=kin["track"] // 2))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def on_grid(kin: pd.DataFrame, ident: str, i_lo: int, i_hi: int) -> pd.DataFrame:
    """One fish's kinematics on the grid i_lo..i_hi (NaN where it is not seen)."""
    k = kin[(kin["identity"] == ident) & kin["i"].between(i_lo, i_hi)].drop_duplicates("i").set_index("i")
    return k.reindex(np.arange(i_lo, i_hi + 1))


def load(run: str, rec: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / run / f"{rec}_kinematics.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--identity-run", default=None, help="default: the same name as the pose run")
    args = ap.parse_args()
    chase_root = c1.OUT / args.run
    ident_root = IDENT / (args.identity_run or args.run)
    stride = int(json.loads((chase_root / "params.json").read_text())["stride"])
    blpx = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    out = OUT / args.run
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec_dir in sorted(p for p in ident_root.iterdir() if (p / "observations_labelled.parquet").exists()):
        rec = rec_dir.name
        tr = pd.read_parquet(chase_root / rec / "tracks.parquet")
        obs = pd.read_parquet(rec_dir / "observations_labelled.parquet", columns=["frame", "track", "label",
                                                                                 "p_winner"])
        d, dup = labelled(tr, obs)
        kin = kinematics(d, float(blpx[rec]), stride)
        d.to_parquet(out / f"{rec}_detections.parquet", index=False)
        kin.to_parquet(out / f"{rec}_kinematics.parquet", index=False)
        n = len(bp.dark_frames(rec, stride)[0])
        per = d.groupby("frame")["identity"].nunique()
        seen = {f: set(d.loc[d["identity"] == f, "frame"]) for f in FISH}
        rows.append({"recording": rec, "frames": n, "detections": len(tr), "labelled": len(d),
                     "duplicates_dropped": dup, "share_winner_seen": len(seen["winner"]) / n,
                     "share_loser_seen": len(seen["loser"]) / n, "share_both_seen": float((per == 2).sum() / n),
                     "runs_winner": int(kin.loc[kin["identity"] == "winner", "run"].nunique()),
                     "runs_loser": int(kin.loc[kin["identity"] == "loser", "run"].nunique())})
        print(f"{rec}: {len(d)} of {len(tr)} detections labelled, {dup} duplicates dropped, both fish seen in "
              f"{rows[-1]['share_both_seen']:.1%} of dark frames", flush=True)
    pd.DataFrame(rows).to_csv(out / "coverage.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
