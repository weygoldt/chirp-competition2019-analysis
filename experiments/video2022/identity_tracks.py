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
from dataclasses import dataclass
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


KP = ("snout_x", "snout_y", "spine1_x", "spine1_y", "tail_x", "tail_y")


@dataclass(frozen=True)
class Fill:
    """Gap filling (author, 2026-11-23: offsets fail where the fish touch and one is lost)."""

    max_fill_s: float = 2.0  # gaps up to this long are searched for unused detections
    gate_bl: float = 0.75  # a detection must lie this close to the fish's interpolated path
    other_bl: float = 0.3  # ... and not this close to the other fish
    max_interp_s: float = 1.0  # gaps up to this long that remain are interpolated


def _runs(missing: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of each run of True that has a False on both sides."""
    out, start = [], None
    for i, m in enumerate(missing):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if start > 0:
                out.append((start, i - 1))
            start = None
    return out


def fill_gaps(lab: pd.DataFrame, raw: pd.DataFrame, frames: np.ndarray, times: np.ndarray, bl: float,
              f: Fill = Fill(), dt: float = 0.08) -> pd.DataFrame:  # noqa: B008 - frozen
    """The labelled detections plus, inside short gaps of a fish, unused detections near its path ('filled')
    and linear interpolation ('interp'); column ``source``. ``raw``: all pose detections (frame, det, keypoints)."""
    used = set(zip(lab["frame"], lab["det"], strict=True))
    cand = raw[[(a, b) not in used for a, b in zip(raw["frame"], raw["det"], strict=True)]]
    by_frame = {fr: g for fr, g in cand.groupby("frame")}
    pos = {k: lab[lab["identity"] == k].drop_duplicates("frame").set_index("frame")[list(KP)].astype(float).reindex(frames)
           for k in FISH}
    src = {k: np.where(pos[k]["spine1_x"].notna(), "labelled", "") for k in FISH}
    max_fill = round(f.max_fill_s / dt)
    for k in FISH:
        other = pos[FISH[1] if k == FISH[0] else FISH[0]][["spine1_x", "spine1_y"]].to_numpy(float)
        p = pos[k]
        for a, b in _runs(p["spine1_x"].isna().to_numpy()):
            if b + 1 >= len(frames) or b - a + 1 > max_fill:
                continue
            p0 = p.iloc[a - 1][["spine1_x", "spine1_y"]].to_numpy(float)
            p1 = p.iloc[b + 1][["spine1_x", "spine1_y"]].to_numpy(float)
            for j in range(a, b + 1):
                g = by_frame.get(frames[j])
                if g is None:
                    continue
                pred = p0 + (p1 - p0) * (j - a + 1) / (b - a + 2)
                xy = g[["spine1_x", "spine1_y"]].to_numpy(float)
                dist = np.hypot(*(xy - pred).T)
                ok = dist < f.gate_bl * bl
                ok &= [(frames[j], d) not in used for d in g["det"]]
                if np.isfinite(other[j]).all():
                    ok &= np.hypot(*(xy - other[j]).T) > f.other_bl * bl
                if not ok.any():
                    continue
                best = np.flatnonzero(ok)[np.argmin(dist[ok])]
                p.iloc[j] = g.iloc[best][list(KP)].to_numpy(float)
                src[k][j] = "filled"
                used.add((frames[j], g.iloc[best]["det"]))
        max_interp = round(f.max_interp_s / dt)
        for a, b in _runs(p["spine1_x"].isna().to_numpy()):
            if b + 1 >= len(frames) or b - a + 1 > max_interp:
                continue
            w = (np.arange(a, b + 1) - a + 1) / (b - a + 2)
            v0, v1 = p.iloc[a - 1].to_numpy(float), p.iloc[b + 1].to_numpy(float)
            p.iloc[a:b + 1] = v0[None, :] + w[:, None] * (v1 - v0)[None, :]
            src[k][a:b + 1] = "interp"
    out = []
    for k in FISH:
        ok = pos[k]["spine1_x"].notna().to_numpy()
        out.append(pos[k][ok].assign(identity=k, source=src[k][ok], t_s=times[ok]).reset_index(names="frame"))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--identity-run", default=None, help="default: the same name as the pose run")
    ap.add_argument("--fill", action="store_true", help="fill short gaps (fill_gaps); output <run>_filled")
    args = ap.parse_args()
    chase_root = c1.OUT / args.run
    ident_root = IDENT / (args.identity_run or args.run)
    stride = int(json.loads((chase_root / "params.json").read_text())["stride"])
    blpx = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    out = OUT / (args.run + ("_filled" if args.fill else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec_dir in sorted(p for p in ident_root.iterdir() if (p / "observations_labelled.parquet").exists()):
        rec = rec_dir.name
        tr = pd.read_parquet(chase_root / rec / "tracks.parquet")
        obs = pd.read_parquet(rec_dir / "observations_labelled.parquet", columns=["frame", "track", "label",
                                                                                 "p_winner"])
        d, dup = labelled(tr, obs)
        if args.fill:
            frames, times = bp.dark_frames(rec, stride)
            n_lab = len(d)
            d = fill_gaps(d, bp.load(args.run, rec), frames, times, float(blpx[rec]))
            print(f"{rec}: filled {int((d['source'] == 'filled').sum())} and interpolated "
                  f"{int((d['source'] == 'interp').sum())} fish-frames (labelled {n_lab})", flush=True)
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
        print(f"{rec}: {len(d)} fish-frames ({len(tr)} tracked detections), {dup} duplicates dropped, both fish seen in "
              f"{rows[-1]['share_both_seen']:.1%} of dark frames", flush=True)
    pd.DataFrame(rows).to_csv(out / "coverage.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
