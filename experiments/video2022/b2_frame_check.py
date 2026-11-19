"""Are SLEAP's frame numbers the frames of ``*converted.mp4``, and is seeking exact?

Two checks per recording, from one sequential decoding pass of the video
(``frames.read_sequential``, exact by construction):

1. **Seek**: 12 frames spread over the whole video, read again by seeking
   (``frames.read_seek``); identical images or not.
2. **SLEAP offset**: 200 confident instances (instance, head, centre and tail
   scores >= 0.5) of fish moving at least 0.15 body lengths per frame, spread
   over the video. For offsets k = -3 .. +3 frames, the median grey value
   (3x3 px) under the three keypoints in video frame ``f + k``. The fish are
   dark on a bright floor, so the offset at which the keypoints are darkest
   is the one at which SLEAP's frame ``f`` shows the fish (0 if the numbering
   agrees). Reported per third of the video, to see drift.

    .venv/bin/python experiments/video2022/b2_frame_check.py 2022-05-10-10_00 2022-06-20-10_00 ...

Output: ``$CHIRP_DATA/video2022/b2/frame_check/<rec>_{seek,offset}.csv`` and
``summary.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames as fr  # noqa: E402
import sleap_nix  # noqa: E402
from paths import DATA  # noqa: E402
from recordings2022 import list_recordings  # noqa: E402

OUT = DATA / "video2022" / "b2" / "frame_check"
OFFSETS = np.arange(-3, 4)
NODES = ("head", "center", "tail")
N_INSTANCES = 200
N_SEEK = 12
MIN_SPEED_BL = 0.15
SEED = 20261119


def moving_instances(df: pd.DataFrame, body_px: float, n_frames: int, rng: np.random.Generator) -> pd.DataFrame:
    """Confident, fast-moving instances, spread over the video (one per equal-length slice)."""
    ok = df["instance_score"] >= 0.5
    for node in NODES:
        ok &= df[f"{node}_score"] >= 0.5
    d = df[ok].sort_values(["track", "frame"]).copy()
    same = (d["track"].shift(-1) == d["track"]) & (d["frame"].shift(-1) == d["frame"] + 1)
    step = np.hypot(d["centroid_x"].shift(-1) - d["centroid_x"], d["centroid_y"].shift(-1) - d["centroid_y"])
    d = d[same & (step >= MIN_SPEED_BL * body_px)]
    d = d[(d["frame"] >= 3) & (d["frame"] < n_frames - 4)]
    slices = np.linspace(0, n_frames, N_INSTANCES + 1)
    picks = []
    for a, b in zip(slices[:-1], slices[1:], strict=True):
        g = d[(d["frame"] >= a) & (d["frame"] < b)]
        if len(g):
            picks.append(g.iloc[rng.integers(len(g))])
    return pd.DataFrame(picks)


def grey_at(img: np.ndarray, x: float, y: float) -> float:
    xi, yi = round(x), round(y)
    h, w = img.shape
    patch = img[max(0, yi - 1):min(h, yi + 2), max(0, xi - 1):min(w, xi + 2)]
    return float(np.median(patch))


def check(rec_name: str, max_frame: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = next(r for r in list_recordings() if r.name == rec_name)
    info = sleap_nix.read_video_info(rec.nix)
    n_frames = info.n_frames if max_frame is None else min(info.n_frames, max_frame)
    df = sleap_nix.read_instances(rec.nix)
    rng = np.random.default_rng(SEED)
    ok = (df["instance_score"] >= 0.5) & (df["head_score"] >= 0.5) & (df["tail_score"] >= 0.5)
    body = float(np.median(np.hypot(df.loc[ok, "head_x"] - df.loc[ok, "tail_x"],
                                    df.loc[ok, "head_y"] - df.loc[ok, "tail_y"])))
    inst = moving_instances(df, body, n_frames, rng)
    need: dict[int, list[tuple[int, int]]] = {}
    for i, f in enumerate(inst["frame"].astype(int)):
        for k in OFFSETS:
            need.setdefault(f + int(k), []).append((i, int(k)))
    seek_frames = np.linspace(100, n_frames - 200, N_SEEK).astype(int)
    grey = np.full((len(inst), len(OFFSETS)), np.nan)
    kept = {}
    xy = inst[[f"{n}_{a}" for n in NODES for a in "xy"]].to_numpy().reshape(len(inst), len(NODES), 2)
    for f, img in fr.read_sequential(rec.video, [*need, *seek_frames]):
        for i, k in need.get(f, []):
            grey[i, k - OFFSETS[0]] = np.mean([grey_at(img, *xy[i, j]) for j in range(len(NODES))])
        if f in seek_frames:
            kept[f] = img
    seek = pd.DataFrame([{"recording": rec_name, "frame": f, "identical": bool(np.array_equal(img, kept[f])),
                          "mean_abs_diff": float(np.abs(img.astype(int) - kept[f].astype(int)).mean())}
                         for f, img in fr.read_seek(rec.video, seek_frames)])
    off = inst[["frame", "track", "instance_score"]].reset_index(drop=True).copy()
    off.insert(0, "recording", rec_name)
    for j, k in enumerate(OFFSETS):
        off[f"grey_{k:+d}"] = grey[:, j]
    off["best_offset"] = OFFSETS[np.nanargmin(grey, axis=1)]
    off["third"] = np.minimum(2, (3 * off["frame"] / n_frames).astype(int))
    return seek, off


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--max-frame", type=int, default=None, help="only the first frames (quick test)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.recordings:
        seek, off = check(name, args.max_frame)
        seek.to_csv(args.out / f"{name}_seek.csv", index=False)
        off.to_csv(args.out / f"{name}_offset.csv", index=False)
        for third, g in off.groupby("third"):
            counts = g["best_offset"].value_counts()
            rows.append({"recording": name, "third": int(third), "n": len(g),
                         "seek_identical": f"{int(seek['identical'].sum())}/{len(seek)}",
                         **{f"best_{k:+d}": int(counts.get(k, 0)) for k in OFFSETS},
                         **{f"median_grey_{k:+d}": float(g[f"grey_{k:+d}"].median()) for k in OFFSETS}})
        print(f"{name}: seek identical {int(seek['identical'].sum())}/{len(seek)}; best offsets "
              f"{off['best_offset'].value_counts().sort_index().to_dict()}", flush=True)
    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
