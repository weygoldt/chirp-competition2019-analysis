"""Frames with a pose model's predictions drawn on them, for looking at the model (dark phase, both fish).

Per recording (all 19), candidate frames from the dark phase (recording time
0-3 h): frames where SLEAP had two instances within 2 body lengths (close fish,
where chases happen) and random frames. The model runs on them; frames with
exactly two fish of box confidence >= 0.5 are kept and drawn by the distance
between the two fish's spine1 points: close (< 1.5 body lengths), medium
(1.5-4), far (> 4), ``--per-recording`` frames per recording (about 40 % close,
40 % medium, 20 % far), at least 30 s apart.

Each image is a crop around both fish (at least 700 px, 150 px margin) at full
resolution, with the predicted skeleton (snout red, spine1 yellow, spine2 / 3
green, tail blue, fins pink / brown; a fin only if its confidence >= 0.3),
fish number, box confidence and, in the corner, recording, frame, recording
time and distance.

    .venv/bin/python experiments/video2022/b3_review_frames.py --model weights/pose7_r2_l.pt [--per-recording 6]

Output: ``$CHIRP_DATA/video2022/review/<model stem>/`` (``<rec>_<frame>.jpg``, ``index.csv``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1  # noqa: E402
import frames as fr  # noqa: E402
import pose_labels as pl  # noqa: E402
import pose_model as pm  # noqa: E402
from paths import DATA  # noqa: E402
from recordings2022 import list_recordings  # noqa: E402

OUT = DATA / "video2022" / "review"
BGR = {"snout": (77, 77, 255), "spine1": (77, 210, 255), "spine2": (180, 233, 199), "spine3": (187, 205, 127),
       "tail": (255, 195, 77), "left_fin": (191, 129, 247), "right_fin": (40, 86, 166)}
FIN_CONF = 0.3
SEED = 20261119


def draw(img: np.ndarray, preds: list[pm.PredFish], caption: str) -> np.ndarray:
    import cv2

    pts_all = np.array([p.points[n] for p in preds for n in ("snout", "spine1", "tail")])
    c = pts_all.mean(axis=0)
    half = max(350, np.ptp(pts_all, axis=0).max() / 2 + 150)
    h, w = img.shape
    x0, x1 = int(max(0, c[0] - half)), int(min(w, c[0] + half))
    y0, y1 = int(max(0, c[1] - half)), int(min(h, c[1] + half))
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for k, p in enumerate(preds, 1):
        shown = {n: xy for n, xy in p.points.items() if n not in pm.FINS or p.kpt_conf[n] >= FIN_CONF}
        for a, b in pl.EDGES:
            if a in shown and b in shown:
                cv2.line(out, tuple(map(int, shown[a])), tuple(map(int, shown[b])), (255, 255, 255), 2, cv2.LINE_AA)
        for n, xy in shown.items():
            cv2.circle(out, tuple(map(int, xy)), 6, BGR[n], -1, cv2.LINE_AA)
            cv2.circle(out, tuple(map(int, xy)), 6, (0, 0, 0), 1, cv2.LINE_AA)
        sx, sy = map(int, p.points["snout"])
        cv2.putText(out, f"{k} ({p.box_conf:.2f})", (sx + 10, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3,
                    cv2.LINE_AA)
        cv2.putText(out, f"{k} ({p.box_conf:.2f})", (sx + 10, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1,
                    cv2.LINE_AA)
    crop = out[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(crop, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-recording", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=40, help="per recording, half SLEAP close pairs, half random")
    args = ap.parse_args()
    weights = Path(args.model) if Path(args.model).is_absolute() else DATA / "video2022" / args.model
    out = OUT / weights.stem
    out.mkdir(parents=True, exist_ok=True)
    model = pm.load(weights)
    rng = np.random.default_rng(SEED)
    recs = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")
    rows = []
    for rec in list_recordings():
        bl = float(recs.loc[rec.name, "body_length_px"])
        t = np.load(b1.TABLES / "frame_time" / f"{rec.name}.npy")
        dark = np.flatnonzero((t >= 0) & (t < b1.LIGHT_ON_S))
        inst = pd.read_parquet(b1.TABLES / "instances" / f"{rec.name}.parquet",
                               columns=["frame", "n_in_frame", "pair_dist_px", "phase"])
        close = np.unique(inst.loc[(inst["phase"] == 1) & (inst["n_in_frame"] == 2)
                                   & (inst["pair_dist_px"] < 2 * bl), "frame"].to_numpy())
        n = args.candidates // 2
        cand = np.concatenate([rng.choice(close, min(n, len(close)), replace=False) if len(close) else [],
                               rng.choice(dark, n, replace=False)]).astype(int)
        imgs = dict(fr.read_seek(rec.video, cand))
        keys = sorted(imgs)
        found = []
        for f, preds in zip(keys, pm.predict(model, [imgs[f] for f in keys], batch=8), strict=True):
            two = [p for p in preds if p.box_conf >= 0.5]
            if len(two) != 2:
                continue
            d = float(np.hypot(*np.subtract(two[0].points["spine1"], two[1].points["spine1"]))) / bl
            found.append((f, two, d))
        quota = {"close": round(0.4 * args.per_recording), "medium": round(0.4 * args.per_recording)}
        quota["far"] = args.per_recording - quota["close"] - quota["medium"]
        taken: list[tuple[int, list, float, str]] = []
        for f, two, d in [found[i] for i in rng.permutation(len(found))]:
            cls = "close" if d < 1.5 else ("medium" if d < 4 else "far")
            if quota[cls] <= 0 or any(abs(t[f] - t[g]) < 30 for g, *_ in taken):
                continue
            quota[cls] -= 1
            taken.append((f, two, d, cls))
        for f, two, d, cls in taken:
            caption = f"{rec.name}  frame {f}  t = {t[f] / 60:.1f} min  distance {d:.1f} BL ({cls})"
            name = f"{rec.name}_{f:06d}.jpg"
            cv2.imwrite(str(out / name), draw(imgs[f], two, caption), [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"file": name, "recording": rec.name, "frame": f, "t_s": float(t[f]), "distance_bl": d,
                         "class": cls, "box_conf_1": two[0].box_conf, "box_conf_2": two[1].box_conf,
                         **{f"conf_{n}": float(np.mean([p.kpt_conf[n] for p in two])) for n in pl.SKELETON}})
        print(f"{rec.name}: {len(found)} of {len(keys)} candidates with two fish, {len(taken)} drawn", flush=True)
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "index.csv", index=False)
    print(idx["class"].value_counts().to_dict(), f"{len(idx)} frames -> {out}")
    print("mean point confidence:", idx[[f"conf_{n}" for n in pl.SKELETON]].mean().round(2).to_dict())


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
