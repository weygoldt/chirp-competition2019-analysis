"""Mine training frames for a fish pose model from confident SLEAP detections (milestone B2).

A mined frame is a video frame in which SLEAP found exactly two fish (the
tank holds two), both confidently and clearly apart, so that the frame is
fully labelled by SLEAP's keypoints:

- each instance: instance score and head, centre and tail scores >= 0.5,
  head-tail distance >= 0.5 body lengths (no collapsed pose);
- the pair: centroids more than 2 body lengths apart and no keypoint shared
  (two instances on one fish); the criteria of B1 (``b1_assess_sleap.pairs_and_mining``)
  plus a confident centre node;
- both fish moving slowly (each instance at most ``--max-step-bl`` body
  lengths from the same track's position one frame earlier). SLEAP's frame
  numbers agree with ``*converted.mp4`` only to within one frame
  (``b2_frame_check.py``), so a slow fish keeps its label on the fish whichever
  frame it is;
- inside the electric recording (dark or light phase).

Spread: per recording at most ``--per-recording`` frames, at least
``--min-gap-s`` apart, drawn round-robin over strata of phase (dark, light;
light frames first, they are rare) x 15-min block of recording time x the
arena cells (4 x 3) of the two fish; random within a stratum (seeded).

Splits are by recording (``SPLITS``): ``test`` recordings are never mined (they
are for hand-labelled evaluation), ``val`` recordings are mined for model
selection, the rest for training.

Output (``$CHIRP_DATA/video2022/datasets/mined_v1/``): Ultralytics YOLO pose
layout, ``images/{train,val}/<rec>_<frame>.jpg`` (grey, JPEG quality 92, full
2048 x 1536), ``labels/{train,val}/<rec>_<frame>.txt`` (``pose_labels.yolo_lines``,
box = keypoints padded by 0.1 body lengths), ``manifest.csv`` (one row per
frame: split, recording, frame, recording time, phase, locations, scores, steps),
``dataset.yaml``, ``summary.csv``.

    .venv/bin/python experiments/video2022/b2_mine.py [--per-recording 150] [--dry-run]
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
from paths import DATA  # noqa: E402
from recordings2022 import list_recordings  # noqa: E402

OUT = DATA / "video2022" / "datasets" / "mined_v1"
NODES = ("head", "center", "tail")  # the SLEAP exports' nodes (= snout, spine1, tail of pose_labels.SKELETON)
SKELETON_3 = ("snout", "spine1", "tail")  # mined_v1 is written with these three keypoints
# test: hand-labelled evaluation only (one recording from each of pairs 1, 6 and 4: early / middle / late);
# val: mined, for model selection (pairs 5 and 3, recordings with many minable frames); everything else: train
SPLITS = {
    "2022-05-11-10_00": "test", "2022-06-14-10_00": "test", "2022-07-01-10_00": "test",
    "2022-06-13-10_00": "val", "2022-06-22-10_00": "val",
}
CONF = 0.5
MIN_HEAD_TAIL_BL = 0.5
MIN_PAIR_BL = 2.0
SHARED_PX = 2.0
PAD_BL = 0.1
BLOCK_S = 900.0
GRID = (4, 3)
JPEG_QUALITY = 92
SEED = 20261119


def split_of(name: str) -> str:
    return SPLITS.get(name, "train")


def candidates(name: str, max_step_bl: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Frames of one recording that pass the criteria (one row per frame) and their two instances."""
    recs = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")
    objs = pd.read_csv(b1.TABLES / "arena_objects.csv")
    bl = float(recs.loc[name, "body_length_px"])
    d = pd.read_parquet(b1.TABLES / "instances" / f"{name}.parquet")
    d = d[d["phase"].isin([1, 2]) & (d["n_in_frame"] == 2)].sort_values(["frame", "track"], kind="stable")
    d["loc"] = b1.classify_location(d["centroid_x"].to_numpy(), d["centroid_y"].to_numpy(),
                                    objs[objs["recording"] == name], bl)
    ht = np.hypot(d["head_x"] - d["tail_x"], d["head_y"] - d["tail_y"]) / bl
    good = (d["instance_score"] >= CONF) & (ht >= MIN_HEAD_TAIL_BL)
    for n in NODES:
        good &= d[f"{n}_score"] >= CONF
    good &= (d["gap_frames"] == 1) & (d["step_px"] <= max_step_bl * bl)
    d = d.assign(good=good.to_numpy())
    a, b = d.iloc[0::2], d.iloc[1::2]
    if not np.array_equal(a["frame"].to_numpy(), b["frame"].to_numpy()):
        raise ValueError(f"{name}: two-instance frames are not in consecutive rows")
    pa = np.stack([a[[f"{n}_x", f"{n}_y"]].to_numpy() for n in NODES], 1)
    pb = np.stack([b[[f"{n}_x", f"{n}_y"]].to_numpy() for n in NODES], 1)
    nearest = np.linalg.norm(pa[:, :, None] - pb[:, None, :], axis=3).min(axis=(1, 2))
    ok = (a["good"].to_numpy() & b["good"].to_numpy() & (a["pair_dist_px"].to_numpy() > MIN_PAIR_BL * bl)
          & (nearest >= SHARED_PX))
    f = pd.DataFrame({
        "recording": name, "frame": a["frame"].to_numpy(), "t_s": a["t_s"].to_numpy(),
        "phase": np.where(a["phase"].to_numpy() == 1, "dark", "light"),
        "loc_a": a["loc"].to_numpy(), "loc_b": b["loc"].to_numpy(),
        "pair_dist_bl": a["pair_dist_px"].to_numpy() / bl,
        "min_instance_score": np.minimum(a["instance_score"].to_numpy(), b["instance_score"].to_numpy()),
        "max_step_bl": np.maximum(a["step_px"].to_numpy(), b["step_px"].to_numpy()) / bl,
        "cell_a": _cell(a, objs, name), "cell_b": _cell(b, objs, name),
    })[ok]
    inst = d[d["frame"].isin(f["frame"])]
    return f.reset_index(drop=True), inst, bl


def _cell(d: pd.DataFrame, objs: pd.DataFrame, name: str) -> np.ndarray:
    a = objs[objs["recording"] == name].iloc[0]
    cx = np.clip(((d["centroid_x"].to_numpy() - a.arena_x0) / (a.arena_x1 - a.arena_x0) * GRID[0]).astype(int), 0, GRID[0] - 1)
    cy = np.clip(((d["centroid_y"].to_numpy() - a.arena_y0) / (a.arena_y1 - a.arena_y0) * GRID[1]).astype(int), 0, GRID[1] - 1)
    return cy * GRID[0] + cx


def spread(f: pd.DataFrame, n: int, min_gap_s: float, rng: np.random.Generator) -> pd.DataFrame:
    """Up to n frames, >= min_gap_s apart, round-robin over strata (light first)."""
    f = f.assign(block=(f["t_s"] // BLOCK_S).astype(int),
                 cells=[tuple(sorted(p)) for p in zip(f["cell_a"], f["cell_b"], strict=True)])
    f = f.iloc[rng.permutation(len(f))]
    strata = [g for _, g in sorted(f.groupby(["phase", "block", "cells"]), key=lambda kv: kv[0][0] != "light")]
    queues = [list(g.index) for g in strata]
    chosen: list[int] = []
    times: list[float] = []
    while len(chosen) < n and any(queues):
        for q in queues:
            while q:
                i = q.pop()
                t = f.at[i, "t_s"]
                if all(abs(t - u) >= min_gap_s for u in times):
                    chosen.append(i)
                    times.append(t)
                    break
            if len(chosen) >= n:
                break
    return f.loc[chosen].drop(columns=["cells"]).sort_values("frame")


def frame_fish(inst: pd.DataFrame, frame: int) -> list[pl.Fish]:
    rows = inst[inst["frame"] == frame]
    return [pl.Fish({pl.LEGACY_3[n]: (float(r[f"{n}_x"]), float(r[f"{n}_y"])) for n in NODES}, origin="sleap")
            for _, r in rows.iterrows()]


def write(name: str, chosen: pd.DataFrame, inst: pd.DataFrame, bl: float, out: Path) -> None:
    import cv2

    split = split_of(name)
    rec = next(r for r in list_recordings() if r.name == name)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for frame, img in fr.read_seek(rec.video, chosen["frame"]):
        stem = f"{name}_{frame:06d}"
        h, w = img.shape
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        lines = pl.yolo_lines(frame_fish(inst, frame), w, h, PAD_BL * bl, SKELETON_3)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-recording", type=int, default=150)
    ap.add_argument("--min-gap-s", type=float, default=2.0)
    ap.add_argument("--max-step-bl", type=float, default=0.05,
                    help="largest step per frame of either fish (body lengths); see b2_frame_check.py")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true", help="select and summarise, write no images")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows, summary = [], []
    for rec in list_recordings():
        split = split_of(rec.name)
        f, inst, bl = candidates(rec.name, args.max_step_bl)
        chosen = (spread(f, args.per_recording, args.min_gap_s, rng) if split != "test"
                  else f.iloc[:0].assign(block=0))
        summary.append({"recording": rec.name, "split": split, "candidates": len(f),
                        "candidates_light": int((f["phase"] == "light").sum()),
                        "candidate_minutes": int(np.unique(np.floor(f["t_s"] / 60)).size), "mined": len(chosen),
                        "mined_light": int((chosen["phase"] == "light").sum())})
        print(f"{rec.name} ({split}): {len(f)} candidates, {len(chosen)} mined", flush=True)
        if len(chosen) == 0:
            continue
        rows.append(chosen.assign(split=split, body_length_px=bl))
        if not args.dry_run:
            write(rec.name, chosen, inst, bl, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat(rows, ignore_index=True)
    manifest["image"] = [f"images/{s}/{r}_{fr_:06d}.jpg" for s, r, fr_ in
                         zip(manifest["split"], manifest["recording"], manifest["frame"], strict=True)]
    manifest.to_csv(args.out / "manifest.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    (args.out / "dataset.yaml").write_text(
        pl.dataset_yaml(args.out, {"train": "images/train", "val": "images/val"}, SKELETON_3, [0, 1, 2]))
    print(pd.DataFrame(summary).to_string(index=False))
    print(manifest.groupby(["split", "phase"]).size())


if __name__ == "__main__":
    main()
