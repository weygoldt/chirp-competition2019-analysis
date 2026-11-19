"""Build a keypoint-correction session aimed at fish at the shelters (round 3, dark phase only).

Why: the round-2 model sometimes reads one fish crossing a block as two fish
(the block hides the middle of the body; two overlapping detections of
confidence 0.5-0.7), and finds a fish that only peeks out of a shelter with low
confidence. For chase detection a false second fish next to the first looks like
contact, so these frames get labelled first.

Candidates per recording (dark phase, recording time 0-3 h; frames of earlier
sessions skipped): ``--sleap-pool`` frames where a SLEAP point lies near a
shelter and ``--random-pool`` random dark frames. The model runs on them
(detections down to box confidence 0.2). A detection is *at a shelter* when one of
its body points lies within ``NEAR_BL`` body lengths of a shelter
(``shelters.mask``). Frames without a detection at a shelter are dropped. Score:

- ``double``: two detections, one of them at a shelter, with body points closer than
  ``DOUBLE_BL`` body lengths (one fish split by a shelter, or two fish touching
  there) -> +2;
- each detection at a shelter with box confidence < 0.5 -> +1;
- ``weak_points`` of ``pose_model.frame_uncertainty`` (0-1).

Items (``kind`` in ``truth.csv``):

- ``new``: ``--n-new`` frames of train / val recordings, highest score first, at
  most ``--per-recording`` per recording, >= 30 s apart; pre-filled with the
  model (fish with box confidence >= 0.5).
- ``test``: ``--n-test`` frames of the three test recordings with a detection at a
  shelter, drawn at random (not by score, so the test set is not built from the
  model's failures), >= 30 s apart; never pre-filled (blind).

    .venv/bin/python experiments/video2022/b3_session_shelters.py r3 --model weights/pose7_r2_l.pt

Output: ``$CHIRP_DATA/video2022/labels/<name>/`` (``items.json``, ``session.json``,
``images/``, ``truth.csv``, ``candidates_scored.csv``), served with ``serve_pose.sh <name>``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1
import b2_mine
import b3_session as bs
import frames as fr
import pose_labels as pl
import pose_model as pm
import shelters
from paths import DATA
from recordings2022 import list_recordings

NEAR_BL = 0.2
DOUBLE_BL = 0.2
SURE = 0.5
MIN_GAP_S = 30.0
SEED = 20261120
NOTE = bs.NOTE + (" ROUND 3, shelters: a fish partly hidden by a block or the tube is ONE fish; put points only on "
                  "its visible parts (e.g. snout and spine1 on one side of a block, tail on the other). If the "
                  "pre-fill shows two fish that are really one, delete one and fix the other. A tail or head "
                  "peeking out of a shelter is a fish (label the visible points). Points that end up on a shelter "
                  "can stay; they are removed automatically.")


def shelter_distance(recording: str) -> np.ndarray:
    """Distance (px) of every pixel to the nearest shelter pixel."""
    import cv2

    return cv2.distanceTransform((~shelters.mask(recording)).astype(np.uint8), cv2.DIST_L2, 5)


def at_shelter(p: pm.PredFish, dist: np.ndarray, near_px: float) -> bool:
    h, w = dist.shape
    for n in pm.BODY:
        x, y = (int(v) for v in np.round(p.points[n]))
        if 0 <= x < w and 0 <= y < h and dist[y, x] < near_px:
            return True
    return False


def min_point_distance(a: pm.PredFish, b: pm.PredFish) -> float:
    pa = np.array([a.points[n] for n in pm.BODY])
    pb = np.array([b.points[n] for n in pm.BODY])
    return float(np.min(np.hypot(*(pa[:, None, :] - pb[None, :, :]).transpose(2, 0, 1))))


def shelter_score(preds: list[pm.PredFish], dist: np.ndarray, bl: float) -> dict | None:
    """Score of a frame (see module docstring), or None if no detection is at a shelter."""
    near = [at_shelter(p, dist, NEAR_BL * bl) for p in preds]
    if not any(near):
        return None
    double = any((near[i] or near[j]) and min_point_distance(preds[i], preds[j]) < DOUBLE_BL * bl
                 for i, j in itertools.combinations(range(len(preds)), 2))
    weak_near = sum(1 for p, nr in zip(preds, near, strict=True) if nr and p.box_conf < SURE)
    weak_points = pm.frame_uncertainty(preds, SURE)["weak_points"]
    stratum = "double" if double else ("weak_at_shelter" if weak_near else "weak_points")
    return {"double": double, "weak_at_shelter": weak_near, "weak_points": weak_points,
            "n_detections": len(preds), "n_confident": sum(p.box_conf >= SURE for p in preds),
            "score": 2 * double + weak_near + weak_points, "stratum": stratum}


def candidate_frames(rec: str, dist: np.ndarray, bl: float, taken: set, n_sleap: int, n_random: int,
                     rng: np.random.Generator) -> np.ndarray:
    t = np.load(b1.TABLES / "frame_time" / f"{rec}.npy")
    dark = np.flatnonzero((t >= 0) & (t < b1.LIGHT_ON_S))
    inst = pd.read_parquet(b1.TABLES / "instances" / f"{rec}.parquet")
    inst = inst[inst["phase"] == 1]
    h, w = dist.shape
    near = np.zeros(len(inst), bool)
    for node in ("head", "center", "tail"):
        x = inst[f"{node}_x"].to_numpy()
        y = inst[f"{node}_y"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        xi = np.clip(np.nan_to_num(x).round().astype(int), 0, w - 1)
        yi = np.clip(np.nan_to_num(y).round().astype(int), 0, h - 1)
        near |= ok & (dist[yi, xi] < NEAR_BL * bl)
    sleap = np.unique(inst.loc[near, "frame"].to_numpy())
    sleap = sleap[[(rec, int(f)) not in taken for f in sleap]] if len(sleap) else sleap
    rand = dark[[(rec, int(f)) not in taken for f in dark]]
    pick = [rng.choice(sleap, min(n_sleap, len(sleap)), replace=False) if len(sleap) else np.array([], int),
            rng.choice(rand, n_random, replace=False)]
    return np.unique(np.concatenate(pick).astype(int))


def spaced(rows: list[dict], per_recording: int, n: int) -> list[dict]:
    """The first rows (in the given order) with at most per_recording per recording, >= MIN_GAP_S apart."""
    chosen: list[dict] = []
    for r in rows:
        mine = [c for c in chosen if c["recording"] == r["recording"]]
        if len(mine) >= per_recording or any(abs(c["t_s"] - r["t_s"]) < MIN_GAP_S for c in mine):
            continue
        chosen.append(r)
        if len(chosen) >= n:
            break
    return chosen


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--model", required=True, help="weights (.pt), absolute or relative to $CHIRP_DATA/video2022")
    ap.add_argument("--n-new", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--per-recording", type=int, default=5)
    ap.add_argument("--sleap-pool", type=int, default=50)
    ap.add_argument("--random-pool", type=int, default=30)
    args = ap.parse_args()
    out = bs.LABELS / args.name
    if out.exists():
        raise SystemExit(f"{out} exists; sessions are never rebuilt")
    weights = Path(args.model) if Path(args.model).is_absolute() else DATA / "video2022" / args.model
    rng = np.random.default_rng(SEED)
    model = pm.load(weights)
    bl_px = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    b2_truth = pd.read_csv(bs.B2 / "truth.csv")
    taken = bs.used_frames() | set(zip(b2_truth["recording"], b2_truth["frame"].astype(int), strict=True))
    scored = []
    for rec in list_recordings():
        bl = float(bl_px[rec.name])
        dist = shelter_distance(rec.name)
        t = np.load(b1.TABLES / "frame_time" / f"{rec.name}.npy")
        cand = candidate_frames(rec.name, dist, bl, taken, args.sleap_pool, args.random_pool, rng)
        imgs = dict(fr.read_seek(rec.video, cand))
        keys = sorted(imgs)
        split = b2_mine.split_of(rec.name)
        n_kept = 0
        for f, preds in zip(keys, pm.predict(model, [imgs[f] for f in keys], batch=8), strict=True):
            s = shelter_score(preds, dist, bl)
            if s is None:
                continue
            n_kept += 1
            scored.append({"recording": rec.name, "split": split, "frame": f, "t_s": float(t[f]), **s,
                           "prefill": bs.model_prefill(preds)})
        print(f"{rec.name} ({split}): {n_kept} of {len(keys)} candidates with a detection at a shelter", flush=True)
    sc = pd.DataFrame(scored)
    train = sc[sc["split"] != "test"].sample(frac=1.0, random_state=SEED).sort_values("score", ascending=False,
                                                                                        kind="stable")
    test = sc[sc["split"] == "test"].sample(frac=1.0, random_state=SEED)
    new = spaced(train.to_dict("records"), args.per_recording, args.n_new)
    blind = spaced(test.to_dict("records"), int(np.ceil(args.n_test / 3)), args.n_test)
    (out / "images").mkdir(parents=True)
    sc.drop(columns=["prefill"]).to_csv(out / "candidates_scored.csv", index=False)
    videos = {rec.name: rec.video for rec in list_recordings()}
    items, truth = [], []
    for kind, rows in (("new", new), ("test", blind)):
        for r in rows:
            iid = secrets.token_hex(5)
            (_, img), = fr.read_seek(videos[r["recording"]], [r["frame"]])
            cv2.imwrite(str(out / "images" / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, bs.JPEG_QUALITY])
            items.append({"item_id": iid, "image": f"images/{iid}.jpg", "width": img.shape[1], "height": img.shape[0],
                          "prefill": r["prefill"] if kind == "new" else []})
            truth.append({"item_id": iid, "kind": kind, "split": "test" if kind == "test" else r["split"],
                          "recording": r["recording"], "frame": int(r["frame"]), "t_s": r["t_s"], "phase": "dark",
                          "stratum": r["stratum"] if kind == "new" else "shelter", "uncertainty": r["score"],
                          "source_item": ""})
    order = rng.permutation(len(items))
    (out / "items.json").write_text(json.dumps([items[i] for i in order]))
    pd.DataFrame(truth).iloc[order].to_csv(out / "truth.csv", index=False)
    (out / "session.json").write_text(json.dumps({"nodes": list(pl.SKELETON), "edges": [list(e) for e in pl.EDGES],
                                                  "colors": bs.COLORS, "note": NOTE, "model": str(weights)}, indent=1))
    td = pd.DataFrame(truth)
    print(td.groupby(["kind", "stratum"]).size().to_string())
    print(f"{len(td)} items -> {out}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
