"""Evaluate a pose model and SLEAP on hand-labelled test frames (milestone B3).

Test frames: the ``test`` split of an assembled dataset (``b3_dataset.py``;
recordings never trained on): the 2022 SLEAP hand labels of 2022-05-11 and the
author's blind test items. For every frame:

- **model**: its predictions on the same image (box confidence >= ``--conf``);
- **SLEAP**: the instances of the SLEAP export (3 nodes = snout, spine1, tail)
  in the export's frame f and, because the export's frame numbers agree with
  ``*converted.mp4`` only to within one frame (``b2_frame_check.py``), also in
  f - 1: ``sleap`` scores frame f, ``sleap_best`` the better of the two per frame.

Matching: hand-labelled fish and predictions are paired one-to-one (Hungarian)
on the mean distance over the points both have; a pair counts if that distance
is below 0.5 body lengths. Reported per method and stratum:

- detection rate: matched hand-labelled fish / hand-labelled fish;
- false detections per frame: unmatched predictions (every visible fish is
  labelled, so these are spurious);
- keypoint error per point (median, px and body lengths) and the share within
  0.1 body lengths (PCK@0.1), over matched fish and the points both have.

    .venv/bin/python experiments/video2022/b3_eval.py pose7_r1 --model runs/<run>/weights/best.pt

Output: ``<run dir>/eval_<dataset>/`` (``fish.csv``, ``points.csv``, ``summary.csv``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1  # noqa: E402
import pose_labels as pl  # noqa: E402
import pose_model as pm  # noqa: E402
from paths import DATA  # noqa: E402

DATASETS = DATA / "video2022" / "datasets"
MATCH_BL = 0.5
PCK_BL = 0.1
SLEAP_MAP = {"head": "snout", "center": "spine1", "tail": "tail"}


def read_yolo(path: Path, w: int = 2048, h: int = 1536) -> list[dict[str, tuple[float, float]]]:
    fish = []
    for line in path.read_text().split("\n"):
        if line.strip():
            v = [float(x) for x in line.split()]
            fish.append({n: (v[5 + 3 * j] * w, v[6 + 3 * j] * h) for j, n in enumerate(pl.SKELETON) if v[7 + 3 * j] > 0})
    return fish


def match(gt: list[dict], pred: list[dict], max_px: float) -> list[tuple[int, int, float]]:
    """(gt index, pred index, mean distance) of one-to-one pairs closer than max_px."""
    if not gt or not pred:
        return []
    cost = np.full((len(gt), len(pred)), 1e9)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            common = [n for n in g if n in p]
            if common:
                cost[i, j] = np.mean([np.hypot(g[n][0] - p[n][0], g[n][1] - p[n][1]) for n in common])
    r, c = linear_sum_assignment(cost)
    return [(i, j, cost[i, j]) for i, j in zip(r, c, strict=True) if cost[i, j] < max_px]


def sleap_fish(inst: pd.DataFrame, frame: int) -> list[dict]:
    rows = inst[inst["frame"] == frame]
    return [{SLEAP_MAP[n]: (r[f"{n}_x"], r[f"{n}_y"]) for n in SLEAP_MAP if np.isfinite(r[f"{n}_x"])}
            for _, r in rows.iterrows()]


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()
    weights = Path(args.model) if Path(args.model).is_absolute() else DATA / "video2022" / args.model
    out = weights.parent.parent / f"eval_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(DATASETS / args.dataset / "manifest.csv")
    man = man[man["split"] == "test"].reset_index(drop=True)
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    model = pm.load(weights)
    root = DATASETS / args.dataset
    images = [cv2.imread(str(root / p), cv2.IMREAD_GRAYSCALE) for p in man["image"]]
    preds = pm.predict(model, images, imgsz=args.imgsz, conf=args.conf)
    inst_cache: dict[str, pd.DataFrame] = {}
    fish_rows, point_rows = [], []
    for k, r in man.iterrows():
        gt = read_yolo((root / r["image"].replace("images/", "labels/", 1)).with_suffix(".txt"))
        if r["recording"] not in inst_cache:
            inst_cache[r["recording"]] = pd.read_parquet(b1.TABLES / "instances" / f"{r['recording']}.parquet")
        inst = inst_cache[r["recording"]]
        px = MATCH_BL * float(bl[r["recording"]])
        methods = {"model": [p.points | {} for p in preds[k]],
                   "sleap": sleap_fish(inst, int(r["frame"]))}
        prev = sleap_fish(inst, int(r["frame"]) - 1)
        m0, m1 = match(gt, methods["sleap"], px), match(gt, prev, px)
        better = methods["sleap"] if (len(m0), -sum(d for *_, d in m0)) >= (len(m1), -sum(d for *_, d in m1)) else prev
        methods["sleap_best"] = better
        for method, pred in methods.items():
            pairs = match(gt, pred, px)
            fish_rows.append({"item": r["item_id"], "source": r["source"], "recording": r["recording"],
                              "phase": r["phase"], "method": method, "n_gt": len(gt), "n_pred": len(pred),
                              "n_matched": len(pairs), "n_false": len(pred) - len(pairs)})
            for i, j, _ in pairs:
                for n, g in gt[i].items():
                    if n in pred[j]:
                        d = float(np.hypot(g[0] - pred[j][n][0], g[1] - pred[j][n][1]))
                        point_rows.append({"item": r["item_id"], "source": r["source"], "phase": r["phase"],
                                           "method": method, "point": n, "err_px": d,
                                           "err_bl": d / float(bl[r["recording"]])})
    fish = pd.DataFrame(fish_rows)
    pts = pd.DataFrame(point_rows)
    fish.to_csv(out / "fish.csv", index=False)
    pts.to_csv(out / "points.csv", index=False)
    rows = []
    for (method, group), g in [(k, v) for k, v in fish.groupby(["method", "source"])] + \
            [((m, "all"), v) for m, v in fish.groupby("method")]:
        p = pts[(pts["method"] == method) & ((pts["source"] == group) | (group == "all"))]
        row = {"method": method, "source": group, "frames": len(g), "gt_fish": int(g["n_gt"].sum()),
               "detection_rate": g["n_matched"].sum() / max(1, g["n_gt"].sum()),
               "false_per_frame": g["n_false"].sum() / len(g)}
        for n in pl.SKELETON:
            e = p.loc[p["point"] == n, "err_bl"]
            row[f"{n}_median_bl"] = float(e.median()) if len(e) else np.nan
            row[f"{n}_pck"] = float((e < PCK_BL).mean()) if len(e) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary[["method", "source", "frames", "gt_fish", "detection_rate", "false_per_frame",
                       "snout_median_bl", "spine1_median_bl", "tail_median_bl", "snout_pck", "tail_pck"]]
              .round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
