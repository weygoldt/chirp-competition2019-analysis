"""Add a pose model's spine2 / spine3 to a label source that has only snout, spine1 and tail (pseudo labels).

Why: the mined frames (``mined_v1``) carry SLEAP's 3 points. YOLO trains the
confidence of an unlabelled point towards "not visible", so with 95 % of the
training frames lacking spine2 / spine3 the round-1 model learned that these
points are never visible (confidence 0.01-0.03), although its positions for
them are usable. Here, for every fish of the source, the model's prediction
whose snout / spine1 / tail lie closest is taken; if they lie within
``--max-bl`` body lengths of the labelled ones on average, its spine2 and
spine3 are added. Fins are not added (their visibility is unknown).

    .venv/bin/python experiments/video2022/b3_pseudo_inner.py mined_v1 mined_v1_inner_r1 --model weights/pose7_r1_l.pt

Output: a new source ``$CHIRP_DATA/video2022/datasets/<out>/`` (``labels.jsonl``,
``index.csv`` with absolute image paths, ``summary.json``); the input is not changed.
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
import b1_assess_sleap as b1  # noqa: E402
import b3_dataset as bd  # noqa: E402
import pose_labels as pl  # noqa: E402
import pose_model as pm  # noqa: E402
from paths import DATA  # noqa: E402

ANCHORS = ("snout", "spine1", "tail")


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-bl", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    src = bd.resolve(args.source)
    out = bd.DATASETS / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    weights = Path(args.model) if Path(args.model).is_absolute() else DATA / "video2022" / args.model
    model = pm.load(weights)
    labels = pl.load_labels(src / "labels.jsonl")
    idx = pd.read_csv(src / "index.csv")
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    out.mkdir(parents=True)
    n_fish = n_added = 0
    rows = idx.to_dict("records")
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        imgs = [cv2.imread(str(src / r["image"]), cv2.IMREAD_GRAYSCALE) for r in chunk]
        for r, preds in zip(chunk, pm.predict(model, imgs, batch=args.batch), strict=True):
            lab = labels[r["item_id"]]
            new_fish = []
            for f in lab.fish:
                n_fish += 1
                common = [n for n in ANCHORS if n in f.points]
                d = [np.mean([np.hypot(f.points[n][0] - p.points[n][0], f.points[n][1] - p.points[n][1])
                              for n in common]) for p in preds] if common else []
                if d and min(d) < args.max_bl * float(bl[r["recording"]]):
                    p = preds[int(np.argmin(d))]
                    pts = dict(f.points)
                    for n in pm.INNER:
                        pts.setdefault(n, p.points[n])
                    f = pl.Fish(pts, origin=f.origin + "+model_inner")
                    n_added += 1
                new_fish.append(f)
            pl.append_label(out / "labels.jsonl", pl.FrameLabel(r["item_id"], "done", new_fish,
                                                                labeller=f"{lab.labeller}+{weights.name}"))
        print(f"{min(i + args.batch, len(rows))}/{len(rows)}", flush=True)
    idx2 = idx.copy()
    idx2["image"] = [str((src / p).resolve()) for p in idx["image"]]
    idx2.to_csv(out / "index.csv", index=False)
    summary = {"source": str(src), "model": str(weights), "max_bl": args.max_bl, "fish": n_fish,
               "fish_with_inner_points_added": n_added}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
