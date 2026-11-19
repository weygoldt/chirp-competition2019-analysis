"""Recompute the model pre-fill of an unlabelled session with the current rule (``pose_model.to_prefill``).

For ``hard`` and ``new`` items: a frame the author labelled in an earlier session
(``truth.csv`` ``source_item``) keeps those labels, with the model's spine2 / spine3
added (``pose_model.add_inner_points``); every other frame gets the model's pre-fill.
``test`` items are never touched (never pre-filled by a model). Refuses a session
that already has labels.

    .venv/bin/python experiments/video2022/b3_refill.py r1 --model weights/pose7_r1_l.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b3_session as bs  # noqa: E402
import pose_labels as pl  # noqa: E402
import pose_model as pm  # noqa: E402
from paths import DATA  # noqa: E402


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    root = bs.LABELS / args.name
    if (root / "labels.jsonl").exists():
        raise SystemExit(f"{root} already has labels; refusing to change its pre-fill")
    weights = Path(args.model) if Path(args.model).is_absolute() else DATA / "video2022" / args.model
    model = pm.load(weights)
    items = json.loads((root / "items.json").read_text())
    truth = pd.read_csv(root / "truth.csv").set_index("item_id")
    earlier: dict[str, dict[str, pl.FrameLabel]] = {}
    n = 0
    for it in items:
        t = truth.loc[it["item_id"]]
        if t["kind"] == "test":
            continue
        img = cv2.imread(str(root / it["image"]), cv2.IMREAD_GRAYSCALE)
        (pred,) = pm.predict(model, [img])
        src = t["source_item"] if isinstance(t["source_item"], str) else ""
        lab = None
        if src:
            sess, iid = src.split("/")
            if sess not in earlier:
                earlier[sess] = pl.load_labels(bs.LABELS / sess / "labels.jsonl")
            lab = earlier[sess].get(iid)
        if lab is not None and lab.status == "done":
            it["prefill"] = [pm.add_inner_points(f, pred).to_json() for f in lab.fish]
        else:
            it["prefill"] = bs.model_prefill(pred)
        n += 1
    (root / "items.json").write_text(json.dumps(items))
    cfg = json.loads((root / "session.json").read_text())
    cfg["model"] = str(weights)
    (root / "session.json").write_text(json.dumps(cfg, indent=1))
    print(f"re-filled {n} items of {root}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
