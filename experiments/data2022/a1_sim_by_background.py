"""Detector on the simulated test scenes, split by the scene's real background recording.

The sim_v4 test set (300 one-minute hybrid scenes, seeds 7700-7999) puts
simulated chirping fish on real background segments; 125 of them use 2022
backgrounds (recordings 2022-05-10 and 2022-06-29, the whole 1x11 line), 139
2019/2020 backgrounds, 36 field recordings. Ground truth is the simulator's.
This script re-reads the existing evaluation of the production detector after
post-processing (``runs/dualhead_r18_sim_v4_r1/eval_postprocess/sim_test_after``)
and splits it by the background's dataset:

- precision, recall and false positives per scene-minute at thresholds
  >= 0.5 and by score bin, from ``items.csv`` (the evaluation's matched items
  at its 0.5 threshold: every true positive and false positive box with its
  score, every missed label as ``fn``; a label matched at 0.5 counts as found
  at a higher threshold if its box's score reaches it). The precision by score
  bin is the simulated counterpart of the real_v1 agreement by score bin.

    .venv/bin/python experiments/data2022/a1_sim_by_background.py

Output: ``$CHIRP_DATA/data2022/a1/sim_by_background/`` (``by_threshold.csv``,
``by_score_bin.csv``, ``scenes.csv``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA

EVAL = DATA / "runs" / "dualhead_r18_sim_v4_r1" / "eval_postprocess" / "sim_test_after"
SCENES = DATA / "scenes" / "train_v4" / "test"
OUT = DATA / "data2022" / "a1" / "sim_by_background"
GROUP = {"competition2019": "2019/2020", "competition2022": "2022",
         "field_flona2025": "field", "field_iriri2026": "field"}
BINS = [0.5, 0.7, 0.8, 0.9, 1.0001]
BIN_NAMES = ["0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1"]


def scene_table() -> pd.DataFrame:
    rows = []
    for d in sorted(p.parent for p in SCENES.glob("*/scene.json")):
        s = json.loads((d / "scene.json").read_text())
        bg = s.get("background") or {}
        rows.append({"scene": d.name, "background_dataset": bg.get("dataset"),
                     "background_recording": bg.get("recording"), "group": GROUP.get(bg.get("dataset"), "none"),
                     "n_channels": len(bg.get("channels", [])), "duration_s": s["duration_s"]})
    return pd.DataFrame(rows)


def scene_of(tile: pd.Series) -> pd.Series:
    """'train_v4_test-007700_f0000440' -> '007700'."""
    return tile.str.extract(r"-(\d{6})_f")[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = scene_table()
    scenes.to_csv(OUT / "scenes.csv", index=False)
    group = scenes.set_index("scene")["group"]

    minutes = scenes.groupby("group")["duration_s"].sum() / 60
    items = pd.read_csv(EVAL / "items.csv")
    items["scene"] = scene_of(items["recording"])
    items["group"] = items["scene"].map(group)
    boxes = items[items["outcome"].isin(["tp", "fp"])].copy()
    boxes["score_bin"] = pd.cut(boxes["score"], BINS, right=False, labels=BIN_NAMES)
    by_bin = (boxes.groupby(["group", "score_bin"], observed=False)["outcome"]
              .agg(n="size", tp=lambda o: int((o == "tp").sum())).reset_index())
    by_bin["fp"] = by_bin["n"] - by_bin["tp"]
    by_bin["precision"] = by_bin["tp"] / by_bin["n"].replace(0, np.nan)
    by_bin["fp_per_min"] = by_bin["fp"] / by_bin["group"].map(minutes)
    by_bin.to_csv(OUT / "by_score_bin.csv", index=False)
    rows = []
    for g, it in items.groupby("group"):
        n_labels = int(it["outcome"].isin(["tp", "fn"]).sum())
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            tp = int(((it["outcome"] == "tp") & (it["score"] >= thr)).sum())
            fp = int(((it["outcome"] == "fp") & (it["score"] >= thr)).sum())
            rows.append({"group": g, "threshold": thr, "labels": n_labels, "tp": tp, "fp": fp,
                         "precision": tp / (tp + fp), "recall": tp / n_labels,
                         "fp_per_min": fp / minutes[g], "labels_per_min": n_labels / minutes[g]})
    by_thr = pd.DataFrame(rows)
    by_thr.to_csv(OUT / "by_threshold.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(scenes.groupby("group").agg(scenes=("scene", "size"), recordings=("background_recording", "nunique")))
        print(by_thr.round(3).to_string(index=False))
        print(by_bin.round(3).to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
