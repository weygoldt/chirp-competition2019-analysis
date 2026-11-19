"""Contact sheet of a labelling session's items with their pre-filled points (what the labeller shows first).

    .venv/bin/python experiments/video2022/b3_session_sheet.py SESSION_DIR [--kinds hard new] [--n 12]

Writes ``SESSION_DIR/sheet_<kinds>.png``: per item a crop around its pre-filled fish (or the
whole frame if there is none), the points coloured as in the labeller, the lines of the skeleton.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_labels as pl  # noqa: E402


def main() -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("--kinds", nargs="+", default=["hard", "new"])
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    items = {it["item_id"]: it for it in json.loads((args.session / "items.json").read_text())}
    cfg = json.loads((args.session / "session.json").read_text())
    colors = cfg["colors"]
    truth = pd.read_csv(args.session / "truth.csv")
    rows = truth[truth["kind"].isin(args.kinds)]
    rows = rows.sample(min(args.n, len(rows)), random_state=1)
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(len(rows) / ncol)), ncol, figsize=(4 * ncol, 4 * np.ceil(len(rows) / ncol)),
                             constrained_layout=True, squeeze=False)
    for ax, r in zip(axes.ravel(), rows.itertuples(), strict=False):
        img = cv2.imread(str(args.session / "images" / f"{r.item_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        fish = [pl.Fish.from_dict(f) for f in items[r.item_id]["prefill"]]
        allp = [p for f in fish for p in f.points.values()]
        for f in fish:
            for a, b in pl.EDGES:
                if a in f.points and b in f.points:
                    ax.plot([f.points[a][0], f.points[b][0]], [f.points[a][1], f.points[b][1]], color="white", lw=0.7)
            for n, p in f.points.items():
                ax.plot(*p, "o", ms=4, color=colors[n], mec="k", mew=0.4)
        if allp:
            a = np.array(allp)
            c, s = a.mean(0), max(500, np.ptp(a, axis=0).max() + 200)
            ax.set_xlim(c[0] - s / 2, c[0] + s / 2)
            ax.set_ylim(c[1] + s / 2, c[1] - s / 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{r.kind} | {r.stratum} | {r.phase} | {r.recording[5:10]} | {len(fish)} fish", fontsize=8)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"{args.session.name}: pre-filled points ({', '.join(args.kinds)} items)", fontsize=11)
    out = args.session / f"sheet_{'_'.join(args.kinds)}.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
