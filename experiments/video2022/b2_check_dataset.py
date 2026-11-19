"""Look at a YOLO pose dataset of fish frames: example crops with their labels, and coverage.

Reads the dataset back from disk (images and ``.txt`` labels, i.e. what the
trainer will read), so it checks the whole chain from SLEAP keypoints to
normalised labels.

- ``check_crops.png``: random fish (seeded), a 400 x 400 px crop around each with
  its box and head / centre / tail as stored in the label file;
- ``check_coverage.png``: frames per recording and split, the arena positions of
  the labelled fish over one frame, and the frames' recording times.

    .venv/bin/python experiments/video2022/b2_check_dataset.py [DATASET_DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2_mine  # noqa: E402

COLORS = {"head": "#ff4d4d", "center": "#ffd24d", "tail": "#4dc3ff"}
CROP = 400
SEED = 7


def read_label(path: Path, w: int, h: int) -> list[dict]:
    fish = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        v = [float(x) for x in line.split()]
        cx, cy, bw, bh = v[1] * w, v[2] * h, v[3] * w, v[4] * h
        kp = {}
        for j, n in enumerate(("head", "center", "tail")):
            x, y, vis = v[5 + 3 * j: 8 + 3 * j]
            kp[n] = (x * w, y * h) if vis > 0 else None
        fish.append({"box": (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), **kp})
    return fish


def crops_figure(root: Path, manifest: pd.DataFrame, out: Path, n: int = 24) -> None:
    """Random fish of the dataset with their stored labels."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(SEED)
    rows = manifest.iloc[rng.choice(len(manifest), min(n, len(manifest)), replace=False)]
    ncol = 6
    fig, axes = plt.subplots(int(np.ceil(len(rows) / ncol)), ncol, figsize=(2.6 * ncol, 2.9 * np.ceil(len(rows) / ncol)),
                             constrained_layout=True)
    for ax, r in zip(axes.ravel(), rows.itertuples(), strict=False):
        img = cv2.imread(str(root / r.image), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        lab = Path(str(root / r.image).replace("/images/", "/labels/")).with_suffix(".txt")
        fish = read_label(lab, w, h)
        f = fish[rng.integers(len(fish))]
        x0, y0, x1, y1 = f["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = int(np.clip(cx - CROP / 2, 0, w - CROP)), int(np.clip(cy - CROP / 2, 0, h - CROP))
        ax.imshow(img[b:b + CROP, a:a + CROP], cmap="gray", vmin=0, vmax=255, extent=(a, a + CROP, b + CROP, b))
        for g in fish:
            gx0, gy0, gx1, gy1 = g["box"]
            ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, ec="#7fff7f", lw=0.8))
            pts = [g[n] for n in ("head", "center", "tail") if g[n] is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="white", lw=0.6)
            for n, c in COLORS.items():
                if g[n] is not None:
                    ax.plot(*g[n], "o", ms=3.5, color=c, mec="k", mew=0.4)
        ax.set_xlim(a, a + CROP)
        ax.set_ylim(b + CROP, b)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{r.recording[5:10]} {r.phase} f{r.frame}", fontsize=7)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"{root.name}: frames as the trainer reads them; box (green), head (red), centre (yellow), "
                 "tail (blue)", fontsize=10)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def coverage_figure(root: Path, manifest: pd.DataFrame, out: Path) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.2, 1.3, 1]})
    counts = manifest.groupby(["recording", "split"]).size().unstack(fill_value=0)
    recs = sorted(set(counts.index) | set(b2_mine.SPLITS))
    counts = counts.reindex(recs, fill_value=0)
    left = np.zeros(len(recs))
    for split, colour in (("train", "#2b8cbe"), ("val", "#fdae6b"), ("test", "#74c476")):
        if split in counts:
            axes[0].barh(range(len(recs)), counts[split], left=left, color=colour, label=split)
            left += counts[split].to_numpy()
    for i, r in enumerate(recs):
        if b2_mine.split_of(r) == "test" and counts.loc[r].sum() == 0:
            axes[0].text(1, i, "test recording: not mined", va="center", fontsize=7, color="#555")
    axes[0].set_yticks(range(len(recs)), [r[:10] for r in recs], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("frames")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("frames per recording", fontsize=10)
    first = manifest.iloc[0]
    img = cv2.imread(str(root / first.image), cv2.IMREAD_GRAYSCALE)
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=255)
    xs, ys = [], []
    for r in manifest.itertuples():
        lab = Path(str(root / r.image).replace("/images/", "/labels/")).with_suffix(".txt")
        for f in read_label(lab, img.shape[1], img.shape[0]):
            if f["center"] is not None:
                xs.append(f["center"][0])
                ys.append(f["center"][1])
    axes[1].plot(xs, ys, ".", ms=1.5, color="#e6550d", alpha=0.5)
    axes[1].set_axis_off()
    axes[1].set_title(f"centre of every labelled fish ({len(xs):,})", fontsize=10)
    axes[2].hist(manifest["t_s"] / 3600, bins=np.arange(0, 4.05, 0.1), color="#636363")
    axes[2].axvline(3.0, color="k", ls="--", lw=1)
    axes[2].text(3.02, axes[2].get_ylim()[1] * 0.9, "light on", fontsize=8)
    axes[2].set_xlabel("recording time (h)")
    axes[2].set_ylabel("frames")
    axes[2].set_title("when the frames are", fontsize=10)
    light = int((manifest["phase"] == "light").sum())
    fig.suptitle(f"{root.name}: {len(manifest):,} frames from {manifest['recording'].nunique()} recordings, "
                 f"{light} of them in the light phase", fontsize=11)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", type=Path, nargs="?", default=b2_mine.OUT)
    args = ap.parse_args()
    manifest = pd.read_csv(args.dataset / "manifest.csv")
    crops_figure(args.dataset, manifest, args.dataset / "check_crops.png")
    coverage_figure(args.dataset, manifest, args.dataset / "check_coverage.png")
    print(f"wrote {args.dataset / 'check_crops.png'}, {args.dataset / 'check_coverage.png'}")


if __name__ == "__main__":
    main()
