"""Assemble a 7-keypoint YOLO pose dataset from several label sources (milestone B3, one per round).

Every source is a folder with ``labels.jsonl`` (``pose_labels`` frame labels,
the latest line of an item wins; only ``done`` items are used) and an
``index.csv`` (``item_id, image, recording, frame, split, phase, t_s``; ``image``
relative to the folder). Sources so far:

- ``mined_v1``: confident two-fish SLEAP frames (``b2_mine.py``), 3 of the 7
  points (snout, spine1, tail); written here from its YOLO labels the first
  time (:func:`mined_source`);
- ``sleap_v000``: the 2022 SLEAP hand labels, all 7 points
  (``b2_convert_sleap_labels.py``);
- labelling sessions (``labels/b2_v1``, later rounds): their ``truth.csv`` gives
  the index (:func:`session_source`); test items go to ``test``.

Splits are by recording (``b2_mine.split_of``: test / val / train), except
that a session's blind test items always go to ``test``. Images are linked
(symlinks; nothing is copied), labels written with ``pose_labels.yolo_lines``
(box = labelled points padded by 0.1 body lengths of the recording).

    .venv/bin/python experiments/video2022/b3_dataset.py pose7_r1 mined_v1 sleap_v000 labels/b2_v1

Output: ``$CHIRP_DATA/video2022/datasets/<name>/`` with ``images/``, ``labels/``,
``manifest.csv``, ``summary.csv``, ``dataset.yaml``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1  # noqa: E402
import b2_mine  # noqa: E402
import pose_labels as pl  # noqa: E402
from paths import DATA  # noqa: E402

ROOT = DATA / "video2022"
DATASETS = ROOT / "datasets"
MINED_3 = ("snout", "spine1", "tail")  # keypoint order of the mined_v1 YOLO files
W, H = 2048, 1536


def mined_source(root: Path = DATASETS / "mined_v1") -> Path:
    """Write ``labels.jsonl`` and ``index.csv`` for mined_v1 from its YOLO label files (once)."""
    if (root / "labels.jsonl").exists() and (root / "index.csv").exists():
        return root
    man = pd.read_csv(root / "manifest.csv")
    man["item_id"] = [Path(p).stem for p in man["image"]]
    for r in man.itertuples():
        lab = (root / r.image.replace("images/", "labels/", 1)).with_suffix(".txt")
        fish = []
        for line in lab.read_text().split("\n"):
            if not line.strip():
                continue
            v = [float(x) for x in line.split()]
            pts = {n: (v[5 + 3 * j] * W, v[6 + 3 * j] * H) for j, n in enumerate(MINED_3) if v[7 + 3 * j] > 0}
            fish.append(pl.Fish(pts, origin="sleap"))
        pl.append_label(root / "labels.jsonl", pl.FrameLabel(r.item_id, "done", fish, labeller="mined_v1"))
    man[["item_id", "image", "recording", "frame", "split", "phase", "t_s"]].to_csv(root / "index.csv", index=False)
    return root


def session_source(root: Path) -> Path:
    """Write ``index.csv`` for a labelling session from its ``truth.csv`` (test items -> test)."""
    t = pd.read_csv(root / "truth.csv")
    idx = pd.DataFrame({"item_id": t["item_id"], "image": [f"images/{i}.jpg" for i in t["item_id"]],
                        "recording": t["recording"], "frame": t["frame"],
                        "split": ["test" if k == "test" else b2_mine.split_of(r) for k, r in zip(t["kind"], t["recording"], strict=True)],
                        "phase": t["phase"], "t_s": t["t_s"]})
    idx.to_csv(root / "index.csv", index=False)
    return root


def resolve(spec: str) -> Path:
    """'mined_v1' / 'sleap_v000' (datasets) or 'labels/<session>' (sessions) -> a source folder with an index."""
    if spec == "mined_v1":
        return mined_source()
    path = ROOT / spec if not Path(spec).is_absolute() else Path(spec)
    if not path.exists():
        path = DATASETS / spec
    if (path / "truth.csv").exists():
        return session_source(path)
    if not (path / "index.csv").exists():
        raise FileNotFoundError(f"{path}: no index.csv / truth.csv")
    return path


def assemble(name: str, sources: list[str]) -> pd.DataFrame:
    out = DATASETS / name
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new name (datasets are not overwritten)")
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    rows = []
    for spec in sources:
        src = resolve(spec)
        labels = pl.load_labels(src / "labels.jsonl")
        idx = pd.read_csv(src / "index.csv")
        tag = src.name
        for r in idx.itertuples():
            lab = labels.get(r.item_id)
            if lab is None or lab.status != "done":
                continue
            stem = f"{tag}__{r.item_id}"
            img = out / "images" / r.split / f"{stem}.jpg"
            txt = out / "labels" / r.split / f"{stem}.txt"
            img.parent.mkdir(parents=True, exist_ok=True)
            txt.parent.mkdir(parents=True, exist_ok=True)
            os.symlink((src / r.image).resolve(), img)
            lines = pl.yolo_lines(lab.fish, W, H, b2_mine.PAD_BL * float(bl[r.recording]))
            txt.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_pts = [f.n_points() for f in lab.fish]
            rows.append({"source": tag, "item_id": r.item_id, "split": r.split, "recording": r.recording,
                         "frame": r.frame, "phase": r.phase, "t_s": r.t_s, "n_fish": len(lab.fish),
                         "n_points": sum(n_pts), "image": f"images/{r.split}/{stem}.jpg"})
    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)
    summary = man.groupby(["split", "source"]).agg(frames=("item_id", "size"), fish=("n_fish", "sum"),
                                                   points=("n_points", "sum")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    splits = {s: f"images/{s}" for s in ("train", "val", "test") if (out / "images" / s).exists()}
    (out / "dataset.yaml").write_text(pl.dataset_yaml(out, splits))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", help="dataset name, e.g. pose7_r1")
    ap.add_argument("sources", nargs="+", help="mined_v1, sleap_v000, labels/<session> ...")
    args = ap.parse_args()
    print(assemble(args.name, args.sources).to_string(index=False))


if __name__ == "__main__":
    main()
