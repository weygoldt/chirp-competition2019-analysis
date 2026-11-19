"""Convert the 2022 SLEAP hand labels (``sleap_tracking/labels.v000.slp``) to the YOLO pose format.

The file holds 100 hand-labelled frames (Leonie Reetz, May 2022) with a
7-point fish skeleton (snout, spine1-3, tail, left and right fin; the file
also names an ``LED_on`` node that is not part of the skeleton). All 7 points
are kept: this is the skeleton of the pose model (``pose_labels.SKELETON``,
author's decision 2026-11-19). spine1 lies 18 % of the way from snout to tail
(median of the 149 fully labelled fish), where the 3-node exports put their
"center", so the mined frames' 3 points are snout, spine1 and tail.

Per frame:

- **user instances only** (predicted instances stored in the file are model
  output);
- **duplicates removed**: some fish carry two near-identical user instances;
  when two instances share >= 2 visible body nodes at a median distance below
  20 px, the less complete one is dropped (``sleap_slp.duplicates``);
- **video frame**: the labels index the original ``<date>_0.mp4``, the
  dataset has ``*converted.mp4``, whose frames agree to within one
  (``b2_frame_check.py``). The frame used is the one of f - 1, f, f + 1 in
  which the labelled body points (snout, spine1-3, tail) are darkest (the
  fish are dark on a bright floor); the grey values are kept in the manifest;
- **box**: the labelled points padded by 0.1 body lengths (``pose_labels.yolo_lines``),
  the rule of every source.

Splits by recording (``b2_mine.split_of``): 2022-05-10 and 2022-05-12 train,
2022-05-11 **test** (a held-out recording: these frames must not be trained
on). The 2022-05-03 frame has no video on disk and is skipped.

    .venv/bin/python experiments/video2022/b2_convert_sleap_labels.py

Output: ``$CHIRP_DATA/video2022/datasets/sleap_v000/`` with ``images/{train,test}/``,
``labels/{train,test}/`` (YOLO, 7 keypoints), ``labels.jsonl`` + ``index.csv`` (the
canonical labels for ``b3_dataset.py``), ``manifest.csv``, ``summary.json``, ``dataset.yaml``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1  # noqa: E402
import b2_mine  # noqa: E402
import frames as fr  # noqa: E402
import pose_labels as pl  # noqa: E402
import sleap_slp  # noqa: E402
from paths import DATA, REC_2022  # noqa: E402
from recordings2022 import list_recordings  # noqa: E402

SLP = REC_2022 / "sleap_tracking" / "labels.v000.slp"
OUT = DATA / "video2022" / "datasets" / "sleap_v000"
BODY = ("snout", "spine1", "spine2", "spine3", "tail")
MAP = {n: n for n in pl.SKELETON}
DUPLICATE_PX = 20.0
OFFSETS = (-1, 0, 1)
JPEG_QUALITY = 92


def recording_of(video_file: str) -> str | None:
    """'/home/efish/leonie/videos/2022.05.10_0.mp4' -> '2022-05-10-10_00' (None if not on disk)."""
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})_0\.mp4$", video_file)
    if not m:
        return None
    date = "-".join(m.groups())
    hits = [r.name for r in list_recordings() if r.name.startswith(date)]
    return hits[0] if len(hits) == 1 else None


def grey_under(img: np.ndarray, pts: np.ndarray) -> float:
    """Median of the 3x3-px median grey values at the points."""
    h, w = img.shape
    vals = []
    for x, y in pts:
        xi, yi = round(x), round(y)
        vals.append(np.median(img[max(0, yi - 1):min(h, yi + 2), max(0, xi - 1):min(w, xi + 2)]))
    return float(np.median(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slp", type=Path, default=SLP)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    import cv2

    if (args.out / "labels.jsonl").exists():
        raise SystemExit(f"{args.out} exists; remove it to rebuild")
    args.out.mkdir(parents=True, exist_ok=True)

    df, _ = sleap_slp.read_user_instances(args.slp)
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    rows, skipped = [], []
    for (video_file, frame_idx), g in df.groupby(["video_file", "frame_idx"]):
        name = recording_of(video_file)
        if name is None:
            skipped.append({"video_file": video_file, "frame_idx": frame_idx, "reason": "video not on disk"})
            continue
        xy = [np.array([[r[f"{n}_x"], r[f"{n}_y"]] for n in BODY]) for _, r in g.iterrows()]
        drop = sleap_slp.duplicates(xy, DUPLICATE_PX)
        keep = [r for k, (_, r) in enumerate(g.iterrows()) if k not in drop]
        fish = [pl.Fish({k: (float(r[f"{v}_x"]), float(r[f"{v}_y"])) for k, v in MAP.items()
                         if np.isfinite(r[f"{v}_x"])}, origin="sleap_hand") for r in keep]
        if any(f.n_points() == 0 for f in fish):
            skipped.append({"video_file": video_file, "frame_idx": frame_idx,
                            "reason": "a fish without points"})
            continue
        body = np.concatenate([np.array([[r[f"{n}_x"], r[f"{n}_y"]] for n in BODY]) for r in keep])
        body = body[np.isfinite(body[:, 0])]
        rows.append({"recording": name, "frame_idx_sleap": int(frame_idx), "fish": fish, "body": body,
                     "n_instances": len(g), "n_duplicates": len(drop), "split": b2_mine.split_of(name)})
    out_rows = []
    for name in sorted({r["recording"] for r in rows}):
        rec = next(x for x in list_recordings() if x.name == name)
        t_frame = np.load(b1.TABLES / "frame_time" / f"{name}.npy")
        mine = [r for r in rows if r["recording"] == name]
        wanted = {r["frame_idx_sleap"] + k for r in mine for k in OFFSETS}
        imgs = dict(fr.read_seek(rec.video, wanted))
        for r in mine:
            grey = {k: grey_under(imgs[r["frame_idx_sleap"] + k], r["body"]) for k in OFFSETS}
            k = min(grey, key=grey.get)
            frame = r["frame_idx_sleap"] + k
            img = imgs[frame]
            split = r["split"]
            stem = f"{name}_{frame:06d}"
            pl.append_label(args.out / "labels.jsonl", pl.FrameLabel(stem, "done", r["fish"], labeller="sleap_v000"))
            (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            h, w = img.shape
            lines = pl.yolo_lines(r["fish"], w, h, b2_mine.PAD_BL * float(bl[name]))
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            out_rows.append({"split": split, "recording": name, "frame_idx_sleap": r["frame_idx_sleap"],
                             "offset": k, "frame": frame, "t_s": float(t_frame[frame]),
                             "phase": "dark" if t_frame[frame] < b1.LIGHT_ON_S else "light",
                             "n_fish": len(r["fish"]), "n_instances": r["n_instances"],
                             "n_duplicates": r["n_duplicates"],
                             **{f"grey_{o:+d}": grey[o] for o in OFFSETS},
                             "image": f"images/{split}/{stem}.jpg", "item_id": stem})
        print(f"{name}: {len(mine)} frames", flush=True)
    man = pd.DataFrame(out_rows).sort_values(["recording", "frame"])
    man.to_csv(args.out / "manifest.csv", index=False)
    man[["item_id", "image", "recording", "frame", "split", "phase", "t_s"]].to_csv(args.out / "index.csv", index=False)
    summary = {"source": str(args.slp), "frames_in_file": int(df["frame_id"].nunique()),
               "frames_written": len(man), "by_split": man["split"].value_counts().to_dict(),
               "by_recording": man["recording"].value_counts().to_dict(),
               "duplicates_dropped": int(man["n_duplicates"].sum()),
               "fish_written": int(man["n_fish"].sum()), "offsets": man["offset"].value_counts().to_dict(),
               "skipped": skipped, "node_map": MAP}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    yaml = pl.dataset_yaml(args.out, {"train": "images/train", "val": "images/train", "test": "images/test"})
    (args.out / "dataset.yaml").write_text(
        "# no validation split of its own: train together with mined_v1 (its images/val);\n"
        "# images/test = held-out recording 2022-05-11, never train on it\n" + yaml)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
