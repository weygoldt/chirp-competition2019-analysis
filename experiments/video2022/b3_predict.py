"""Run a pose model over the dark phase of the 2022 videos (milestone B3, the full run).

Frames: recording time 0-3 h (dark, IR light), every ``--stride``-th video frame
(default 2: 12.5 frames/s). The video is decoded in a second thread while the
model runs. Every detection with box confidence >= ``--conf`` (default 0.2) is
stored with all seven keypoints and their confidences, so thresholds can be chosen
afterwards without running the model again.

The run is resumable: each recording is cut into chunks of ``--chunk`` predicted
frames; a chunk is written as one parquet file when it is complete, and chunks
that exist are skipped. A recording is complete when ``<rec>/done.json`` exists.

    .venv/bin/python experiments/video2022/b3_predict.py poses_r3 --model weights/pose7_r3_l.pt

Output: ``$CHIRP_DATA/video2022/poses/<run>/``: ``meta.json`` (model, its sha256,
settings), ``<rec>/part_<first frame>.parquet`` (columns ``frame``, ``t_s``, ``det``,
``box_conf``, ``<point>_x``, ``<point>_y``, ``<point>_c`` for the seven points),
``<rec>/done.json``. :func:`load` reads one recording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1
import pose_labels as pl
import pose_model as pm
from paths import DATA
from recordings2022 import LIGHT_ON_S, list_recordings

ROOT = DATA / "video2022" / "poses"
# pair 3 first (most chirps), then the rest in date order
FIRST = ("2022-06-20-10_00", "2022-06-22-10_00", "2022-06-24-10_00")


def dark_frames(recording: str, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices of the dark phase (recording time 0 to LIGHT_ON_S) with frame % stride == 0, and their times."""
    t = np.load(b1.TABLES / "frame_time" / f"{recording}.npy")
    f = np.flatnonzero((t >= 0) & (t < LIGHT_ON_S))
    f = f[f % stride == 0]
    return f, t[f]


def read_range(video: Path, frames: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame, grey image) for the wanted frames (increasing): one seek to the first, then sequential decoding."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open {video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
        wanted = set(int(f) for f in frames)
        for f in range(int(frames[0]), int(frames[-1]) + 1):
            if f in wanted:
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f"{video}: cannot read frame {f}")
                yield f, img[..., 0].copy()
            elif not cap.grab():
                raise RuntimeError(f"{video}: video ends before frame {f}")
    finally:
        cap.release()


def rows_of(frame: int, t_s: float, preds: list[pm.PredFish]) -> list[dict]:
    out = []
    for j, p in enumerate(preds):
        r = {"frame": frame, "t_s": t_s, "det": j, "box_conf": p.box_conf}
        for n in pl.SKELETON:
            r[f"{n}_x"], r[f"{n}_y"] = p.points[n]
            r[f"{n}_c"] = p.kpt_conf[n]
        out.append(r)
    return out


def predict_chunk(model, video: Path, frames: np.ndarray, times: np.ndarray, batch: int, imgsz: int,
                  conf: float) -> pd.DataFrame:
    """Detections in the given frames; decoding runs in a thread, the model in this one."""
    q: queue.Queue = queue.Queue(maxsize=4)
    err: list[BaseException] = []

    def producer() -> None:
        try:
            buf = []
            for f, img in read_range(video, frames):
                buf.append((f, img))
                if len(buf) == batch:
                    q.put(buf)
                    buf = []
            if buf:
                q.put(buf)
        except BaseException as e:
            err.append(e)
        finally:
            q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    t_of = dict(zip(frames.tolist(), times.tolist(), strict=True))
    rows, n_frames = [], 0
    while (item := q.get()) is not None:
        preds = pm.predict(model, [img for _, img in item], imgsz=imgsz, conf=conf, batch=len(item))
        for (f, _), p in zip(item, preds, strict=True):
            rows += rows_of(f, t_of[f], p)
        n_frames += len(item)
    th.join()
    if err:
        raise err[0]
    if n_frames != len(frames):
        raise RuntimeError(f"{video}: {n_frames} of {len(frames)} frames decoded")
    cols = ["frame", "t_s", "det", "box_conf"] + [f"{n}_{c}" for n in pl.SKELETON for c in "xyc"]
    d = pd.DataFrame(rows, columns=cols)
    return d.astype({c: "float32" for c in cols[3:]} | {"frame": "int64", "det": "int16"})


def load(run: str, recording: str) -> pd.DataFrame:
    """All detections of one recording (raises if the recording is not complete)."""
    rec_dir = ROOT / run / recording
    if not (rec_dir / "done.json").exists():
        raise FileNotFoundError(f"{rec_dir}: not complete")
    parts = sorted(rec_dir.glob("part_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values(["frame", "det"],
                                                                                       ignore_index=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--model", required=True, help="weights (.pt), absolute or relative to $CHIRP_DATA/video2022")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=4500, help="predicted frames per part file")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--recordings", nargs="*", help="default: all 19")
    ap.add_argument("--max-chunks", type=int, default=0, help="stop after this many chunks (testing)")
    args = ap.parse_args()
    weights = Path(args.model) if Path(args.model).is_absolute() else DATA / "video2022" / args.model
    out = ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": str(weights), "sha256": sha256(weights), "stride": args.stride, "chunk": args.chunk,
            "imgsz": args.imgsz, "conf": args.conf, "dark_phase_s": [0.0, LIGHT_ON_S]}
    if (out / "meta.json").exists():
        old = json.loads((out / "meta.json").read_text())
        if any(old[k] != meta[k] for k in ("sha256", "stride", "chunk", "imgsz", "conf")):
            raise SystemExit(f"{out}: exists with other settings: {old}")
    else:
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    recs = {r.name: r for r in list_recordings()}
    order = [n for n in FIRST if n in recs] + [n for n in sorted(recs) if n not in FIRST]
    if args.recordings:
        order = [n for n in order if n in args.recordings]
    model = pm.load(weights)
    n_done_chunks = 0
    for name in order:
        rec_dir = out / name
        if (rec_dir / "done.json").exists():
            continue
        rec_dir.mkdir(exist_ok=True)
        frames, times = dark_frames(name, args.stride)
        starts = range(0, len(frames), args.chunk)
        for s in starts:
            f, t = frames[s:s + args.chunk], times[s:s + args.chunk]
            part = rec_dir / f"part_{int(f[0]):06d}.parquet"
            if part.exists():
                continue
            t0 = time.time()
            d = predict_chunk(model, recs[name].video, f, t, args.batch, args.imgsz, args.conf)
            tmp = part.with_suffix(".tmp")
            d.to_parquet(tmp, index=False)
            tmp.rename(part)
            dt = time.time() - t0
            print(f"{name} {part.name}: {len(f)} frames, {len(d)} detections, {len(f) / dt:.1f} frames/s",
                  flush=True)
            n_done_chunks += 1
            if args.max_chunks and n_done_chunks >= args.max_chunks:
                return
        n_parts = len(list(rec_dir.glob("part_*.parquet")))
        if n_parts == len(starts):
            (rec_dir / "done.json").write_text(json.dumps({"frames": len(frames), "parts": n_parts}))
            print(f"{name}: done", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
