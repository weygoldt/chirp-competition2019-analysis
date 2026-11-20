"""Build a session for labelling chase onsets and offsets by eye (author's request, 2026-11-20).

Clips: ``--n`` chase candidates of a chase run (``c1_chases.py``), drawn at random
(at most ``--per-recording`` per recording; events already rendered as videos in
``<run>/videos/`` are left out, so the author has not seen them). Each clip runs
from ``BEFORE_S`` before the detected onset to ``AFTER_S`` after the detected offset,
every video frame (25 frames/s), cropped around both fish (their tracks chained
over the clip, ``c5_chase_timeseries.chain``). Two versions per clip: the raw video,
and the same with the two fish marked (orange / cyan dots on snout, spine1 and tail;
other detections grey). Nothing shows the detector's onset or offset; those, the
recording and the times are only in ``truth.csv`` (never served).

    .venv/bin/python experiments/video2022/c6_chase_label_session.py poses_r3 chase_v1
    experiments/video2022/serve_chase_labels.sh chase_v1

Output: ``$CHIRP_DATA/video2022/chases/labels/<name>/``: ``clips.json``, ``clips/``,
``truth.csv``; the labeller writes ``labels.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_assess_sleap as b1
import b3_predict as bp
import c1_chases as c1
import c5_chase_timeseries as c5
from recordings2022 import list_recordings

LABELS = c1.OUT / "labels"
BEFORE_S = 4.0
AFTER_S = 8.0
MARGIN_BL = 1.0
MAX_SIDE = 960
SEED = 20261121
LEADER, FOLLOWER, OTHER = (0, 140, 255), (255, 220, 0), (170, 170, 170)  # BGR


def crop_box(xs: np.ndarray, ys: np.ndarray, bl: float, w: int = 2048, h: int = 1536) -> tuple[int, int, int, int]:
    m = MARGIN_BL * bl
    x0, x1, y0, y1 = np.nanmin(xs) - m, np.nanmax(xs) + m, np.nanmin(ys) - m, np.nanmax(ys) + m
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = max((x1 - x0) / 2, 400), max((y1 - y0) / 2, 300)
    return int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(w, cx + hw)), int(min(h, cy + hh))


def writer(path: Path, w: int, h: int) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r",
           str(c1.FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-g", "10",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_clip(e: pd.Series, tr: pd.DataFrame, spans: pd.DataFrame, video: Path, t_frame: np.ndarray, bl: float,
                stride: int, out_raw: Path, out_marked: Path) -> dict:
    import cv2

    t0, t1 = e["t_start"] - BEFORE_S, e["t_end"] + AFTER_S
    frames = np.flatnonzero((t_frame >= t0) & (t_frame <= t1))
    lt = c5.chain(spans, int(e["leader"]), t0, t1, bl, {int(e["follower"])})
    ft = c5.chain(spans, int(e["follower"]), t0, t1, bl, set(lt))
    win = tr[tr["frame"].between(frames[0] - stride, frames[-1])]
    pair = win[win["track"].isin(lt + ft)]
    x0, y0, x1, y1 = crop_box(pair["spine1_x"].to_numpy(), pair["spine1_y"].to_numpy(), bl)
    scale = min(1.0, MAX_SIDE / max(x1 - x0, y1 - y0))
    ow, oh = int((x1 - x0) * scale) // 2 * 2, int((y1 - y0) * scale) // 2 * 2
    color = {k: LEADER for k in lt} | {k: FOLLOWER for k in ft}
    by_frame = {f: g for f, g in win.groupby("frame")}
    raw, marked = writer(out_raw, ow, oh), writer(out_marked, ow, oh)
    for f, img in bp.read_range(video, frames):
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        raw.stdin.write(cv2.resize(canvas[y0:y1, x0:x1], (ow, oh), interpolation=cv2.INTER_AREA).tobytes())
        fp = f - f % stride
        if fp in by_frame:
            for _, r in by_frame[fp].iterrows():
                c = color.get(r["track"], OTHER)
                pts = [(int(r[f"{n}_x"]), int(r[f"{n}_y"])) for n in ("snout", "spine1", "tail")]
                cv2.polylines(canvas, [np.array(pts, np.int32)], False, c, 2, cv2.LINE_AA)
                cv2.circle(canvas, pts[0], 7, c, -1, cv2.LINE_AA)
        marked.stdin.write(cv2.resize(canvas[y0:y1, x0:x1], (ow, oh), interpolation=cv2.INTER_AREA).tobytes())
    for w in (raw, marked):
        w.stdin.close()
        if w.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    return {"frame0": int(frames[0]), "n_frames": len(frames), "t0": float(t_frame[frames[0]]), "width": ow,
            "height": oh}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--per-recording", type=int, default=2)
    args = ap.parse_args()
    root = c1.OUT / args.run
    out = LABELS / args.name
    if out.exists():
        raise SystemExit(f"{out} exists; sessions are never rebuilt")
    stride = int(json.loads((root / "params.json").read_text())["stride"])
    ev = pd.read_csv(root / "events.csv")
    seen = root / "videos" / "rendered.csv"
    if seen.exists():
        ev = ev[~ev["event"].isin(pd.read_csv(seen)["event"])]
    trials = set(c1.pairing().index)
    ev = ev[ev["recording"].isin(trials)]
    rng = np.random.default_rng(SEED)
    sel = ev.iloc[rng.permutation(len(ev))].groupby("recording").head(args.per_recording).head(args.n)
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    videos = {r.name: r.video for r in list_recordings()}
    (out / "clips").mkdir(parents=True)
    clips, truth = [], []
    for rec, g in sel.groupby("recording"):
        tr = pd.read_parquet(root / rec / "tracks.parquet")
        spans = c5.spans_of(pd.read_parquet(root / rec / "kinematics.parquet"))
        t_frame = np.load(b1.TABLES / "frame_time" / f"{rec}.npy")
        for _, e in g.iterrows():
            cid = secrets.token_hex(4)
            info = render_clip(e, tr, spans, videos[rec], t_frame, float(bl[rec]), stride,
                               out / "clips" / f"{cid}.mp4", out / "clips" / f"{cid}_m.mp4")
            clips.append({"clip_id": cid, "video": f"clips/{cid}.mp4", "video_marked": f"clips/{cid}_m.mp4",
                          "fps": c1.FPS, "n_frames": info["n_frames"], "width": info["width"],
                          "height": info["height"]})
            truth.append({"clip_id": cid, "event": e["event"], "recording": rec, **info,
                          "detected_onset_s": e["t_start"] - info["t0"], "detected_offset_s": e["t_end"] - info["t0"],
                          "leader": int(e["leader"]), "follower": int(e["follower"])})
            print(f"{cid}: {e['event']} ({info['n_frames']} frames)", flush=True)
    order = rng.permutation(len(clips))
    (out / "clips.json").write_text(json.dumps([clips[i] for i in order], indent=1))
    pd.DataFrame(truth).to_csv(out / "truth.csv", index=False)
    print(f"{len(clips)} clips -> {out}")


if __name__ == "__main__":
    main()
