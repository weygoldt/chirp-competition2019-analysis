"""Build a session for labelling chase onsets and offsets by eye (author's request, 2026-11-20).

Clips (``kind`` in ``truth.csv``):

- ``detected``: ``--n`` chase candidates of a chase run (``c1_chases.py``), drawn at
  random (at most ``--per-recording`` per recording). Events the author has seen
  (rendered in ``<run>/videos/`` or in an earlier session) are left out. The clip
  runs from ``--before`` before the detected onset to ``--after`` after the offset.
- ``close``: ``--close`` moments the detector did not call a chase but the two fish
  were close (<= ``CLOSE_BL``) and one swam >= ``CLOSE_SPEED`` BL/s, >= ``AWAY_S``
  from any detected chase (non-chases and missed chases); the clip runs from
  ``--before`` before to ``--after`` after that moment.

Every video frame (25 frames/s), cropped around both fish (their tracks chained
over the clip, ``c5_chase_timeseries.chain``). Two versions per clip: the raw video,
and the same with the two fish marked (orange / cyan dots on snout, spine1 and tail,
assigned at random, so the colours say nothing about roles; other detections grey).
Nothing shows the detector's times; those, the recording and the kind are only in
``truth.csv`` (never served).

    .venv/bin/python experiments/video2022/c6_chase_label_session.py poses_r3 chase_v1
    .venv/bin/python experiments/video2022/c6_chase_label_session.py poses_r3 chase_v2 --n 32 --close 18 \
        --before 6 --after 18 --max-side 1100
    experiments/video2022/serve_chase_labels.sh chase_v2

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
MARGIN_BL = 1.0
CLOSE_BL = 2.0
CLOSE_SPEED = 1.5
AWAY_S = 10.0
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
                stride: int, out_raw: Path, out_marked: Path, before: float, after: float, max_side: int,
                swap: bool = False) -> dict:
    import cv2

    t0, t1 = e["t_start"] - before, e["t_end"] + after
    frames = np.flatnonzero((t_frame >= t0) & (t_frame <= t1))
    lt = c5.chain(spans, int(e["leader"]), t0, t1, bl, {int(e["follower"])})
    ft = c5.chain(spans, int(e["follower"]), t0, t1, bl, set(lt))
    win = tr[tr["frame"].between(frames[0] - stride, frames[-1])]
    pair = win[win["track"].isin(lt + ft)]
    x0, y0, x1, y1 = crop_box(pair["spine1_x"].to_numpy(), pair["spine1_y"].to_numpy(), bl)
    scale = min(1.0, max_side / max(x1 - x0, y1 - y0))
    ow, oh = int((x1 - x0) * scale) // 2 * 2, int((y1 - y0) * scale) // 2 * 2
    c_l, c_f = (FOLLOWER, LEADER) if swap else (LEADER, FOLLOWER)
    color = {k: c_l for k in lt} | {k: c_f for k in ft}
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


def seen_events(root: Path) -> set[str]:
    """Events the author has already seen: rendered as videos, or in an earlier labelling session."""
    out = set()
    if (root / "videos" / "rendered.csv").exists():
        out |= set(pd.read_csv(root / "videos" / "rendered.csv")["event"])
    for t in LABELS.glob("*/truth.csv"):
        out |= set(pd.read_csv(t)["event"])
    return out


def close_moments(root: Path, recs: list[str], ev: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n`` moments with both fish close and one fast but no detected chase near, as pseudo events."""
    rows = []
    for rec in recs:
        cf = pd.read_parquet(root / rec / "chase_frames.parquet", columns=["a", "b", "t_s", "dist", "speed_a",
                                                                          "speed_b", "chase"])
        c = cf[(cf["dist"] <= CLOSE_BL) & (np.maximum(cf["speed_a"], cf["speed_b"]) >= CLOSE_SPEED) & ~cf["chase"]]
        e = ev[ev["recording"] == rec]
        if len(e) and len(c):
            t = c["t_s"].to_numpy()[:, None]
            gap = np.maximum(e["t_start"].to_numpy()[None] - t, t - e["t_end"].to_numpy()[None])
            c = c[(gap > AWAY_S).all(axis=1)]
        rows.append(c.assign(recording=rec))
    c = pd.concat(rows, ignore_index=True)
    c = c.iloc[rng.permutation(len(c))]
    per = int(np.ceil(n / len(recs))) + 1
    taken: list[pd.Series] = []
    for _, r in c.iterrows():
        mine = [x for x in taken if x["recording"] == r["recording"]]
        if len(mine) >= per or any(abs(x["t_s"] - r["t_s"]) < 60 for x in mine):
            continue
        taken.append(r)
        if len(taken) >= n:
            break
    return pd.DataFrame([{"event": f"{r['recording']}_close_{int(r['t_s'])}", "recording": r["recording"],
                          "t_start": r["t_s"], "t_end": r["t_s"], "leader": int(r["a"]), "follower": int(r["b"]),
                          "kind": "close"} for r in taken])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=20, help="detected chases")
    ap.add_argument("--close", type=int, default=0, help="moments with the fish close but no detected chase")
    ap.add_argument("--per-recording", type=int, default=2)
    ap.add_argument("--before", type=float, default=4.0)
    ap.add_argument("--after", type=float, default=8.0)
    ap.add_argument("--max-side", type=int, default=960)
    args = ap.parse_args()
    root = c1.OUT / args.run
    out = LABELS / args.name
    if out.exists():
        raise SystemExit(f"{out} exists; sessions are never rebuilt")
    stride = int(json.loads((root / "params.json").read_text())["stride"])
    all_ev = pd.read_csv(root / "events.csv")
    trials = set(c1.pairing().index)
    ev = all_ev[all_ev["recording"].isin(trials) & ~all_ev["event"].isin(seen_events(root))]
    rng = np.random.default_rng(SEED + len(args.name))
    sel = ev.iloc[rng.permutation(len(ev))].groupby("recording").head(args.per_recording).head(args.n)
    sel = sel.assign(kind="detected")
    if args.close:
        recs = sorted(r for r in trials if (root / r / "chase_frames.parquet").exists())
        sel = pd.concat([sel, close_moments(root, recs, all_ev, args.close, rng)], ignore_index=True)
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
                               out / "clips" / f"{cid}.mp4", out / "clips" / f"{cid}_m.mp4", args.before, args.after,
                               args.max_side, swap=bool(rng.random() < 0.5))
            clips.append({"clip_id": cid, "video": f"clips/{cid}.mp4", "video_marked": f"clips/{cid}_m.mp4",
                          "fps": c1.FPS, "n_frames": info["n_frames"], "width": info["width"],
                          "height": info["height"]})
            det = e["kind"] == "detected"
            truth.append({"clip_id": cid, "kind": e["kind"], "event": e["event"], "recording": rec, **info,
                          "detected_onset_s": e["t_start"] - info["t0"] if det else np.nan,
                          "detected_offset_s": e["t_end"] - info["t0"] if det else np.nan,
                          "moment_s": np.nan if det else e["t_start"] - info["t0"],
                          "leader": int(e["leader"]), "follower": int(e["follower"])})
            print(f"{cid}: {e['event']} ({info['n_frames']} frames)", flush=True)
    order = rng.permutation(len(clips))
    (out / "clips.json").write_text(json.dumps([clips[i] for i in order], indent=1))
    pd.DataFrame(truth).to_csv(out / "truth.csv", index=False)
    print(f"{len(clips)} clips -> {out}")


if __name__ == "__main__":
    main()
