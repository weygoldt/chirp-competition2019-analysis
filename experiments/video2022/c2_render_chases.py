"""Render chase events (``c1_chases.py``) as videos with the tracks drawn on (milestone C).

Events: ``--n`` drawn at random (fixed seed) from ``events.csv``, at most
``--per-recording`` per recording, so that the videos show typical events, not
the best ones (``--events`` picks given event ids instead).

Each video: the event from ``PAD_S`` before to ``PAD_S`` after, every video frame
(25 frames/s, real time), cropped around both fish. Drawn: the skeletons of the
tracked fish (the model ran on every second frame; odd frames show the previous
frame's pose), the leader orange and the follower cyan, other tracks grey; the
last ``TRAIL_S`` of each fish's path (spine1); a header with recording, recording
time, event id; a badge "CHASE" while the event lasts.

    .venv/bin/python experiments/video2022/c2_render_chases.py poses_r3 [--n 20]
    .venv/bin/python experiments/video2022/c2_render_chases.py poses_r3_sep --match poses_r3

``--match OLD``: render the events of this run that correspond to the videos of run
``OLD`` (same recording and pair of tracks, onset within 0.2 s), with a timeline
under the video: grey = OLD's event, red = this run's event, white = now.

Output: ``$CHIRP_DATA/video2022/chases/<run>/videos/<event>.mp4``, ``<event>.jpg``
(a still from the middle of the event), ``rendered.csv``, ``stills.jpg`` (all stills).
"""

from __future__ import annotations

import argparse
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
import pose_labels as pl
from recordings2022 import list_recordings

PAD_S = 1.5
TRAIL_S = 1.5
MARGIN_BL = 0.75
MAX_SIDE = 960
SEED = 20261120
LEADER, FOLLOWER, OTHER = (0, 140, 255), (255, 220, 0), (170, 170, 170)  # BGR: orange, cyan, grey


def pick(ev: pd.DataFrame, n: int, per_recording: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    order = ev.iloc[rng.permutation(len(ev))]
    taken = order.groupby("recording").head(per_recording)
    return taken.head(n).sort_values(["recording", "t_start"])


def crop_box(tr: pd.DataFrame, bl: float, w: int, h: int) -> tuple[int, int, int, int]:
    xs = np.concatenate([tr[f"{n}_x"].to_numpy() for n in ("snout", "spine1", "tail")])
    ys = np.concatenate([tr[f"{n}_y"].to_numpy() for n in ("snout", "spine1", "tail")])
    m = MARGIN_BL * bl
    x0, x1, y0, y1 = xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = max((x1 - x0) / 2, 320), max((y1 - y0) / 2, 240)
    x0, x1 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y0, y1 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    return x0, y0, x1, y1


def draw_fish(img: np.ndarray, row: pd.Series, color: tuple[int, int, int], thick: int) -> None:
    import cv2

    pts = {n: (int(row[f"{n}_x"]), int(row[f"{n}_y"])) for n in pl.SKELETON if f"{n}_x" in row
           and n not in ("left_fin", "right_fin")}
    for a, b in pl.EDGES:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], color, thick, cv2.LINE_AA)
    for xy in pts.values():
        cv2.circle(img, xy, thick + 3, color, -1, cv2.LINE_AA)
    cv2.circle(img, pts["snout"], thick + 6, (255, 255, 255), 2, cv2.LINE_AA)


def px(t: float, t0: float, t1: float, width: int) -> int:
    """x pixel of time t on a timeline from t0 to t1 across ``width`` pixels."""
    return round((t - t0) / (t1 - t0) * (width - 1))


def render(event: pd.Series, tr: pd.DataFrame, video: Path, t_frame: np.ndarray, bl: float, stride: int,
           out: Path, old: tuple[float, float] | None = None) -> np.ndarray:
    """Write the event's video; return the still from the middle of the event."""
    import cv2

    t0, t1 = event["t_start"] - PAD_S, event["t_end"] + PAD_S
    if old is not None:
        t0, t1 = min(t0, old[0] - PAD_S), max(t1, old[1] + PAD_S)
    frames = np.flatnonzero((t_frame >= t0) & (t_frame <= t1))
    win = tr[(tr["frame"] >= frames[0] - stride) & (tr["frame"] <= frames[-1])]
    pair = win[win["track"].isin([event["leader"], event["follower"]])]
    by_frame = {f: g for f, g in win.groupby("frame")}
    x0, y0, x1, y1 = crop_box(pair, bl, 2048, 1536)
    scale = min(1.0, MAX_SIDE / max(x1 - x0, y1 - y0))
    ow, oh = int((x1 - x0) * scale) // 2 * 2, int((y1 - y0) * scale) // 2 * 2
    head = 34
    foot = 18 if old is not None else 0
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh + head + foot}",
           "-r", str(c1.FPS), "-i", "-", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    mid = (event["t_start"] + event["t_end"]) / 2
    still, best = None, np.inf
    color = {event["leader"]: LEADER, event["follower"]: FOLLOWER}
    trail_n = round(TRAIL_S * c1.FPS / stride)
    paths = {k: g.sort_values("frame") for k, g in win.groupby("track")}
    for f, img in bp.read_range(video, frames):
        t = t_frame[f]
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        fp = f - f % stride
        for k, g in paths.items():
            past = g[(g["frame"] <= fp) & (g["frame"] > fp - trail_n * stride)]
            if len(past) >= 2:
                pts = past[["spine1_x", "spine1_y"]].to_numpy(np.int32)
                cv2.polylines(canvas, [pts], False, color.get(k, OTHER), 3, cv2.LINE_AA)
        if fp in by_frame:
            for _, row in by_frame[fp].iterrows():
                c = color.get(row["track"], OTHER)
                draw_fish(canvas, row, c, 3 if row["track"] in color else 2)
        crop = cv2.resize(canvas[y0:y1, x0:x1], (ow, oh), interpolation=cv2.INTER_AREA)
        frame = np.zeros((oh + head + foot, ow, 3), np.uint8)
        frame[head:head + oh] = crop
        if old is not None:
            y = oh + head
            cv2.rectangle(frame, (px(old[0], t0, t1, ow), y + 2), (px(old[1], t0, t1, ow), y + 7), (150, 150, 150), -1)
            cv2.rectangle(frame, (px(event["t_start"], t0, t1, ow), y + 10), (px(event["t_end"], t0, t1, ow), y + 15),
                          (40, 40, 230), -1)
            cv2.line(frame, (px(t, t0, t1, ow), y), (px(t, t0, t1, ow), y + 17), (255, 255, 255), 2)
        mm, ss = divmod(t, 60)
        cv2.putText(frame, f"{event['recording']}  t = {int(mm):03d}:{ss:04.1f}  {event['event']}", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        inside = event["t_start"] <= t <= event["t_end"]
        label, col = ("CHASE", (40, 40, 230)) if inside else (("before" if t < event["t_start"] else "after"),
                                                              (90, 90, 90))
        cv2.rectangle(frame, (ow - 92, 5), (ow - 6, 29), col, -1)
        cv2.putText(frame, label, (ow - 84, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        proc.stdin.write(frame.tobytes())
        if abs(t - mid) < best:
            best, still = abs(t - mid), frame.copy()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")
    return still


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """``img`` scaled to fit a w x h box, centred on black."""
    import cv2

    sc = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return out


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--per-recording", type=int, default=2)
    ap.add_argument("--events", nargs="*", help="event ids instead of a random draw")
    ap.add_argument("--match", help="render the events matching the videos of this earlier run")
    args = ap.parse_args()
    root = c1.OUT / args.run
    ev = pd.read_csv(root / "events.csv")
    old_window = {}
    if args.match:
        old = pd.read_csv(c1.OUT / args.match / "videos" / "rendered.csv")
        rows = []
        for o in old.itertuples():
            c = ev[(ev["recording"] == o.recording) & (ev["a"] == o.a) & (ev["b"] == o.b)
                   & ((ev["t_start"] - o.t_start).abs() <= 0.2)]
            if len(c):
                rows.append(c.index[0])
                old_window[c.iloc[0]["event"]] = (o.t_start, o.t_end)
            else:
                print(f"{o.event}: no matching event in {args.run}")
        sel = ev.loc[rows]
    elif args.events:
        sel = ev[ev["event"].isin(args.events)]
    else:
        sel = pick(ev, args.n, args.per_recording)
    stride = int(pd.read_json(root / "params.json", typ="series")["stride"])
    bl = pd.read_csv(b1.TABLES / "recordings.csv").set_index("recording")["body_length_px"]
    videos = {r.name: r.video for r in list_recordings()}
    out = root / "videos"
    out.mkdir(exist_ok=True)
    stills = []
    for rec, g in sel.groupby("recording"):
        tr = pd.read_parquet(root / rec / "tracks.parquet")
        t_frame = np.load(b1.TABLES / "frame_time" / f"{rec}.npy")
        for _, e in g.iterrows():
            still = render(e, tr, videos[rec], t_frame, float(bl[rec]), stride, out / f"{e['event']}.mp4",
                           old_window.get(e["event"]))
            cv2.imwrite(str(out / f"{e['event']}.jpg"), still, [cv2.IMWRITE_JPEG_QUALITY, 90])
            stills.append(fit(still, 480, 400))
            print(f"{e['event']}: {e['duration_s']:.1f} s", flush=True)
    sel.to_csv(out / "rendered.csv", index=False)
    if stills:
        tiles = list(stills)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / "stills.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(sel)} videos -> {out}")


if __name__ == "__main__":
    main()
