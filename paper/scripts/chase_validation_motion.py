"""Video motion around chase onsets / offsets: BORIS marks (all analysed chases) and author vs BORIS marks
(the re-labelled chases), for the timing-validation figure.

New step. Motion = mean absolute difference between consecutive video frames (the file BORIS was scored
on, decoded to 188 x 145 gray), frames at LED switches and the one after set to NaN (the sync LED is in
the picture). Each chase's trace is divided by its median over the decoded stretch (from 8 s before the
earlier onset to 8 s after the later offset) and aligned to each mark, +-8 s.

Video frames are numbered as in ``LED_frames.npy``: the ``_mod.avi`` files are h264 with B-frames in AVI,
and ffmpeg stamps frame k with k + delay (``session.json``: ``delay_frames``), so the decoder seeks to
(f + delay) / 25. Lags are in recording seconds (frames x the LED map's slope).

Inputs: the frozen review session (``paper_config.CHASE_VALIDATION``) for the videos and LED maps,
``chase_validation/pairs.csv``; outputs ``chase_validation/motion.npz``. Run after ``chase_validation.py``.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from paper_config import CHASE_VALIDATION, DATA_ROOT, PROCESSED

T_MAX_S = 3 * 60 * 60
PAD_S = 8.0
W, H = 188, 145
WORKERS = 6
SESSION = json.loads((CHASE_VALIDATION / "session.json").read_text())


def motion(rec: str, f0: int, f1: int) -> np.ndarray:
    info = SESSION["recordings"][rec]
    n = f1 - f0 + 1
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{(f0 + info['delay_frames']) / 25:.6f}",
                          "-i", str(DATA_ROOT / rec / info["video"]), "-frames:v", str(n),
                          "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    if len(fr) != n:
        raise RuntimeError(f"{rec} {f0}-{f1}: {len(fr)} frames decoded, {n} expected")
    me = np.full(n, np.nan)
    me[1:] = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    led = np.load(DATA_ROOT / rec / "LED_val.npy").astype(float)
    on = led > (np.percentile(led, 5) + np.percentile(led, 99.5)) / 2
    sw = np.flatnonzero(on[1:] != on[:-1]) + 1
    for s in sw[(sw >= f0) & (sw <= f1 + 1)]:
        me[max(0, s - f0 - 1): s - f0 + 2] = np.nan
    return me / np.nanmedian(me)


def job(item):
    key, rec, marks = item
    slope = SESSION["recordings"][rec]["led_map"]["slope"][0]
    pad = int(round(PAD_S / slope))
    f0 = max(0, min(marks.values()) - pad)
    f1 = min(SESSION["recordings"][rec]["n_frames"] - 1, max(marks.values()) + pad)
    me = motion(rec, f0, f1)
    lags = np.arange(-pad, pad + 1)
    out = {}
    for name, f in marks.items():
        i = f - f0 + lags
        row = np.full(len(lags), np.nan)
        ok = (i >= 0) & (i < len(me))
        row[ok] = me[i[ok]]
        out[name] = row
    return key, rec, lags * slope, out


def main() -> None:
    pairs = pd.read_csv(PROCESSED / "chase_validation" / "pairs.csv")
    items = []
    for r in pairs.itertuples():
        marks = {"boris_onset": int(r.boris_onset_frame), "boris_offset": int(r.boris_offset_frame)}
        if not r.onset_censored:
            marks["author_onset"] = int(r.onset_frame)
        if not r.offset_censored:
            marks["author_offset"] = int(r.offset_frame)
        items.append((f"pair:{r.chase_id}", r.recording, marks))
    for c in SESSION["chases"]:
        if 0 <= c["onset_rec_s"] and c["offset_rec_s"] < T_MAX_S:
            items.append((f"all:{c['id']}", c["recording"],
                          {"boris_onset": c["onset_frame"], "boris_offset": c["offset_frame"]}))
    print(f"{len(items)} chase windows ({len(pairs)} re-labelled pairs)", flush=True)
    grid = np.arange(-PAD_S, PAD_S + 1e-9, 0.04)
    acc: dict[str, list] = {}
    with ProcessPoolExecutor(WORKERS) as ex:
        for key, rec, lag_s, out in ex.map(job, items, chunksize=4):
            group = key.split(":")[0]
            for name, row in out.items():
                acc.setdefault(f"{group}_{name}", []).append(np.interp(grid, lag_s, row, left=np.nan, right=np.nan))
    arrays = {k: np.array(v) for k, v in acc.items()}
    np.savez_compressed(PROCESSED / "chase_validation" / "motion.npz", lags_s=grid, **arrays)
    print({k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
