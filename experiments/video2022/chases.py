"""Tracks of the two fish from per-frame pose detections, and chase events (milestone C).

Input: detections of one recording (``b3_predict.load``: ``frame``, ``t_s``, ``det``,
``box_conf`` and ``<point>_x/_y`` for at least snout, spine1 and tail), predicted on
every ``stride``-th video frame. Distances are in body lengths (BL, the
recording's median SLEAP body length), speeds in BL/s.

1. :func:`clean`: detections with box confidence >= ``conf_min`` whose skeleton
   (snout -> spine1 -> spine2 -> spine3 -> tail) is at least ``min_len_bl`` long
   (a fish cut in two by a shelter gives short pieces).
2. :func:`track`: frame-to-frame linking (Hungarian assignment on the mean
   distance of snout, spine1 and tail to the track's pose, moved on by its
   velocity); a track ends after ``max_gap_s`` without a detection. Tracks are
   anonymous pieces (tracklets): which fish is which across tracks is not decided here.
3. :func:`kinematics`: per track the position of spine1 (just behind the head),
   smoothed over ``smooth_s``, its velocity and speed.
4. :func:`chase_frames`: for every pair of tracks at every frame both are seen:
   both speeds >= ``v_min``, distance <= ``d_max``, headings aligned
   (cos >= ``cos_min``), heads >= ``min_sep`` apart; the fish behind is the
   *follower* and must be on the *leader's path*: within ``r_path`` of where the
   leader was 1 frame to ``lag_max_s`` earlier.
5. :func:`events`: chase frames of a pair merged across gaps <= ``merge_gap_s``;
   the chase ends when the fish separate (:func:`separation_pieces`: the distance has
   grown more than ``sep_grow_bl`` above its minimum since the onset; the event ends at
   the last frame the distance was within ``sep_tol_bl`` of that minimum, i.e. when the
   gap began to open; author, 2026-11-20: offsets were too late). A run of chase
   frames counts if it lasts >= ``min_dur_s`` before the offset is moved (so the onsets
   are those of the plain rule); a second chase within the run, after the follower
   closed in again, counts if it lasts >= ``min_dur_s`` itself. Leader = the fish in
   front in most of the frames.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

BODY = ("snout", "spine1", "spine2", "spine3", "tail")
ANCHORS = ("snout", "spine1", "tail")



@dataclass(frozen=True)
class Params:
    conf_min: float = 0.5
    min_len_bl: float = 0.6
    max_gap_s: float = 0.5
    gate_bl: float = 0.6
    gate_bl_per_s: float = 3.0
    smooth_s: float = 0.4
    v_min: float = 1.0
    d_max: float = 3.0
    cos_min: float = 0.5
    min_sep: float = 0.5
    r_path: float = 0.4
    lag_max_s: float = 2.0
    merge_gap_s: float = 0.4
    min_dur_s: float = 1.0
    sep_grow_bl: float = 0.5
    sep_tol_bl: float = 0.15


DEFAULT = Params()


def skeleton_length(d: pd.DataFrame) -> np.ndarray:
    """Length (px) along snout -> spine1 -> spine2 -> spine3 -> tail (points that are present)."""
    pts = [n for n in BODY if f"{n}_x" in d]
    xy = np.stack([d[[f"{n}_x", f"{n}_y"]].to_numpy(float) for n in pts], axis=1)
    return np.hypot(*np.diff(xy, axis=1).transpose(2, 0, 1)).sum(axis=1)


def clean(d: pd.DataFrame, bl: float, p: Params = DEFAULT) -> pd.DataFrame:
    """Confident, full-length detections."""
    d = d[d["box_conf"] >= p.conf_min]
    return d[skeleton_length(d) >= p.min_len_bl * bl].reset_index(drop=True)


def _pose(row: pd.Series | dict) -> np.ndarray:
    return np.array([[row[f"{n}_x"], row[f"{n}_y"]] for n in ANCHORS], float)


def track(d: pd.DataFrame, bl: float, p: Params = DEFAULT) -> pd.DataFrame:
    """``d`` with a ``track`` column (detections that start no track are dropped: none are)."""
    from scipy.optimize import linear_sum_assignment

    d = d.sort_values(["frame", "det"]).reset_index(drop=True)
    poses = np.stack([d[[f"{n}_x", f"{n}_y"]].to_numpy(float) for n in ANCHORS], axis=1)  # (n, 3, 2)
    t_all = d["t_s"].to_numpy(float)
    track_of = np.full(len(d), -1, int)
    active: dict[int, dict] = {}  # id -> {"t", "pose", "v"}
    next_id = 0
    bounds = np.flatnonzero(np.r_[True, np.diff(d["frame"].to_numpy()) != 0, True])
    for a, b in itertools.pairwise(bounds):
        t = t_all[a]
        for k in [k for k, s in active.items() if t - s["t"] > p.max_gap_s]:
            del active[k]
        ids = list(active)
        idx = np.arange(a, b)
        if ids:
            cost = np.empty((len(ids), len(idx)))
            gate = np.empty(len(ids))
            for i, k in enumerate(ids):
                s = active[k]
                pred = s["pose"] + s["v"] * (t - s["t"])
                cost[i] = np.linalg.norm(poses[idx] - pred, axis=2).mean(axis=1)
                gate[i] = (p.gate_bl + p.gate_bl_per_s * (t - s["t"])) * bl
            rows, cols = linear_sum_assignment(np.where(cost <= gate[:, None], cost, 1e12))
            taken = set()
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] > gate[r]:
                    continue
                k, j = ids[r], idx[c]
                s = active[k]
                v_new = (poses[j] - s["pose"]) / (t - s["t"])
                s["v"] = 0.5 * s["v"] + 0.5 * v_new
                s["pose"], s["t"] = poses[j], t
                track_of[j] = k
                taken.add(j)
            free = [j for j in idx if j not in taken]
        else:
            free = list(idx)
        for j in free:
            active[next_id] = {"t": t, "pose": poses[j], "v": np.zeros((len(ANCHORS), 2))}
            track_of[j] = next_id
            next_id += 1
    return d.assign(track=track_of)


def kinematics(d: pd.DataFrame, bl: float, stride: int, fps: float = 25.0, p: Params = DEFAULT) -> pd.DataFrame:
    """Per track and detection: grid index ``i`` (= frame // stride), smoothed spine1 ``x``, ``y`` (px),
    velocity ``vx``, ``vy`` and ``speed`` (BL/s), snout ``sx``, ``sy``."""
    win = max(1, round(p.smooth_s * fps / stride))
    out = []
    for k, g in d.groupby("track", sort=False):
        g = g.sort_values("frame")
        x = g["spine1_x"].rolling(win, center=True, min_periods=1).mean().to_numpy(float)
        y = g["spine1_y"].rolling(win, center=True, min_periods=1).mean().to_numpy(float)
        t = g["t_s"].to_numpy(float)
        if len(g) >= 2:
            vx, vy = np.gradient(x, t), np.gradient(y, t)
        else:
            vx = vy = np.zeros(len(g))
        out.append(pd.DataFrame({"track": k, "frame": g["frame"].to_numpy(), "i": g["frame"].to_numpy() // stride,
                                 "t_s": t, "x": x, "y": y, "vx": vx / bl, "vy": vy / bl,
                                 "speed": np.hypot(vx, vy) / bl,
                                 "sx": g["snout_x"].to_numpy(float), "sy": g["snout_y"].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _on_grid(g: pd.DataFrame, i0: int, n: int) -> dict[str, np.ndarray]:
    out = {c: np.full(n, np.nan) for c in ("x", "y", "vx", "vy", "speed", "sx", "sy", "t_s")}
    j = g["i"].to_numpy() - i0
    inside = (j >= 0) & (j < n)
    for c in out:
        out[c][j[inside]] = g[c].to_numpy(float)[inside]
    return out


def path_distance(fx: np.ndarray, fy: np.ndarray, lx: np.ndarray, ly: np.ndarray, lag_max: int) -> tuple[np.ndarray,
                                                                                                         np.ndarray]:
    """For each sample: min over lags 1..lag_max of |follower(i) - leader(i - lag)| (px), and the lag (samples)."""
    n = len(fx)
    best = np.full(n, np.inf)
    lag = np.zeros(n, int)
    for k in range(1, lag_max + 1):
        dd = np.full(n, np.inf)
        dd[k:] = np.hypot(fx[k:] - lx[:-k], fy[k:] - ly[:-k])
        dd[np.isnan(dd)] = np.inf
        better = dd < best
        best[better], lag[better] = dd[better], k
    return best, lag


def chase_frames(kin: pd.DataFrame, bl: float, stride: int, fps: float = 25.0, p: Params = DEFAULT) -> pd.DataFrame:
    """Per pair of tracks (a < b) and grid sample both are seen: the pair features and ``chase`` (bool)."""
    if kin.empty:
        return pd.DataFrame()
    span = kin.groupby("track")["i"].agg(["min", "max"])
    lag_max = round(p.lag_max_s * fps / stride)
    tracks = span.sort_values("min").index.to_numpy()
    starts, ends = span.loc[tracks, "min"].to_numpy(), span.loc[tracks, "max"].to_numpy()
    by_track = dict(tuple(kin.groupby("track")))
    out = []
    for ia, a in enumerate(tracks):
        for ib in range(ia + 1, len(tracks)):
            if starts[ib] > ends[ia]:
                break
            b = tracks[ib]
            lo, hi = max(starts[ia], starts[ib]), min(ends[ia], ends[ib])
            if hi < lo:
                continue
            i0 = lo - lag_max
            n = hi - i0 + 1
            A, B = _on_grid(by_track[a], i0, n), _on_grid(by_track[b], i0, n)
            both = ~np.isnan(A["x"]) & ~np.isnan(B["x"])
            both[:lag_max] = False
            if not both.any():
                continue
            dist = np.hypot(A["x"] - B["x"], A["y"] - B["y"]) / bl
            sep = np.hypot(A["sx"] - B["sx"], A["sy"] - B["sy"]) / bl
            cos = (A["vx"] * B["vx"] + A["vy"] * B["vy"]) / (A["speed"] * B["speed"] + 1e-12)
            # a leads when it is ahead along the mean heading
            a_leads = ((A["x"] - B["x"]) * (A["vx"] + B["vx"]) + (A["y"] - B["y"]) * (A["vy"] + B["vy"])) > 0
            pd_ab, lag_ab = path_distance(B["x"], B["y"], A["x"], A["y"], lag_max)  # b follows a
            pd_ba, lag_ba = path_distance(A["x"], A["y"], B["x"], B["y"], lag_max)  # a follows b
            path = np.where(a_leads, pd_ab, pd_ba) / bl
            lag = np.where(a_leads, lag_ab, lag_ba) * stride / fps
            chase = (both & (A["speed"] >= p.v_min) & (B["speed"] >= p.v_min) & (dist <= p.d_max)
                     & (cos >= p.cos_min) & (sep >= p.min_sep) & (path <= p.r_path))
            j = np.flatnonzero(both)
            out.append(pd.DataFrame({
                "a": a, "b": b, "i": j + i0, "t_s": np.where(np.isnan(A["t_s"]), B["t_s"], A["t_s"])[j],
                "speed_a": A["speed"][j], "speed_b": B["speed"][j], "dist": dist[j], "sep": sep[j], "cos": cos[j],
                "leader": np.where(a_leads, a, b)[j], "path": path[j], "lag_s": lag[j], "chase": chase[j]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def separation_pieces(dist: np.ndarray, grow: float, tol: float) -> list[tuple[int, int]]:
    """Split a run of chase frames where the fish separate: (first, last) positions of the pieces.

    A piece ends once ``dist`` has grown more than ``grow`` above its running minimum
    since the piece began; its last frame is the last one before that with ``dist``
    within ``tol`` of the running minimum. The next piece begins once the distance stops
    growing (the follower closes in again); it becomes an event of its own only if it
    lasts long enough.
    """
    pieces = []
    start, n = 0, len(dist)
    while start < n:
        d = dist[start:]
        run_min = np.minimum.accumulate(d)
        over = np.flatnonzero(d > run_min + grow)
        if len(over) == 0:
            pieces.append((start, n - 1))
            break
        k = over[0]
        near = np.flatnonzero(d[:k] <= run_min[:k] + tol)
        pieces.append((start, start + int(near[-1])))
        closing = np.flatnonzero(np.diff(d[k:]) < 0)
        if len(closing) == 0:
            break
        start += int(k + closing[0])
    return pieces


def events(cf: pd.DataFrame, stride: int, fps: float = 25.0, p: Params = DEFAULT) -> pd.DataFrame:
    """Chase events: one row per event with times, leader / follower track and summary features."""
    cols = ["a", "b", "leader", "follower", "leader_share", "i_start", "i_end", "t_start", "t_end", "duration_s",
            "n_frames", "speed_leader", "speed_follower", "speed_max", "dist", "lag_s", "cos", "separated",
            "rechase", "t_end_run"]
    if cf.empty or not cf["chase"].any():
        return pd.DataFrame(columns=cols)
    gap = round(p.merge_gap_s * fps / stride)
    rows = []
    for (a, b), g in cf[cf["chase"]].groupby(["a", "b"]):
        i = g["i"].to_numpy()
        cut = np.flatnonzero(np.diff(i) > gap + 1) + 1
        t = g["t_s"].to_numpy()
        pieces = []
        for seg in np.split(np.arange(len(g)), cut):
            # a run qualifies on its full length (the onsets do not change); the separation rule moves the
            # offset of its first chase; later chases in the run (the follower closed in again) must last
            # min_dur_s on their own
            if t[seg[-1]] - t[seg[0]] + stride / fps < p.min_dur_s:
                continue
            parts = separation_pieces(g["dist"].to_numpy()[seg], p.sep_grow_bl, p.sep_tol_bl)
            for k, (f, la) in enumerate(parts):
                if k == 0 or t[seg[la]] - t[seg[f]] + stride / fps >= p.min_dur_s:
                    pieces.append((seg[f:la + 1], la < len(seg) - 1, k > 0, float(t[seg[-1]])))
        for seg, separated, rechase, t_end_run in pieces:
            s = g.iloc[seg]
            dur = (s["t_s"].iloc[-1] - s["t_s"].iloc[0]) + stride / fps
            share_a = float((s["leader"] == a).mean())
            leader, follower = (a, b) if share_a >= 0.5 else (b, a)
            sp_l = np.where(s["leader"] == a, s["speed_a"], s["speed_b"])
            sp_f = np.where(s["leader"] == a, s["speed_b"], s["speed_a"])
            rows.append({"a": a, "b": b, "leader": leader, "follower": follower,
                         "leader_share": max(share_a, 1 - share_a), "i_start": int(s["i"].iloc[0]),
                         "i_end": int(s["i"].iloc[-1]), "t_start": float(s["t_s"].iloc[0]),
                         "t_end": float(s["t_s"].iloc[-1]), "duration_s": float(dur), "n_frames": len(s),
                         "speed_leader": float(np.mean(sp_l)), "speed_follower": float(np.mean(sp_f)),
                         "speed_max": float(np.max(np.r_[sp_l, sp_f])), "dist": float(s["dist"].mean()),
                         "lag_s": float(s["lag_s"].median()), "cos": float(s["cos"].mean()),
                         "separated": bool(separated), "rechase": bool(rechase), "t_end_run": t_end_run})
    return pd.DataFrame(rows, columns=cols).sort_values("t_start", ignore_index=True)


def detect(d: pd.DataFrame, bl: float, stride: int, fps: float = 25.0, p: Params = DEFAULT) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detections -> (tracked detections, kinematics, chase frames, events)."""
    tr = track(clean(d, bl, p), bl, p)
    kin = kinematics(tr, bl, stride, fps, p)
    cf = chase_frames(kin, bl, stride, fps, p)
    return tr, kin, cf, events(cf, stride, fps, p)
