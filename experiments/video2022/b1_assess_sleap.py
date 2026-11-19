"""Milestone B1: how good is the SLEAP export of the 2022 competition videos?

Per recording: how many fish SLEAP finds per frame (0/1/2), its instance and
node scores, how long its tracks are, implausible jumps (switch proxy), how
long and where fish drop out, example frames (occlusion vs missed detection),
and the video -> recording time map from the LED flashes.

Stages, in this order (outputs under ``$CHIRP_DATA/video2022/b1_sleap``)::

    PY=/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python
    $PY experiments/video2022/b1_assess_sleap.py tables      # tables/ (reads the NIX exports, ~3 min)
    $PY experiments/video2022/b1_assess_sleap.py background  # one frame per video + arena objects
    $PY experiments/video2022/b1_assess_sleap.py locations   # dropout positions -> tube / block / wall / open
    $PY experiments/video2022/b1_assess_sleap.py examples    # picks and renders the example frames
    $PY experiments/video2022/b1_assess_sleap.py figures     # figures/
    $PY experiments/video2022/b1_assess_sleap.py report      # report.md (+ examples/judgements.csv)

``examples/judgements.csv`` is filled in by hand (visual judgement of each
rendered example frame); ``report`` counts it.

Definitions used throughout:

- recording time: seconds of the electric recording (sample index / 20 kHz),
  from ``led_alignment.fit_led_alignment`` on the LED pairs; dark = 0-10800 s
  (meta.csv ``light_sec``), light = 10800 s to the end of the recording;
  "before"/"after" = video frames outside the electric recording.
- body length (BL): per recording, the median head-centre-tail polyline length
  of confident instances (all three nodes, node scores >= 0.5).
- step: displacement of a track between two consecutive frames, the mean over
  the nodes found in both frames (so a node appearing or vanishing does not
  move the fish). A jump is a step > 0.5 BL in one frame (40 ms, i.e. faster
  than 12.5 BL/s).
- dropout: a run of consecutive frames with fewer than two instances.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "paper" / "scripts"))
sys.path.insert(0, str(HERE.parent))

from led_alignment import LedAlignmentError, _nn_resid, check_led_alignment, fit_led_alignment  # noqa: E402
from paths import DATA  # noqa: E402
from recordings2022 import LIGHT_ON_S, Recording2022, list_recordings  # noqa: E402
from sleap_nix import instances_per_frame, read_instances, read_video_info  # noqa: E402

OUT = DATA / "video2022" / "b1_sleap"
TABLES = OUT / "tables"
NODES = ("head", "center", "tail")
PHASES = ("before", "dark", "light", "after")
JUMP_BL = 0.5  # step > 0.5 body lengths in one frame = implausible jump
BIG_JUMP_BL = 1.0
SWAP_BL = 0.5  # a jump that lands within 0.5 BL of the other fish's previous position
CONF_NODE = 0.5  # node score of a "confident" node
CONF_INSTANCE = 0.5  # instance score of a "confident" instance
MIN_DROPOUT_S = 1.0  # dropouts shorter than this are flicker


# ----------------------------------------------------------------------------- helpers
def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (inclusive) indices of the runs of True in a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def phase_codes(t: np.ndarray, rec_dur: float, light_on: float = LIGHT_ON_S) -> np.ndarray:
    """0 before the recording, 1 dark, 2 light, 3 after the recording."""
    return np.select([t < 0, t < light_on, t < rec_dur], [0, 1, 2], 3).astype(np.int8)


def xy(df: pd.DataFrame, node: str) -> np.ndarray:
    return df[[f"{node}_x", f"{node}_y"]].to_numpy()


def body_length_px(df: pd.DataFrame) -> float:
    ok = np.all([df[f"{n}_score"].to_numpy() >= CONF_NODE for n in NODES], axis=0)
    ok &= df["instance_score"].to_numpy() >= CONF_INSTANCE
    d = df[ok]
    L = np.linalg.norm(xy(d, "head") - xy(d, "center"), axis=1) + np.linalg.norm(
        xy(d, "center") - xy(d, "tail"), axis=1)
    return float(np.median(L))


def add_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Per instance: frames since the same track's previous instance and the step (px).

    Rows must be sorted by track, frame. ``gap_frames`` = frame difference to
    the previous instance of the track (NaN for a track's first instance);
    ``step_px`` = mean displacement over the nodes found in both instances, only
    for consecutive frames (gap 1).
    """
    same = np.r_[False, df["track"].to_numpy()[1:] == df["track"].to_numpy()[:-1]]
    gap = np.r_[np.nan, np.diff(df["frame"].to_numpy()).astype(float)]
    gap[~same] = np.nan
    disp = []
    for n in NODES:
        p = xy(df, n)
        d = np.r_[np.nan, np.linalg.norm(p[1:] - p[:-1], axis=1)]
        disp.append(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # no node in common -> NaN
        step = np.nanmean(np.vstack(disp), axis=0) if len(df) else np.zeros(0)
    step[gap != 1] = np.nan
    return df.assign(gap_frames=gap, step_px=step)


def frame_slots(df: pd.DataFrame, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the (up to two) instances: row index (-1 = none) and centroid xy."""
    idx = np.full((n_frames, 2), -1, np.int64)
    rank = df.groupby("frame").cumcount().to_numpy()
    if rank.max() > 1:
        raise ValueError("more than two instances in a frame")
    idx[df["frame"].to_numpy(), rank] = np.arange(len(df))
    c = df[["centroid_x", "centroid_y"]].to_numpy()
    pos = np.where(idx[..., None] >= 0, c[np.clip(idx, 0, None)], np.nan)
    return idx, pos


# ----------------------------------------------------------------------------- LED map
def led_map(rec: Recording2022, n_frames: int) -> tuple[dict, np.ndarray]:
    """LED map of one recording: summary row and the recording time of every frame."""
    lf, li, sr = rec.led_frames(), rec.led_idx(), rec.sample_rate()
    a = fit_led_alignment(lf, li, sr)
    try:
        check_led_alignment(a, rec.name)
        status = "ok"
    except LedAlignmentError as e:
        status = f"refused: {e}"
    t = a.to_grid(np.arange(n_frames))
    gt = li / sr
    # whole-period check: do the 2-s pauses of the LED train line up?
    g_pause = gt[1:][np.diff(gt) > 1.5]
    v_pause = a.to_grid(lf[1:][np.diff(lf) > 1.5 * a.fps_effective])
    r_g, _ = _nn_resid(g_pause, v_pause) if len(v_pause) > 1 else (np.full(len(g_pause), np.inf), None)
    resid, _ = _nn_resid(a.to_grid(lf), gt)
    row = dict(
        recording=rec.name, led_method=a.method, led_status=status, sample_rate=sr,
        n_led_video=a.n_led_video, n_led_grid=a.n_led_grid, n_led_matched=a.n_matched,
        led_resid_med_ms=a.resid_med * 1e3, led_resid_p95_ms=a.resid_p95 * 1e3,
        led_resid_max_ms=float(np.max(np.abs(resid))) * 1e3,
        n_steps=len(a.steps), step_sizes_ms=";".join(f"{s[1] * 1e3:.0f}" for s in a.steps),
        fps_effective=a.fps_effective,
        grid_pauses=len(g_pause), grid_pauses_matched=int(np.sum(np.abs(r_g) < 0.2)),
        t_first_frame_s=float(t[0]), t_last_frame_s=float(t[-1]),
    )
    (TABLES / "led_resid").mkdir(exist_ok=True)
    (TABLES / "frame_time").mkdir(exist_ok=True)
    np.save(TABLES / "led_resid" / f"{rec.name}.npy", np.c_[a.to_grid(lf), resid])  # (t_s, residual_s) per video LED
    np.save(TABLES / "frame_time" / f"{rec.name}.npy", t)  # recording time (s) of every video frame
    return row, t


def light_on_from_led_roi(rec: Recording2022, t_frame: np.ndarray) -> float:
    """Recording time at which the baseline brightness of the LED region steps up.

    ``LED_val.npy`` is the brightness of the LED's image region in every frame;
    between flashes it shows the room light. Baseline = per-second minimum,
    5-s running median; light on = first second after which the baseline stays
    above the midpoint of the dark (first 2 h) and light (last 20 min) levels
    for 5 min. NaN if the two levels differ by less than 5 grey values.
    """
    v = np.load(rec.interrim / "LED_val.npy")
    n = len(v) // 25
    base = pd.Series(v[: n * 25].reshape(n, 25).min(1)).rolling(5, center=True, min_periods=1).median().to_numpy()
    np.save(TABLES / "frame_time" / f"{rec.name}_led_roi_baseline.npy", base)  # per video second
    tb = t_frame[np.arange(n) * 25 + 12] if len(t_frame) >= n * 25 else np.interp(np.arange(n) * 25 + 12, np.arange(len(t_frame)), t_frame)
    dark = np.median(base[(tb > 0) & (tb < 7200)])
    light = np.median(base[tb > tb[-1] - 1200])
    if light - dark < 5:
        return float("nan")
    above = base > 0.5 * (dark + light)
    stay = pd.Series(above).rolling(300).min().shift(-299).fillna(0).to_numpy().astype(bool)
    k = np.flatnonzero(stay & (tb > 0))
    return float(tb[k[0]]) if len(k) else float("nan")


# ----------------------------------------------------------------------------- per recording
def process(rec: Recording2022) -> dict:
    info = read_video_info(rec.nix)
    fps, nfr = info.fps, info.n_frames
    led_row, t_frame = led_map(rec, nfr)
    rec_dur = rec.recording_duration_s()
    light_on_measured = light_on_from_led_roi(rec, t_frame)
    phase_f = phase_codes(t_frame, rec_dur)

    df = read_instances(rec.nix)
    counts = instances_per_frame(df["frame"].to_numpy(), nfr)
    bl = body_length_px(df)

    # ---- instance table with derived columns
    df = df.sort_values(["track", "frame"], kind="stable").reset_index(drop=True)
    df = add_steps(df)
    df["n_in_frame"] = counts[df["frame"].to_numpy()].astype(np.int8)
    df["t_s"] = t_frame[df["frame"].to_numpy()]
    df["phase"] = phase_f[df["frame"].to_numpy()]
    df["n_nodes"] = np.sum([np.isfinite(df[f"{n}_score"]) for n in NODES], axis=0).astype(np.int8)
    df["jump"] = df["step_px"] > JUMP_BL * bl
    df["big_jump"] = df["step_px"] > BIG_JUMP_BL * bl
    # swap proxy: the jump lands within SWAP_BL of the other instance's position one frame earlier
    df = df.sort_values(["frame", "track"], kind="stable").reset_index(drop=True)
    slot_idx, slot_pos = frame_slots(df, nfr)
    j = np.flatnonzero(df["jump"].to_numpy())
    swap = np.zeros(len(df), bool)
    if len(j):
        f = df["frame"].to_numpy()[j]
        prev_idx, prev_pos = slot_idx[f - 1], slot_pos[f - 1]
        tr_prev = np.where(prev_idx >= 0, df["track"].to_numpy()[np.clip(prev_idx, 0, None)], -1)
        other = (prev_idx >= 0) & (tr_prev != df["track"].to_numpy()[j][:, None])
        dist = np.linalg.norm(prev_pos - df[["centroid_x", "centroid_y"]].to_numpy()[j][:, None, :], axis=2)
        swap[j] = np.any(other & (dist < SWAP_BL * bl), axis=1)
    df["swap_like"] = swap
    # pair distance in 2-instance frames (centroids), for the example selection and B2
    pair_d = np.linalg.norm(slot_pos[:, 0] - slot_pos[:, 1], axis=1)
    df["pair_dist_px"] = pair_d[df["frame"].to_numpy()]
    df.astype({c: "float32" for c in df.columns if df[c].dtype == np.float64}).to_parquet(
        TABLES / "instances" / f"{rec.name}.parquet", index=False)

    # ---- coverage
    row = dict(recording=rec.name, video=rec.video.name, nix=rec.nix.name, nix_video=info.name,
               n_frames=nfr, fps=fps, video_h=nfr / fps / 3600, rec_dur_s=rec_dur,
               light_on_assumed_s=LIGHT_ON_S, light_on_measured_s=light_on_measured,
               n_instances=len(df), n_tracks=int(df["track"].nunique()), body_length_px=bl,
               first_instance_frame=int(df["frame"].min()), last_instance_frame=int(df["frame"].max()))
    row.update(led_row)
    row["video_covers_dark"] = bool(t_frame[0] <= 0 and t_frame[-1] >= LIGHT_ON_S)
    row["video_covers_recording"] = bool(t_frame[0] <= 0 and t_frame[-1] >= rec_dur)
    for label, sel in [("all", np.ones(nfr, bool))] + [(p, phase_f == k) for k, p in enumerate(PHASES)]:
        c = counts[sel]
        row[f"frames_{label}"] = int(sel.sum())
        for k in (0, 1, 2):
            row[f"share{k}_{label}"] = float(np.mean(c == k)) if len(c) else np.nan
        row[f"share3plus_{label}"] = float(np.mean(c > 2)) if len(c) else np.nan

    # coverage per recording-time minute
    minute = np.floor(t_frame / 60).astype(int)
    cov = pd.DataFrame(dict(minute=minute, c0=counts == 0, c1=counts == 1, c2=counts == 2)).groupby("minute").mean()
    cov.insert(0, "recording", rec.name)
    cov = cov.reset_index()

    # ---- tracklets
    g = df.groupby("track")
    trk = pd.DataFrame(dict(
        n_instances=g.size(), first_frame=g["frame"].min(), last_frame=g["frame"].max(),
        n_gaps=g["gap_frames"].apply(lambda s: int((s > 1).sum())),
        gap_frames=g["gap_frames"].apply(lambda s: float((s[s > 1] - 1).sum())),
        max_gap_frames=g["gap_frames"].max(),
        n_jumps=g["jump"].sum(), n_big_jumps=g["big_jump"].sum(), n_swap_like=g["swap_like"].sum(),
        median_instance_score=g["instance_score"].median(),
        share_two_fish=g["n_in_frame"].apply(lambda s: float((s == 2).mean())),
    )).reset_index()
    trk["span_frames"] = trk["last_frame"] - trk["first_frame"] + 1
    trk["span_s"] = trk["span_frames"] / fps
    trk["start_phase"] = phase_f[trk["first_frame"].to_numpy()]
    trk.insert(0, "recording", rec.name)
    n_steps = int(np.isfinite(df["step_px"]).sum())
    row.update(
        track_span_s_median=float(trk["span_s"].median()),
        track_span_s_q25=float(trk["span_s"].quantile(0.25)),
        track_span_s_q75=float(trk["span_s"].quantile(0.75)),
        # length of the track a random detection belongs to (detection-weighted median)
        track_span_s_detection_weighted_median=float(weighted_median(trk["span_s"], trk["n_instances"])),
        share_tracks_under_1s=float((trk["span_s"] < 1).mean()),
        share_detections_in_tracks_over_10s=float(trk.loc[trk["span_s"] > 10, "n_instances"].sum() / len(df)),
        share_detections_in_tracks_over_60s=float(trk.loc[trk["span_s"] > 60, "n_instances"].sum() / len(df)),
        tracks_per_video_hour=len(trk) / (nfr / fps / 3600),
        share_tracks_with_gaps=float((trk["n_gaps"] > 0).mean()),
        n_steps=n_steps, n_jumps=int(df["jump"].sum()), n_big_jumps=int(df["big_jump"].sum()),
        n_swap_like=int(df["swap_like"].sum()),
        jumps_per_1000_steps=1000 * df["jump"].sum() / max(n_steps, 1),
        big_jumps_per_1000_steps=1000 * df["big_jump"].sum() / max(n_steps, 1),
        jumps_per_fish_hour=df["jump"].sum() / (n_steps / fps / 3600) if n_steps else np.nan,
        swap_like_share_of_jumps=float(df["swap_like"].sum() / max(df["jump"].sum(), 1)),
        step_bl_p99=float(np.nanpercentile(df["step_px"], 99) / bl),
        step_bl_p999=float(np.nanpercentile(df["step_px"], 99.9) / bl),
    )
    for p in ("dark", "light"):
        k = PHASES.index(p)
        s = df[df["phase"] == k]
        ns = int(np.isfinite(s["step_px"]).sum())
        row[f"jumps_per_1000_steps_{p}"] = 1000 * s["jump"].sum() / max(ns, 1)
        sel = phase_f == k
        row[f"tracks_per_hour_{p}"] = float((trk["start_phase"] == k).sum() / (sel.sum() / fps / 3600)) if sel.any() else np.nan

    # ---- dropouts (runs with < 2 instances) and where the fish was lost / reappeared
    drop = dropout_table(rec.name, counts, df, slot_idx, slot_pos, t_frame, phase_f, fps)
    unc = drop[~drop["censored"]]
    missing = unc["missing_fish_frames"].sum()  # missing fish-frames in complete dropouts
    longd = unc[unc["dur_s"] >= MIN_DROPOUT_S]
    row.update(
        n_dropouts=len(unc), n_dropouts_over_1s=len(longd),
        dropout_s_median=float(unc["dur_s"].median()) if len(unc) else np.nan,
        dropout_over_1s_s_median=float(longd["dur_s"].median()) if len(longd) else np.nan,
        share_missing_fish_frames_in_dropouts_over_1s=float(
            (unc["missing_fish_frames"] * (unc["dur_s"] >= 1)).sum() / missing) if missing else np.nan,
        share_missing_fish_frames_in_dropouts_over_60s=float(
            (unc["missing_fish_frames"] * (unc["dur_s"] >= 60)).sum() / missing) if missing else np.nan,
        longest_dropout_s=float(unc["dur_s"].max()) if len(unc) else np.nan,
    )
    # B2: frames with two well-separated, confident fish. "all3": instance score and all
    # three node scores >= 0.5; "headtail": the same for head and tail, centre optional
    # (in the light phase SLEAP rarely places the centre node).
    sc = df["instance_score"].to_numpy() >= CONF_INSTANCE
    node_ok = {n: df[f"{n}_score"].to_numpy() >= CONF_NODE for n in NODES}
    in_rec = (phase_f == 1) | (phase_f == 2)
    row["n_frames_two_instances_close"] = int(np.sum((counts == 2) & (pair_d < bl)))
    for tag, ok in (("all3", sc & node_ok["head"] & node_ok["center"] & node_ok["tail"]),
                    ("headtail", sc & node_ok["head"] & node_ok["tail"])):
        per_frame = np.bincount(df["frame"].to_numpy()[ok], minlength=nfr)
        two = (per_frame == 2) & (counts == 2)
        sep = two & (pair_d > 2 * bl)
        row[f"n_frames_two_confident_{tag}"] = int(np.sum(two & in_rec))
        row[f"n_frames_two_confident_separated_{tag}"] = int(np.sum(sep & in_rec))
        row[f"n_frames_two_confident_separated_{tag}_dark"] = int(np.sum(sep & (phase_f == 1)))
        row[f"n_frames_two_confident_separated_{tag}_light"] = int(np.sum(sep & (phase_f == 2)))
        # in how many separate minutes do they occur (spread over the recording)
        row[f"n_minutes_with_two_confident_separated_{tag}"] = int(len(np.unique(minute[sep & in_rec])))
    return dict(row=row, coverage=cov, tracks=trk, dropouts=drop)


def weighted_median(x, w) -> float:
    o = np.argsort(np.asarray(x))
    x, w = np.asarray(x)[o], np.asarray(w, float)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def dropout_table(name, counts, df, slot_idx, slot_pos, t_frame, phase_f, fps) -> pd.DataFrame:
    """One row per run of frames with < 2 instances.

    Lost fish: in the last two-fish frame before the run, the instance farther
    from the instance that stays (both instances if the run starts with an empty
    frame). Reappeared fish: likewise in the first two-fish frame after the run.
    Censored runs touch the first or last video frame.
    """
    nfr = len(counts)
    s, e = runs(counts < 2)
    rows = []
    sc = df["instance_score"].to_numpy()
    nn = df["n_nodes"].to_numpy()

    def odd_one(f_two, f_one):
        """Instances of two-fish frame f_two that are not the one seen in frame f_one."""
        pos2 = slot_pos[f_two]
        if counts[f_one] == 0:
            return [0, 1]
        p1 = slot_pos[f_one][slot_idx[f_one] >= 0][0]
        d = np.linalg.norm(pos2 - p1, axis=1)
        return [int(np.argmax(d))]

    for a, b in zip(s, e):
        cen = a == 0 or b == nfr - 1
        r = dict(recording=name, start_frame=int(a), end_frame=int(b), n_frames=int(b - a + 1),
                 dur_s=(b - a + 1) / fps, t_start_s=float(t_frame[a]), phase=int(phase_f[a]),
                 min_count=int(counts[a:b + 1].min()), share_zero=float(np.mean(counts[a:b + 1] == 0)),
                 missing_fish_frames=int((2 - counts[a:b + 1]).sum()), censored=bool(cen))
        for tag, f_two, f_one in (("lost", a - 1, a), ("new", b + 1, b)):
            ks = odd_one(f_two, f_one) if 0 <= f_two < nfr and counts[f_two] == 2 else []
            for i in (0, 1):
                if i < len(ks):
                    k = ks[i]
                    ii = slot_idx[f_two][k]
                    r[f"{tag}{i + 1}_x"], r[f"{tag}{i + 1}_y"] = slot_pos[f_two][k]
                    r[f"{tag}{i + 1}_score"], r[f"{tag}{i + 1}_nodes"] = sc[ii], nn[ii]
                else:
                    r[f"{tag}{i + 1}_x"] = r[f"{tag}{i + 1}_y"] = r[f"{tag}{i + 1}_score"] = np.nan
                    r[f"{tag}{i + 1}_nodes"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def stage_tables() -> None:
    (TABLES / "instances").mkdir(parents=True, exist_ok=True)
    rows, cov, trk, drop = [], [], [], []
    for rec in list_recordings():
        res = process(rec)
        rows.append(res["row"])
        cov.append(res["coverage"])
        trk.append(res["tracks"])
        drop.append(res["dropouts"])
        r = res["row"]
        print(f"{rec.name}: 0/1/2 = {r['share0_all']:.3f}/{r['share1_all']:.3f}/{r['share2_all']:.3f}, "
              f"{r['n_tracks']} tracks, LED {r['led_status'][:7]} med {r['led_resid_med_ms']:.1f} ms, "
              f"video {r['t_first_frame_s']:.0f}..{r['t_last_frame_s']:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "recordings.csv", index=False)
    pd.concat(cov).to_parquet(TABLES / "coverage_per_minute.parquet", index=False)
    pd.concat(trk).to_parquet(TABLES / "tracklets.parquet", index=False)
    pd.concat(drop).to_parquet(TABLES / "dropouts.parquet", index=False)
    print("wrote", TABLES)


# ----------------------------------------------------------------------------- video frames
BACKGROUND = OUT / "background"
EXAMPLES = OUT / "examples"
LOC_MARGIN_BL = 0.5  # a position within 0.5 BL of an object's outline counts as "at" it
LOCATIONS = ("tube", "block", "edge", "open")


def read_frames(video: Path, frames: list[int]) -> dict[int, np.ndarray]:
    """Grey images of a few frames, read in increasing order by seeking (never the whole video)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = {}
    for f in sorted(set(int(x) for x in frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"{video}: could not read frame {f}")
        out[f] = img[..., 0].copy()
    cap.release()
    return out


def find_objects(gray: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int]]:
    """Dark objects (tube, blocks) and the bright back-lit arena area of one frame.

    The arena floor is back-lit, fish and objects are dark silhouettes. Arena =
    bounding box of the largest bright component. Objects = dark components
    inside the arena after an opening with a 41-px square, which removes
    cables, electrodes' leads and fish (about 20-30 px wide). The tube is the
    elongated one (aspect >= 2, about 295 x 85 px), the four blocks the compact
    ones of about 95 x 95 px; the rest (the two round electrodes in the middle,
    about 56 px wide) are "small".
    """
    import cv2

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ax, ay, aw, ah = (int(v) for v in stats[k, :4])
    floor = np.median(gray[lab == k])
    dark = (blur < 0.45 * floor).astype(np.uint8)
    dark[:ay] = dark[ay + ah:] = 0
    dark[:, :ax] = dark[:, ax + aw:] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(dark)
    objs = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if area < 1500:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        kind = "tube" if aspect >= 2 and area > 10000 else "block" if area > 5000 else "small"
        objs.append(dict(kind=kind, x0=x, y0=y, x1=x + w, y1=y + h, area=area,
                         cx=float(cen[i][0]), cy=float(cen[i][1])))
    return objs, (ax, ay, ax + aw, ay + ah)


def stage_background() -> None:
    """One lit frame per recording and its arena objects.

    The frame: two confident fish more than 2 BL apart (so neither fish merges with
    an object), at least 60 s after the measured light-on, closest to recording
    time 12600 s. The bright-floor box varies by a few px with the brightness of the
    frame, while the camera did not move (objects within 2 px), so every recording
    gets the median box over all recordings.
    """
    import cv2

    BACKGROUND.mkdir(parents=True, exist_ok=True)
    recs = pd.read_csv(TABLES / "recordings.csv").set_index("recording")
    old = pd.read_csv(TABLES / "arena_objects.csv") if (TABLES / "arena_objects.csv").exists() else None
    rows = []
    for rec in list_recordings():
        d = pd.read_parquet(TABLES / "instances" / f"{rec.name}.parquet",
                            columns=["frame", "t_s", "phase", "instance_score", "head_score", "tail_score",
                                     "n_in_frame", "pair_dist_px"])
        bl, lit = recs.loc[rec.name, "body_length_px"], recs.loc[rec.name, "light_on_measured_s"] + 60
        ok = (d.instance_score >= CONF_INSTANCE) & (d.head_score >= CONF_NODE) & (d.tail_score >= CONF_NODE)
        good = d[ok & (d.n_in_frame == 2) & (d.pair_dist_px > 2 * bl)].groupby("frame").agg(
            n=("t_s", "size"), t=("t_s", "first"))
        good = good[(good.n == 2) & (good.t > lit) & (good.t < recs.loc[rec.name, "rec_dur_s"])]
        if len(good):
            f = int((good.t - 12600).abs().idxmin())
        else:
            t = np.load(TABLES / "frame_time" / f"{rec.name}.npy")
            f = int(np.argmin(np.abs(t - 12600)))
        png = BACKGROUND / f"{rec.name}.png"
        cached = old is not None and (old[old.recording == rec.name].frame == f).any() and png.exists()
        if cached:  # this frame was read before: reuse it
            gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        else:
            gray = read_frames(rec.video, [f])[f]
            cv2.imwrite(str(png), gray)
        objs, arena = find_objects(gray)
        for o in objs:
            rows.append(dict(recording=rec.name, frame=f, **o, arena_x0=arena[0], arena_y0=arena[1],
                             arena_x1=arena[2], arena_y1=arena[3]))
        kinds = [o["kind"] for o in objs]
        print(rec.name, "frame", f, "(cached)" if cached else "(read)", "arena", arena,
              {k: kinds.count(k) for k in set(kinds)}, flush=True)
    df = pd.DataFrame(rows)
    box = df.groupby("recording")[["arena_x0", "arena_y0", "arena_x1", "arena_y1"]].first()
    print("bright-floor box per recording, range:", {c: (int(box[c].min()), int(box[c].max())) for c in box})
    for c in box:
        df[f"{c}_own"] = df[c]
        df[c] = int(box[c].median())
    df.to_csv(TABLES / "arena_objects.csv", index=False)
    fig_arena_objects(df)


def fig_arena_objects(objs: pd.DataFrame) -> None:
    """QC: the background frame of every recording with the detected arena and objects."""
    import cv2

    plt = _style()
    labels = rec_labels()
    names = sorted(objs.recording.unique())
    fig, axes = plt.subplots(4, 5, figsize=(17 * CM, 11 * CM))
    for ax, name in zip(axes.flat, names):
        bg = cv2.imread(str(BACKGROUND / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        o = objs[objs.recording == name]
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255)
        a = o.iloc[0]
        ax.add_patch(plt.Rectangle((a.arena_x0, a.arena_y0), a.arena_x1 - a.arena_x0, a.arena_y1 - a.arena_y0,
                                   fill=False, ec="yellow", lw=0.7, ls="--"))
        for r in o.itertuples():
            ax.add_patch(plt.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False,
                                       ec=LOC_COLORS.get(r.kind, "cyan"), lw=0.8))
        kinds = o.kind.value_counts().to_dict()
        ax.set_title(f"{labels[name]}: {kinds.get('tube', 0)} tube, {kinds.get('block', 0)} blocks", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    # how far does each object move between recordings (centre vs its median over recordings)?
    big = objs[objs.kind.isin(["tube", "block"])].copy()
    big["slot"] = big.groupby("recording").cx.rank().astype(int) * 10 + big.groupby("recording").cy.rank().astype(int)
    dev = big.groupby("slot")[["cx", "cy"]].transform(lambda v: v - v.median())
    shift = float(np.hypot(dev.cx, dev.cy).max())
    fig.suptitle("The tube (red) and the four blocks (purple) are in the same place in all 19 videos "
                 f"(centres within {shift:.0f} px);\ncyan: the two round electrodes; yellow dashed: bright "
                 "floor (median box over the recordings). One lit frame per recording.", x=0.01, ha="left", fontsize=8.5,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "00_arena_objects.png")
    plt.close(fig)


def classify_location(x: np.ndarray, y: np.ndarray, objs: pd.DataFrame, bl: float) -> np.ndarray:
    """'tube', 'block', 'edge' or 'open' for positions (px); NaN positions -> ''."""
    m = LOC_MARGIN_BL * bl
    out = np.full(len(x), "open", dtype=object)
    a = objs.iloc[0]
    edge = (x < a.arena_x0 + m) | (x > a.arena_x1 - m) | (y < a.arena_y0 + m) | (y > a.arena_y1 - m)
    out[edge] = "edge"
    for kind in ("block", "tube"):  # tube last: it wins where both apply
        for o in objs[objs.kind == kind].itertuples():
            at = (x > o.x0 - m) & (x < o.x1 + m) & (y > o.y0 - m) & (y < o.y1 + m)
            out[at] = kind
    out[~(np.isfinite(x) & np.isfinite(y))] = ""
    return out


def stage_locations() -> None:
    """Classify where fish are lost / reappear, and where detections are, by location."""
    objs = pd.read_csv(TABLES / "arena_objects.csv")
    recs = pd.read_csv(TABLES / "recordings.csv").set_index("recording")
    drop = pd.read_parquet(TABLES / "dropouts.parquet")
    parts, base, mining = [], [], []
    for name, dr in drop.groupby("recording"):
        o, bl = objs[objs.recording == name], recs.loc[name, "body_length_px"]
        dr = dr.copy()
        for tag in ("lost1", "lost2", "new1", "new2"):
            dr[f"{tag}_loc"] = classify_location(dr[f"{tag}_x"].to_numpy(), dr[f"{tag}_y"].to_numpy(), o, bl)
        parts.append(dr)
        d = pd.read_parquet(TABLES / "instances" / f"{name}.parquet")
        d["loc"] = classify_location(d.centroid_x.to_numpy(), d.centroid_y.to_numpy(), o, bl)
        d = d[d.phase.isin([1, 2])]
        b = d.groupby(["phase", "loc"]).agg(n=("loc", "size"), single_node=("n_nodes", lambda s: (s == 1).mean()),
                                            median_score=("instance_score", "median")).reset_index()
        b.insert(0, "recording", name)
        base.append(b)
        mining.append(pairs_and_mining(name, d, bl))
    pd.concat(parts).to_parquet(TABLES / "dropouts.parquet", index=False)
    pd.concat(base).to_csv(TABLES / "detections_by_location.csv", index=False)
    pd.DataFrame(mining).to_csv(TABLES / "pairs_and_mining.csv", index=False)
    print("wrote locations, pairs_and_mining")


SHARED_NODE_PX = 2.0  # two instances with a node closer than this share it: one fish detected twice
COLLAPSED_BL = 0.25  # head and tail closer than this: the pose has collapsed (e.g. only a head out of the tube)
MIN_HEAD_TAIL_BL = 0.5  # minimum head-tail distance of a minable instance


def pairs_and_mining(name: str, d: pd.DataFrame, bl: float) -> dict:
    """Duplicates among two-instance frames and the frames B2 could mine, of one recording.

    ``d``: the recording's instances inside the electric recording (dark or light),
    with ``loc``. Mining criteria (each instance): instance, head and tail score
    >= 0.5, head-tail distance >= 0.5 BL; the pair: centroids > 2 BL apart and no
    shared node. "clean" also requires both instances away from the tube and the
    blocks (location open or edge).
    """
    d = d.sort_values(["frame", "track"], kind="stable")
    ht = np.hypot(d.head_x - d.tail_x, d.head_y - d.tail_y).to_numpy() / bl
    has_ht = np.isfinite(ht)
    ok = ((d.instance_score >= CONF_INSTANCE) & (d.head_score >= CONF_NODE) & (d.tail_score >= CONF_NODE)).to_numpy()
    two = d[d.n_in_frame == 2]
    a, b = two.iloc[0::2], two.iloc[1::2]
    if not np.array_equal(a.frame.to_numpy(), b.frame.to_numpy()):
        raise ValueError(f"{name}: two-instance frames are not in consecutive rows")
    na = np.stack([a[[f"{n}_x", f"{n}_y"]].to_numpy() for n in NODES], 1)  # (m, 3, 2)
    nb = np.stack([b[[f"{n}_x", f"{n}_y"]].to_numpy() for n in NODES], 1)
    dist = np.linalg.norm(na[:, :, None, :] - nb[:, None, :, :], axis=3)  # (m, 3, 3)
    shared = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), axis=(1, 2)) < SHARED_NODE_PX
    cd = a.pair_dist_px.to_numpy() / bl
    idx = {k: v for v, k in enumerate(d.index)}
    ia = np.array([idx[k] for k in a.index])
    ib = np.array([idx[k] for k in b.index])
    good = ok & has_ht & (ht >= MIN_HEAD_TAIL_BL)
    away = d["loc"].isin(["open", "edge"]).to_numpy()
    sep = (cd > 2) & ~shared
    mine = sep & good[ia] & good[ib]
    clean = mine & away[ia] & away[ib]
    ph = a.phase.to_numpy()
    minute = np.floor(a.t_s.to_numpy() / 60)
    row = dict(recording=name, n_two_frames=len(a),
               share_two_frames_shared_node=float(shared.mean()) if len(a) else np.nan,
               share_two_frames_closer_0p5bl=float(np.mean(cd < 0.5)) if len(a) else np.nan,
               share_instances_collapsed=float(np.mean(ht[has_ht] < COLLAPSED_BL)),
               share_instances_collapsed_at_tube=float(np.mean(ht[has_ht & (d["loc"] == "tube").to_numpy()] < COLLAPSED_BL)),
               n_instances_head_tail=int(has_ht.sum()), n_collapsed=int(np.sum(ht[has_ht] < COLLAPSED_BL)),
               n_instances_head_tail_at_tube=int(np.sum(has_ht & (d["loc"] == "tube").to_numpy())),
               n_collapsed_at_tube=int(np.sum(ht[has_ht & (d["loc"] == "tube").to_numpy()] < COLLAPSED_BL)))
    for p, k in (("dark", 1), ("light", 2)):
        row[f"n_two_frames_{p}"] = int((ph == k).sum())
        row[f"n_two_frames_shared_node_{p}"] = int((shared & (ph == k)).sum())
        row[f"n_two_frames_closer_0p5bl_{p}"] = int(((cd < 0.5) & (ph == k)).sum())
    for tag, sel in (("mine", mine), ("clean", clean)):
        row[f"n_frames_{tag}"] = int(sel.sum())
        row[f"n_frames_{tag}_dark"] = int((sel & (ph == 1)).sum())
        row[f"n_frames_{tag}_light"] = int((sel & (ph == 2)).sum())
        row[f"n_minutes_{tag}"] = int(len(np.unique(minute[sel])))
    return row


# ----------------------------------------------------------------------------- example frames
LOOK_S = 0.5  # one-fish / no-fish examples are taken 0.5 s after the fish was lost
CROP = 480  # zoom crop (px of the video)
INSTANCE_COLORS = ("#1f77b4", "#ff7f0e")


def select_examples(seed: int = 2022) -> pd.DataFrame:
    """Example frames, stratified over recordings, dark/light and dropout location.

    - one_fish: 2 per (phase x location of the lost fish), 16 frames, from
      dropouts >= 1 s that still show exactly one instance 0.5 s after the loss;
    - no_fish: 3 dark + 3 light, from dropouts >= 1 s with no instance 0.5 s
      after the loss (location = where the first of the two fish was lost);
    - two_confident: 2 dark + 2 light frames with two confident instances
      (instance, head and tail scores >= 0.5) more than 2 BL apart;
    - two_close: 3 dark + 3 light frames with two instances less than 1 BL apart.
    Recordings are drawn round-robin (least used first) to spread the sample.
    """
    rng = np.random.default_rng(seed)
    recs = pd.read_csv(TABLES / "recordings.csv").set_index("recording")
    drop = pd.read_parquet(TABLES / "dropouts.parquet")
    drop = drop[~drop.censored & (drop.dur_s >= MIN_DROPOUT_S) & drop.phase.isin([1, 2])]
    used: dict[str, int] = {r: 0 for r in recs.index}
    counts_cache: dict[str, np.ndarray] = {}
    inst_cache: dict[str, pd.DataFrame] = {}

    def inst(name):
        if name not in inst_cache:
            inst_cache[name] = pd.read_parquet(TABLES / "instances" / f"{name}.parquet")
            counts_cache[name] = instances_per_frame(inst_cache[name]["frame"].to_numpy(), int(recs.loc[name, "n_frames"]))
        return inst_cache[name]

    def pick(cands: pd.DataFrame, n: int) -> list[pd.Series]:
        out = []
        cands = cands.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        for _ in range(n):
            if cands.empty:
                break
            order = cands["recording"].map(used).to_numpy()
            row = cands.iloc[int(np.flatnonzero(order == order.min())[0])]
            used[row["recording"]] += 1
            cands = cands.drop(index=row.name)
            out.append(row)
        return out

    rows = []
    look = int(round(LOOK_S * 25))
    drop = drop.reset_index(drop=True)
    for want, label in ((1, "one_fish"), (0, "no_fish")):
        ok = []
        for i, r in drop.iterrows():
            f = r.start_frame + look
            if f <= r.end_frame:
                ok.append(i)
        cand = drop.loc[ok].copy()
        cand["frame"] = cand.start_frame + look
        n_at = []
        for name, g in cand.groupby("recording"):
            inst(name)
            n_at.append(pd.Series(counts_cache[name][g.frame.to_numpy()], index=g.index))
        cand["n_at"] = pd.concat(n_at)
        cand = cand[cand.n_at == want]
        if label == "one_fish":  # only dropouts that began with one fish leaving (the missing fish is known)
            cand = cand[cand.lost2_x.isna()]
        if label == "one_fish":
            strata = [(p, loc, 2) for p in (1, 2) for loc in LOCATIONS]
        else:
            strata = [(p, None, 3) for p in (1, 2)]
        for p, loc, n in strata:
            c = cand[(cand.phase == p) & ((cand.lost1_loc == loc) if loc else True)]
            for r in pick(c, n):
                rows.append(dict(category=label, recording=r.recording, frame=int(r.frame), phase=int(p),
                                 lost_loc=r.lost1_loc, focus_x=r.lost1_x, focus_y=r.lost1_y,
                                 lost2_x=r.lost2_x, lost2_y=r.lost2_y, dropout_s=r.dur_s,
                                 lost_score=r.lost1_score, lost_nodes=r.lost1_nodes))
    # two-instance examples
    two = []
    for name in recs.index:
        d = inst(name)
        bl = recs.loc[name, "body_length_px"]
        d = d[d.phase.isin([1, 2]) & (d.n_in_frame == 2)]
        ok = (d.instance_score >= CONF_INSTANCE) & (d.head_score >= CONF_NODE) & (d.tail_score >= CONF_NODE)
        g = d.assign(ok=ok).groupby("frame").agg(n_ok=("ok", "sum"), phase=("phase", "first"),
                                                 dist=("pair_dist_px", "first"),
                                                 x=("centroid_x", "mean"), y=("centroid_y", "mean"))
        g["recording"], g["bl"] = name, bl
        two.append(g.reset_index())
    two = pd.concat(two, ignore_index=True)
    for label, sel, n in (("two_confident", (two.n_ok == 2) & (two.dist > 2 * two.bl), 2),
                          ("two_close", two.dist < two.bl, 3)):
        for p in (1, 2):
            for r in pick(two[sel & (two.phase == p)], n):
                rows.append(dict(category=label, recording=r.recording, frame=int(r.frame), phase=int(p),
                                 lost_loc="", focus_x=r.x, focus_y=r.y))
    ex = pd.DataFrame(rows)
    ex.insert(0, "example", [f"ex{i:02d}" for i in range(len(ex))])
    return ex


def _crop_box(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    """Top-left corner of a CROP x CROP box centred on (x, y), kept inside the frame."""
    x = w / 2 if not np.isfinite(x) else x
    y = h / 2 if not np.isfinite(y) else y
    return (int(np.clip(x - CROP / 2, 0, w - CROP)), int(np.clip(y - CROP / 2, 0, h - CROP)))


def _second_focus(ex: pd.Series, inst: pd.DataFrame) -> tuple[float, float, str]:
    """Where the third panel looks: the fish SLEAP does see, or the other lost fish."""
    if ex.category == "one_fish" and len(inst):
        r = inst.iloc[0]
        return r.centroid_x, r.centroid_y, "the fish SLEAP finds"
    if ex.category == "no_fish" and np.isfinite(ex.get("lost2_x", np.nan)):
        return ex.lost2_x, ex.lost2_y, "where the other fish was last seen"
    if ex.category.startswith("two") and len(inst) == 2:
        r = inst.iloc[1]
        return r.centroid_x, r.centroid_y, "second instance"
    return np.nan, np.nan, "frame centre"


def render_example(ex: pd.Series, gray: np.ndarray, inst: pd.DataFrame, objs: pd.DataFrame, t_s: float) -> None:
    """Full frame and two zoom crops with the SLEAP keypoints; saves the PNG and the first crop."""
    import cv2
    import matplotlib.pyplot as plt

    h, w = gray.shape
    if ex.category.startswith("two") and len(inst) == 2:
        f1 = (inst.iloc[0].centroid_x, inst.iloc[0].centroid_y, "first instance")
    else:
        f1 = (ex.focus_x, ex.focus_y, "where the missing fish was last seen" if ex.category in (
            "one_fish", "no_fish") else "pair")
    f2 = _second_focus(ex, inst)
    boxes = [_crop_box(f1[0], f1[1], w, h), _crop_box(f2[0], f2[1], w, h)]
    x0, y0 = boxes[0]
    cv2.imwrite(str(EXAMPLES / "crops" / f"{ex.example}.png"), gray[y0:y0 + CROP, x0:x0 + CROP])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[4 / 3, 1, 1]))
    for k_ax, ax in enumerate(axes):
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
        for k, (_, r) in enumerate(inst.iterrows()):
            c = INSTANCE_COLORS[k % 2]
            ax.plot([r[f"{n}_x"] for n in NODES], [r[f"{n}_y"] for n in NODES], "-", color=c, lw=1.0)
            for n, m in zip(NODES, ("o", "s", "^")):
                ax.plot(r[f"{n}_x"], r[f"{n}_y"], m, color=c, ms=5 if k_ax == 0 else 8, mfc="none", mew=1.3)
            if k_ax == 0:
                ax.annotate(f"{r.instance_score:.2f}", (r.centroid_x, r.centroid_y), xytext=(8, -8),
                            textcoords="offset points", color=c, fontsize=8)
        if ex.category in ("one_fish", "no_fish"):
            for tag in ("focus", "lost2"):
                x, y = ex.get(f"{tag}_x", np.nan), ex.get(f"{tag}_y", np.nan)
                if np.isfinite(x):
                    ax.plot(x, y, "x", color="red", ms=9 if k_ax == 0 else 14, mew=2)
        if k_ax == 0:
            for o in objs[objs.kind.isin(["tube", "block"])].itertuples():
                ax.add_patch(plt.Rectangle((o.x0, o.y0), o.x1 - o.x0, o.y1 - o.y0, fill=False,
                                           ec="yellow", ls=":", lw=0.8))
            for (bx, by), c in zip(boxes, ("red", "lime")):
                ax.add_patch(plt.Rectangle((bx, by), CROP, CROP, fill=False, ec=c, lw=0.9))
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        else:
            bx, by = boxes[k_ax - 1]
            ax.set_xlim(bx, bx + CROP)
            ax.set_ylim(by + CROP, by)
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color("red" if k_ax == 1 else "lime")
                side.set_linewidth(1.5)
            ax.set_title((f1 if k_ax == 1 else f2)[2], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    what = {"one_fish": "SLEAP finds one fish", "no_fish": "SLEAP finds no fish",
            "two_confident": "two confident fish", "two_close": "two fish < 1 body length apart"}[ex.category]
    lost = (f"; red x = missing fish last seen {LOOK_S:.1f} s earlier ({ex.lost_loc})"
            if ex.category in ("one_fish", "no_fish") else "")
    fig.suptitle(f"{ex.example} {ex.recording} frame {int(ex.frame)}, recording time {t_s / 60:.1f} min "
                 f"({PHASES[int(ex.phase)]}): {what}{lost}\n"
                 "SLEAP keypoints: o head, square centre, triangle tail (numbers: instance score); "
                 "zooms 480 x 480 px", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(EXAMPLES / f"{ex.example}_{ex.category}.png", dpi=110)
    plt.close(fig)


def render_scan(ex: pd.Series, gray: np.ndarray, inst: pd.DataFrame) -> None:
    """Large contrast-enhanced view of the whole frame, to search it for fish by eye.

    CLAHE-equalised (the walls are dark), 1400 px wide; SLEAP instances as small
    coloured circles (head, centre, tail), the missing fish's last position as a red x.
    """
    import cv2

    img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for k, (_, r) in enumerate(inst.iterrows()):
        c = ((180, 119, 31), (14, 127, 255))[k % 2]  # BGR of INSTANCE_COLORS
        for n in NODES:
            if np.isfinite(r[f"{n}_x"]):
                cv2.circle(img, (int(r[f"{n}_x"]), int(r[f"{n}_y"])), 14, c, 3)
    if ex.category in ("one_fish", "no_fish"):
        for tag in ("focus", "lost2"):
            x, y = ex.get(f"{tag}_x", np.nan), ex.get(f"{tag}_y", np.nan)
            if np.isfinite(x):
                cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 4)
    img = cv2.resize(img, (1400, int(1400 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(EXAMPLES / "scan" / f"{ex.example}_scan.png"), img)


def stage_examples() -> None:
    """Pick the example frames, read them (once; cached as PNG) and render them."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    for sub in ("crops", "frames", "scan"):
        (EXAMPLES / sub).mkdir(parents=True, exist_ok=True)
    frozen = EXAMPLES / "selection.csv"
    if frozen.exists():  # the judged sample: never re-drawn (judgements.csv refers to it)
        ex = pd.read_csv(frozen)
    else:
        ex = select_examples()
        ex.to_csv(frozen, index=False)
    objs = pd.read_csv(TABLES / "arena_objects.csv")
    recs = {r.name: r for r in list_recordings()}
    out = []
    for name, g in ex.groupby("recording"):
        d = pd.read_parquet(TABLES / "instances" / f"{name}.parquet")
        cached = {int(f): EXAMPLES / "frames" / f"{name}_{int(f)}.png" for f in g.frame}
        todo = [f for f, p in cached.items() if not p.exists()]
        frames = read_frames(recs[name].video, todo) if todo else {}  # sequential, a few frames per video
        for f, img in frames.items():
            cv2.imwrite(str(cached[f]), img)
        t_frame = np.load(TABLES / "frame_time" / f"{name}.npy")
        for _, e in g.iterrows():
            f = int(e.frame)
            gray = frames[f] if f in frames else cv2.imread(str(cached[f]), cv2.IMREAD_GRAYSCALE)
            inst = d[d.frame == f]
            t_s = float(t_frame[f])
            render_example(e, gray, inst, objs[objs.recording == name], t_s)
            render_scan(e, gray, inst)
            e = e.copy()
            e["t_s"] = t_s
            e["n_instances"] = len(inst)
            e["instance_scores"] = ";".join(f"{s:.2f}" for s in inst.instance_score)
            e["instance_nodes"] = ";".join(str(int(n)) for n in inst.n_nodes)
            e["frame_mean_grey"] = float(gray.mean())
            out.append(e)
        print(name, len(g), "examples", f"({len(todo)} frames read from the video)", flush=True)
    pd.DataFrame(out).sort_values("example").to_csv(EXAMPLES / "examples.csv", index=False)
    print("wrote", EXAMPLES)


# ----------------------------------------------------------------------------- figures
FIGURES = OUT / "figures"
CM = 1 / 2.54
C_TWO, C_ONE, C_ZERO = "#08519c", "#6baed6", "#bdbdbd"
C_DARK, C_LIGHT = "#253494", "#e6550d"
LOC_COLORS = {"tube": "#d62728", "block": "#9467bd", "edge": "#8c564b", "open": "#7f7f7f"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 200,
                         "savefig.dpi": 200, "axes.titlelocation": "left"})
    return plt


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    print("wrote", FIGURES / name)


def rec_labels() -> dict[str, str]:
    """'05-10 (1a)': date and pairing from raw/meta.csv; 07-07 is the control (Versuchszettel: 2l4w)."""
    from paths import REC_2022

    m = pd.read_csv(REC_2022 / "raw" / "meta.csv", index_col=0)
    lab = {r: f"{r[5:10]} ({m.loc[r, 'Pairing']})" for r in m.index}
    return {r.name: lab.get(r.name, f"{r.name[5:10]} (control)") for r in list_recordings()}


def load_instances(cols: list[str]) -> pd.DataFrame:
    parts = []
    for rec in list_recordings():
        d = pd.read_parquet(TABLES / "instances" / f"{rec.name}.parquet", columns=cols)
        d["recording"] = rec.name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _plain_log_ticks(axis, ticks=(0.1, 1, 10, 100, 1000)) -> None:
    import matplotlib.ticker as mt

    axis.set_major_locator(mt.FixedLocator(ticks))
    axis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mt.NullFormatter())


def _frame_bins(max_s: float, fps: float = 25.0, n: int = 60) -> np.ndarray:
    """Log-spaced bin edges that straddle whole frames (durations are multiples of 1/fps)."""
    k = np.unique(np.round(np.logspace(0, np.log10(max_s * fps), n)).astype(int))
    return np.r_[k - 0.5, k[-1] + 0.5] / fps


def fig_coverage(recs: pd.DataFrame, labels: dict) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 9.5 * CM), sharey=True)
    y = np.arange(len(recs))[::-1]
    pooled = {}
    for ax, ph, title in zip(axes, ("dark", "light"), ("dark phase (recording time 0-3 h)",
                                                       "light phase (3 h to the end, about 1 h)")):
        s2, s1, s0 = (recs[f"share{k}_{ph}"].to_numpy() for k in (2, 1, 0))
        ax.barh(y, s2, color=C_TWO, label="two fish")
        ax.barh(y, s1, left=s2, color=C_ONE, label="one fish")
        ax.barh(y, s0, left=s2 + s1, color=C_ZERO, label="no fish")
        w = recs[f"frames_{ph}"].to_numpy()
        pooled[ph] = [np.sum(recs[f"share{k}_{ph}"] * w) / w.sum() for k in (2, 1, 0)]
        ax.set_title(f"{title}\npooled: two {pooled[ph][0]:.0%}, one {pooled[ph][1]:.0%}, "
                     f"none {pooled[ph][2]:.0%}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of video frames")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_yticks(y, [labels[r] for r in recs.recording])
    axes[0].set_ylabel("recording (date, pair and bout)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.55, -0.06), ncols=3, handlelength=1)
    fig.suptitle(f"SLEAP finds both fish in {pooled['dark'][0]:.0%} of the dark-phase frames and in "
                 f"{pooled['light'][0]:.0%} of the light-phase frames", x=0.01, ha="left", fontsize=9,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_coverage_per_recording.png")
    plt.close(fig)


def fig_coverage_time(recs: pd.DataFrame, labels: dict) -> None:
    plt = _style()
    cov = pd.read_parquet(TABLES / "coverage_per_minute.parquet")
    cov = cov[(cov.minute >= 0) & (cov.minute < 240)]
    m = cov.pivot(index="recording", columns="minute", values="c2").reindex(recs.recording)
    fig, ax = plt.subplots(figsize=(17 * CM, 8.5 * CM))
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
                   extent=(0, 240, len(m) - 0.5, -0.5), origin="upper")
    ax.axvline(LIGHT_ON_S / 60, color="white", lw=1, ls="--")
    ax.plot(recs.light_on_measured_s / 60, np.arange(len(recs)), "w|", ms=7, mew=1.5)
    ax.text(LIGHT_ON_S / 60 + 1, -0.9, "light on (dashed: meta.csv, 180 min; ticks: measured in the video)",
            fontsize=7, va="bottom")
    ax.set_yticks(np.arange(len(m)), [labels[r] for r in m.index])
    ax.set_xlabel("recording time (min)")
    ax.set_xticks(np.arange(0, 241, 30))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("share of frames with two fish (per minute)")
    ax.set_title("Two-fish coverage changes from minute to minute and drops after the light comes on "
                 "in most recordings", fontsize=9, fontweight="bold", pad=14)
    _save(fig, "02_coverage_over_time.png")
    plt.close(fig)


def fig_instance_scores(inst: pd.DataFrame, drop: pd.DataFrame) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 7.5 * CM), sharey=True)
    d = drop[~drop.censored & (drop.dur_s >= MIN_DROPOUT_S)]
    med = {}
    for ax, ph in zip(axes, (1, 2)):
        s = inst[inst.phase == ph]
        dd = d[d.phase == ph]
        groups = [
            (s[s.n_in_frame == 2].instance_score, "black", "in two-fish frames", "-"),
            (s[s.n_in_frame == 1].instance_score, "0.55", "in one-fish frames", "-"),
            (pd.concat([dd.lost1_score, dd.lost2_score]), "#d62728", "last before a dropout >= 1 s", "-"),
            (pd.concat([dd.new1_score, dd.new2_score]), "#2ca02c", "first after the dropout", "--"),
            (s[s.jump].instance_score, "#9467bd", "right after a jump (> 0.5 BL)", "-"),
        ]
        for k, (v, c, lab, ls) in enumerate(groups):
            v = np.sort(v.to_numpy()[np.isfinite(v.to_numpy())])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.2, ls=ls)
            med[(ph, k)] = float(np.median(v))
            ax.text(1.28, 0.36 - 0.075 * k, f"{lab}: {np.median(v):.2f} (n = {len(v):,})", ha="right",
                    fontsize=6.3, color=c)
        ax.text(1.28, 0.435, "median instance score of detections", ha="right", fontsize=6.3)
        ax.set_xlabel("instance score")
        ax.set_title("dark phase" if ph == 1 else "light phase")
        ax.set_xlim(0.2, 1.3)
    axes[0].set_ylabel("cumulative share of detections")
    dark = inst[inst.phase == 1].instance_score
    fig.suptitle(f"Detections just before a fish is lost and just after it reappears score low (dark: median "
                 f"{med[(1, 2)]:.2f} and {med[(1, 3)]:.2f}\nvs {med[(1, 0)]:.2f}), but so do many others: "
                 f"{np.mean(dark < 0.3):.0%} of all dark-phase detections score below 0.3\n(instance score = "
                 "SLEAP's centroid confidence; SLEAP keeps detections >= 0.2)", x=0.01, ha="left", fontsize=8.5,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_instance_scores_by_context.png")
    plt.close(fig)


def fig_node_scores(inst: pd.DataFrame) -> None:
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(17 * CM, 9 * CM), sharex="row", sharey="row",
                             gridspec_kw=dict(height_ratios=[1, 1.6]))
    bins = np.linspace(0.2, 1.2, 51)
    miss = {}
    for j, n in enumerate(NODES):
        for ph, c, lab in ((1, C_DARK, "dark"), (2, C_LIGHT, "light")):
            s = inst.loc[inst.phase == ph, f"{n}_score"].to_numpy()
            miss[(n, ph)] = m = float(np.mean(~np.isfinite(s)))
            axes[0, j].bar(lab, m, color=c, width=0.6)
            axes[0, j].text(lab, m + 0.03, f"{m:.0%}", ha="center", fontsize=7)
            axes[1, j].hist(s[np.isfinite(s)], bins=bins, histtype="step", color=c, density=True, lw=1.2,
                            label=f"{lab} (median {np.nanmedian(s):.2f})")
        axes[0, j].set_title({"head": "head", "center": "centre", "tail": "tail"}[n] + " node")
        axes[0, j].set_ylim(0, 1.05)
        axes[1, j].set_xlabel("node score (found nodes)")
        axes[1, j].legend(loc="upper left", fontsize=6.5)
    axes[0, 0].set_ylabel("share of detections\nwithout this node")
    axes[1, 0].set_ylabel("density")
    fig.suptitle(f"SLEAP leaves out the head in {miss[('head', 1)]:.0%} of the detections and the centre in "
                 f"{miss[('center', 2)]:.0%} of the light-phase detections;\nthe nodes it does place mostly "
                 "score above 0.8", x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_node_scores.png")
    plt.close(fig)


def fig_tracklets(trk: pd.DataFrame, recs: pd.DataFrame) -> None:
    plt = _style()
    fig, ax = plt.subplots(figsize=(17 * CM, 7 * CM))
    bins = _frame_bins(2000)
    span = trk.span_s.to_numpy()
    w_tr = np.full(len(span), 1 / len(span))
    w_det = (trk.n_instances / trk.n_instances.sum()).to_numpy()
    width = np.diff(np.log10(bins))
    for w, c in ((w_tr, "0.3"), (w_det, C_TWO)):
        h, _ = np.histogram(span, bins=bins, weights=w)
        ax.stairs(h / width / 10, bins, color=c, lw=1.3)  # share per 0.1 decade
    med, wmed = np.median(span), weighted_median(span, trk.n_instances)
    ax.axvline(med, color="0.3", ls=":", lw=1)
    ax.axvline(wmed, color=C_TWO, ls=":", lw=1)
    top = ax.get_ylim()[1]
    ax.text(med * 0.9, top * 0.95, f"half of the {len(span):,} tracks\nlast < {med:.1f} s", color="0.3",
            ha="right", va="top", fontsize=7)
    ax.text(wmed * 1.1, top * 0.95, f"half of the detections are in\ntracks shorter than {wmed:.0f} s",
            color=C_TWO, ha="left", va="top", fontsize=7)
    ax.set_xscale("log")
    _plain_log_ticks(ax.xaxis)
    ax.set_xlabel("track duration, first to last detection (s)")
    ax.set_ylabel("share per 0.1 decade")
    ax.text(0.99, 0.45, "grey: share of tracks\nblue: share of detections", transform=ax.transAxes,
            ha="right", fontsize=7)
    per_h = recs.tracks_per_video_hour
    ax.set_title(f"SLEAP's tracks are short: {per_h.min():.0f}-{per_h.max():.0f} new track ids per video hour "
                 "for two fish (19 recordings pooled)", fontsize=9, fontweight="bold")
    _save(fig, "05_tracklet_durations.png")
    plt.close(fig)


def fig_jumps(inst: pd.DataFrame, recs: pd.DataFrame, labels: dict) -> None:
    plt = _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(17 * CM, 7.5 * CM), gridspec_kw=dict(width_ratios=[1.2, 1]))
    bl = inst.recording.map(recs.set_index("recording").body_length_px)
    step_all = (inst.step_px / bl).to_numpy()
    ok = np.isfinite(step_all)
    step = step_all[ok]
    lo = 1e-3
    bins = np.logspace(np.log10(lo), np.log10(15), 70)
    a.hist(np.clip(step, lo * 1.001, None), bins=bins, color="0.4")
    a.set_xscale("log")
    a.set_yscale("log")
    _plain_log_ticks(a.xaxis, (0.001, 0.01, 0.1, 1, 10))
    a.axvline(JUMP_BL, color="#d62728", lw=1)
    share = np.mean(step > JUMP_BL)
    a.text(JUMP_BL * 1.15, 3e6, f"jump: > {JUMP_BL} BL\n{share:.1%} of steps", color="#d62728", fontsize=7,
           va="top")
    a.text(4, 1.2e4, "second peak:\njumps across\nthe arena", fontsize=6.5, ha="center", va="bottom")
    a.set_xlabel("step of a track between consecutive frames\n(body lengths; steps < 0.001 in the first bin)")
    a.set_ylabel("number of steps")
    a.set_title(f"{len(step):,} steps, all recordings")
    y = np.arange(len(recs))[::-1]
    b.plot(recs.jumps_per_1000_steps_dark, y, "o", color=C_DARK, ms=3.5, label="dark")
    b.plot(recs.jumps_per_1000_steps_light, y, "o", color=C_LIGHT, ms=3.5, label="light")
    b.set_yticks(y, [labels[r] for r in recs.recording])
    b.set_xlabel("jumps per 1000 steps")
    b.legend(loc="lower right")
    sw = recs.n_swap_like.sum() / recs.n_jumps.sum()
    b.set_title(f"{sw:.0%} of the jumps land where the other fish was")
    ph = inst.phase.to_numpy()[ok]
    rate = {p: 1000 * np.mean(step[ph == p] > JUMP_BL) for p in (1, 2)}
    fig.suptitle(f"{share:.1%} of the frame-to-frame steps of a SLEAP track are implausible jumps "
                 f"(> {JUMP_BL} body lengths in 40 ms):\n{rate[1]:.1f} per 1000 steps in the dark, "
                 f"{rate[2]:.1f} in the light", x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_jumps.png")
    plt.close(fig)


def fig_dropout_durations(drop: pd.DataFrame) -> None:
    plt = _style()
    d = drop[~drop.censored]
    fig, ax = plt.subplots(figsize=(17 * CM, 7 * CM))
    bins = _frame_bins(600)
    width = np.diff(np.log10(bins))
    w = (d.missing_fish_frames / d.missing_fish_frames.sum()).to_numpy()
    for wt, c in ((np.full(len(d), 1 / len(d)), "0.3"), (w, "#d62728")):
        h, _ = np.histogram(d.dur_s, bins=bins, weights=wt)
        ax.stairs(h / width / 10, bins, color=c, lw=1.3)
    ax.set_xscale("log")
    _plain_log_ticks(ax.xaxis, (0.04, 0.1, 1, 10, 100))
    ax.set_xlabel("dropout duration: consecutive frames with fewer than two fish (s)")
    ax.set_ylabel("share per 0.1 decade")
    short = np.mean(d.dur_s < MIN_DROPOUT_S)
    t1 = w[(d.dur_s >= MIN_DROPOUT_S).to_numpy()].sum()
    t60 = w[(d.dur_s >= 60).to_numpy()].sum()
    ax.axvline(MIN_DROPOUT_S, color="0.5", ls=":", lw=1)
    ax.text(0.99, 0.95, f"grey: share of the {len(d):,} dropouts\nred: share of the missing fish-time",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_title(f"{short:.0%} of the dropouts are flicker shorter than 1 s, but {t1:.0%} of the missing fish-time "
                 f"is in dropouts of 1 s or longer ({t60:.0%} in >= 60 s)", fontsize=8.5, fontweight="bold")
    _save(fig, "07_dropout_durations.png")
    plt.close(fig)


def _object_outlines(ax, objs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    for o in objs[objs.kind.isin(["tube", "block"])].itertuples():
        ax.add_patch(plt.Rectangle((o.x0, o.y0), o.x1 - o.x0, o.y1 - o.y0, fill=False,
                                   ec="#2ca02c", lw=0.8))


def fig_dropout_locations(inst: pd.DataFrame, drop: pd.DataFrame, objs: pd.DataFrame, ref: str) -> None:
    import cv2
    import matplotlib.colors as mcolors

    plt = _style()
    bg = cv2.imread(str(BACKGROUND / f"{ref}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = bg.shape
    d = drop[~drop.censored & (drop.dur_s >= MIN_DROPOUT_S) & drop.phase.isin([1, 2])]
    lost = np.r_[np.c_[d.lost1_x, d.lost1_y], np.c_[d.lost2_x, d.lost2_y]]
    new = np.r_[np.c_[d.new1_x, d.new1_y], np.c_[d.new2_x, d.new2_y]]
    det = inst[inst.phase.isin([1, 2])][["centroid_x", "centroid_y"]].to_numpy()
    a = objs[objs.recording == ref].iloc[0]
    x0, x1, y0, y1 = a.arena_x0 - 60, a.arena_x1 + 60, a.arena_y0 - 60, a.arena_y1 + 60
    fig, axes = plt.subplots(1, 3, figsize=(17 * CM, 5.0 * CM))
    xb, yb = np.arange(0, w + 1, 32), np.arange(0, h + 1, 32)
    for ax, pts, title in zip(axes, (det, lost, new), (
            f"all detections\n(n = {len(det):,})", f"last position before a dropout >= 1 s\n(n = "
            f"{np.isfinite(lost[:, 0]).sum():,})", f"first position after it\n(n = {np.isfinite(new[:, 0]).sum():,})")):
        pts = pts[np.isfinite(pts[:, 0])]
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=(xb, yb))
        H = H.T / H.sum()
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255, extent=(0, w, h, 0))
        im = ax.imshow(np.ma.masked_where(H <= 0, H), cmap="magma", extent=(0, w, h, 0), alpha=0.9,
                       interpolation="nearest", norm=mcolors.LogNorm(vmin=1e-5, vmax=3e-2))
        _object_outlines(ax, objs[objs.recording == ref])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
    cb.set_label("share of the panel's positions per 32 x 32 px bin (log scale)")
    fig.suptitle("Where SLEAP loses fish and finds them again: at the tube, the four blocks (green outlines) and "
                 "the walls\n(all 19 recordings pooled, dark and light phase; background: a frame of "
                 f"{ref})", x=0.01, ha="left", fontsize=8.5, fontweight="bold", y=1.02)
    _save(fig, "08_dropout_locations_map.png")
    plt.close(fig)


def fig_location_shares(drop: pd.DataFrame) -> None:
    plt = _style()
    base = pd.read_csv(TABLES / "detections_by_location.csv")
    d = drop[~drop.censored & (drop.dur_s >= MIN_DROPOUT_S) & drop.phase.isin([1, 2])]
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.8 * CM), sharey=True)
    shares = {}
    for ax, ph in zip(axes, (1, 2)):
        b = base[base.phase == ph].groupby("loc").n.sum()
        b = b / b.sum()
        dd = d[d.phase == ph]
        lost = pd.concat([dd.lost1_loc, dd.lost2_loc])
        lost = lost[lost != ""].value_counts(normalize=True)
        new = pd.concat([dd.new1_loc, dd.new2_loc])
        new = new[new != ""].value_counts(normalize=True)
        shares[ph] = (b, lost)
        x = np.arange(len(LOCATIONS))
        for k, (sh, c, lab) in enumerate(((b, "0.6", "all detections"), (lost, "#d62728", "lost (last seen)"),
                                          (new, "#2ca02c", "reappeared (first seen)"))):
            v = [sh.get(loc, 0) for loc in LOCATIONS]
            ax.bar(x + (k - 1) * 0.27, v, width=0.27, color=c, label=lab)
            for xi, vi in zip(x + (k - 1) * 0.27, v):
                ax.text(xi, vi + 0.01, f"{vi:.0%}", ha="center", fontsize=5.5)
        ax.set_xticks(x, ["at the tube", "at a block", "at the edge", "open water"])
        ax.set_title("dark phase" if ph == 1 else "light phase")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("share")
    axes[0].legend(loc="upper right")
    (b1, l1), (b2, l2) = shares[1], shares[2]
    fig.suptitle(f"The tube is where fish vanish: {l1.get('tube', 0):.0%} (dark) and {l2.get('tube', 0):.0%} "
                 f"(light) of the losses, but {b1.get('tube', 0):.0%} and {b2.get('tube', 0):.0%} of the "
                 f"detections are there\n(dropouts >= 1 s; 'at' = within {LOC_MARGIN_BL} body lengths of the "
                 "object's outline or of the edge of the bright floor)", x=0.01, ha="left", fontsize=8.5,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "09_dropout_location_shares.png")
    plt.close(fig)


def fig_video_time(recs: pd.DataFrame, labels: dict) -> None:
    plt = _style()
    fig, ax = plt.subplots(figsize=(17 * CM, 8 * CM))
    y = np.arange(len(recs))[::-1]
    ax.axvspan(0, LIGHT_ON_S / 3600, color="0.9", zorder=0)
    ax.text(LIGHT_ON_S / 7200, len(recs) - 0.2, "dark phase (meta.csv)", ha="center", fontsize=7)
    for yi, r in zip(y, recs.itertuples()):
        ax.plot([0, r.rec_dur_s / 3600], [yi + 0.18] * 2, color="0.2", lw=2.5, solid_capstyle="butt")
        ax.plot([r.t_first_frame_s / 3600, r.t_last_frame_s / 3600], [yi - 0.18] * 2, color=C_TWO, lw=2.5,
                solid_capstyle="butt")
        ax.plot(r.light_on_measured_s / 3600, yi, "|", color=C_LIGHT, ms=9, mew=1.5)
    ax.set_yticks(y, [labels[r] for r in recs.recording])
    ax.set_xlabel("recording time (h)")
    ax.set_xlim(-0.1, 4.1)
    ax.plot([], [], color="0.2", lw=2.5, label="electric recording")
    ax.plot([], [], color=C_TWO, lw=2.5, label="video (LED map)")
    ax.plot([], [], "|", color=C_LIGHT, ms=9, mew=1.5, label="light on, measured in the video")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    start = -recs.t_first_frame_s
    ax.set_title(f"Every video starts {start.min() / 60:.1f}-{start.max() / 60:.1f} min before the electric "
                 "recording and ends after it: the whole dark phase is filmed", fontsize=9,
                 fontweight="bold", pad=18)
    _save(fig, "10_video_vs_recording_time.png")
    plt.close(fig)


def fig_led_residuals(recs: pd.DataFrame, labels: dict) -> None:
    plt = _style()
    fig, axes = plt.subplots(4, 5, figsize=(17 * CM, 12 * CM), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, recs.itertuples()):
        tr = np.load(TABLES / "led_resid" / f"{r.recording}.npy")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.plot(tr[:, 0] / 3600, tr[:, 1] * 1e3, ".", ms=0.8, color=C_TWO, rasterized=True)
        ax.set_title(f"{labels[r.recording]}: {r.led_resid_med_ms:.0f} / {r.led_resid_p95_ms:.0f} ms", fontsize=6.5)
        ax.set_ylim(-60, 60)
    for ax in axes.flat[len(recs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("LED residual (ms)")
    for ax in axes[-1]:
        ax.set_xlabel("recording time (h)")
    axes[-2, -1].xaxis.set_tick_params(labelbottom=True)
    axes[-2, -1].set_xlabel("recording time (h)")
    fig.suptitle("The LED map (led_alignment.fit_led_alignment) fits all 19 recordings to within about one video "
                 "frame (40 ms)\nresidual = mapped video flash minus the nearest flash in led_idxs.csv; panel titles: "
                 "median / 95th percentile of |residual|;\nthe saw-tooth is the video drifting by up to half a frame "
                 "and then jumping back by one frame", x=0.01, ha="left", fontsize=8, fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_led_residuals.png")
    plt.close(fig)


def fig_b2_yield(labels: dict) -> None:
    plt = _style()
    m = pd.read_csv(TABLES / "pairs_and_mining.csv")
    fig, ax = plt.subplots(figsize=(17 * CM, 8.5 * CM))
    y = np.arange(len(m))[::-1]
    for col, c, lab, dy in (("n_frames_mine_dark", "#9ecae1", "dark, any position", 0.2),
                            ("n_frames_clean_dark", C_DARK, "dark, both fish away from tube and blocks", 0.0),
                            ("n_frames_mine_light", C_LIGHT, "light, any position", -0.2)):
        v = m[col].to_numpy()
        ax.barh(y + dy, np.where(v > 0, v, np.nan), height=0.2, color=c, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 2e4)
    ax.set_yticks(y, [labels[r] for r in m.recording])
    ax.set_xlabel("video frames with two minable fish (log scale; no bar = none)")
    _plain_log_ticks(ax.xaxis, (1, 10, 100, 1000, 10000))
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncols=3)
    ax.set_title(f"Mining for B2: {m.n_frames_mine.sum():,} frames with two confident, well-separated fish, "
                 f"{m.n_frames_mine_light.sum():,} of them in the light\n"
                 "(each fish: instance, head and tail score >= 0.5, head-tail >= 0.5 BL; pair: > 2 BL apart, "
                 "no shared keypoint)", fontsize=9, fontweight="bold")
    _save(fig, "12_minable_two_fish_frames.png")
    plt.close(fig)


def fig_light_on(recs: pd.DataFrame, labels: dict) -> None:
    plt = _style()
    fig, axes = plt.subplots(4, 5, figsize=(17 * CM, 11 * CM), sharex=True)
    for ax, r in zip(axes.flat, recs.itertuples()):
        base = np.load(TABLES / "frame_time" / f"{r.recording}_led_roi_baseline.npy")
        t = np.load(TABLES / "frame_time" / f"{r.recording}.npy")[np.arange(len(base)) * 25 + 12]
        sel = (t > 9600) & (t < 12000)
        ax.plot(t[sel] / 60, base[sel], color="0.3", lw=0.6)
        ax.axvline(LIGHT_ON_S / 60, color="0.6", ls="--", lw=0.8)
        ax.axvline(r.light_on_measured_s / 60, color=C_LIGHT, lw=1)
        d = (r.light_on_measured_s - LIGHT_ON_S) / 60
        ax.set_title(f"{labels[r.recording]}: {d:+.1f} min", fontsize=6.5)
    for ax in axes.flat[len(recs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("LED-region\nbrightness")
    for ax in axes[-1]:
        ax.set_xlabel("recording time (min)")
    axes[-2, -1].xaxis.set_tick_params(labelbottom=True)
    axes[-2, -1].set_xlabel("recording time (min)")
    d = (recs.light_on_measured_s - LIGHT_ON_S) / 60
    late = d[d > 0]
    fig.suptitle(f"The room light comes on {late.min():.1f}-{late.max():.1f} min after the 180 min of meta.csv "
                 f"(grey dashed) in {len(late)} recordings and {-d.min():.1f} min before it in 07-05 (started at 10:10)"
                 "\norange: measured step of the background brightness around the LED (LED_val.npy, per-second "
                 "minimum); panel titles: measured minus 180 min", x=0.01, ha="left", fontsize=8, fontweight="bold")
    fig.tight_layout()
    _save(fig, "13_light_on.png")
    plt.close(fig)


SHEETS = {
    "14_examples_missed_but_visible.png": (
        ("visible_missed", "partly_visible_missed"),
        "Visible but missed: in {n} of {m} judged missing fish, the fish is in view (whole or partly out from under "
        "a block) and SLEAP has no instance on it\n(crops 480 x 480 px around the fish; coloured keypoints = "
        "SLEAP's instances in the frame)"),
    "15_examples_hidden.png": (
        ("hidden_tube", "hidden_block", "hidden_other"),
        "Hidden: in {n} of {m} judged missing fish, no fish is visible anywhere in the frame; the red x marks where "
        "it was last seen 0.5 s earlier\n(crops 480 x 480 px around that spot; most are at the tube or a block)"),
    "16_examples_close_pairs.png": (
        ("duplicate_one_fish", "two_real_fish"),
        "Two instances less than 1 body length apart: in {n} of {m} frames both instances are on one and the same "
        "fish\n(crops 480 x 480 px around the first instance; blue and orange = the two instances)"),
}
JUDGEMENT_LABEL = {"visible_missed": "visible, missed", "partly_visible_missed": "partly visible, missed",
                   "hidden_tube": "hidden, tube", "hidden_block": "hidden, block", "hidden_other": "hidden",
                   "duplicate_one_fish": "one fish twice", "two_real_fish": "two real fish", "unclear": "unclear"}


def fig_example_sheets() -> None:
    import cv2

    plt = _style()
    ex = pd.read_csv(EXAMPLES / "examples.csv").set_index("example")
    jd = pd.read_csv(EXAMPLES / "judgements.csv")
    labels = rec_labels()
    missing = jd[jd.fish.isin(["missing", "fish_a", "fish_b"])]
    for name, (kinds, title) in SHEETS.items():
        sel = jd[jd.judgement.isin(kinds)]
        if "two_real_fish" in kinds:
            sel = jd[jd.example.isin(ex.index[ex.category == "two_close"])]
            n, m = int((sel.judgement == "duplicate_one_fish").sum()), len(sel)
        else:
            n, m = len(sel), len(missing)
        if "two_real_fish" not in kinds:  # a fish without its own last-seen position cannot be shown
            own = sel.look_x.notna() | (sel.fish != "fish_b")
            n_skip = int((~own).sum())
            sel = sel[own]
            if n_skip:
                title = title.replace("\n(", f"\n({n_skip} judged fish without a last-seen position not shown; ")
        ncol = 4
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(17 * CM, 4.6 * CM * nrow + 1.4 * CM), squeeze=False)
        for ax, j in zip(axes.flat, sel.itertuples()):
            e = ex.loc[j.example]
            gray = cv2.imread(str(EXAMPLES / "frames" / f"{e.recording}_{int(e.frame)}.png"), cv2.IMREAD_GRAYSCALE)
            inst = pd.read_parquet(TABLES / "instances" / f"{e.recording}.parquet",
                                   filters=[("frame", "==", int(e.frame))])
            if np.isfinite(j.look_x):
                cx, cy = j.look_x, j.look_y
            elif j.fish == "fish_b" and np.isfinite(e.lost2_x):
                cx, cy = e.lost2_x, e.lost2_y
            else:
                cx, cy = e.focus_x, e.focus_y
            if e.category.startswith("two") and len(inst):
                cx, cy = inst.centroid_x.iloc[0], inst.centroid_y.iloc[0]
            h, w = gray.shape
            x0, y0 = _crop_box(cx, cy, w, h)
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            for k, (_, r) in enumerate(inst.iterrows()):
                c = INSTANCE_COLORS[k % 2]
                ax.plot([r[f"{nd}_x"] for nd in NODES], [r[f"{nd}_y"] for nd in NODES], "-", color=c, lw=0.8)
                for nd, mk in zip(NODES, ("o", "s", "^")):
                    ax.plot(r[f"{nd}_x"], r[f"{nd}_y"], mk, color=c, ms=3.5, mfc="none", mew=0.9)
            if e.category in ("one_fish", "no_fish"):
                for tag in ("focus", "lost2"):
                    if np.isfinite(e[f"{tag}_x"]):
                        ax.plot(e[f"{tag}_x"], e[f"{tag}_y"], "x", color="red", ms=6, mew=1.3)
            ax.set_xlim(x0, x0 + CROP)
            ax.set_ylim(y0 + CROP, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{j.example} {labels[e.recording]} {PHASES[int(e.phase)]}\n"
                         f"{JUDGEMENT_LABEL[j.judgement]}", fontsize=6.5)
        for ax in axes.flat[len(sel):]:
            ax.set_visible(False)
        fig.suptitle(title.format(n=n, m=m), x=0.01, ha="left", fontsize=7.5, fontweight="bold")
        fig.tight_layout()
        _save(fig, name)
        plt.close(fig)


FIGURE_FUNCS = ("coverage", "coverage_time", "video_time", "led", "b2", "tracklets", "dropout_durations",
                "location_shares", "scores", "nodes", "jumps", "dropout_map", "light_on", "sheets")


def stage_figures(only: list[str] | None = None) -> None:
    want = set(only or FIGURE_FUNCS)
    recs = pd.read_csv(TABLES / "recordings.csv")
    labels = rec_labels()
    drop = pd.read_parquet(TABLES / "dropouts.parquet")
    objs = pd.read_csv(TABLES / "arena_objects.csv")
    simple = {"coverage": lambda: fig_coverage(recs, labels), "coverage_time": lambda: fig_coverage_time(recs, labels),
              "video_time": lambda: fig_video_time(recs, labels), "led": lambda: fig_led_residuals(recs, labels),
              "b2": lambda: fig_b2_yield(labels), "light_on": lambda: fig_light_on(recs, labels),
              "tracklets": lambda: fig_tracklets(pd.read_parquet(TABLES / "tracklets.parquet"), recs),
              "dropout_durations": lambda: fig_dropout_durations(drop),
              "location_shares": lambda: fig_location_shares(drop), "sheets": fig_example_sheets}
    for k, f in simple.items():
        if k in want:
            f()
    if want & {"scores", "nodes", "jumps", "dropout_map"}:
        inst = load_instances(["phase", "n_in_frame", "instance_score", "head_score", "center_score", "tail_score",
                               "jump", "step_px", "centroid_x", "centroid_y"])
        if "scores" in want:
            fig_instance_scores(inst, drop)
        if "nodes" in want:
            fig_node_scores(inst)
        if "jumps" in want:
            fig_jumps(inst, recs, labels)
        if "dropout_map" in want:
            fig_dropout_locations(inst, drop, objs, recs.recording.iloc[0])


# ----------------------------------------------------------------------------- report
FIGURE_NOTES = {
    "00_arena_objects.png": "the tube and the four blocks found in one light-phase frame per recording (QC of "
                            "the object detection used for the location classes).",
    "01_coverage_per_recording.png": "share of frames with two, one or no fish, per recording, dark and light phase.",
    "02_coverage_over_time.png": "two-fish share per minute of recording time for all recordings; light-on marked.",
    "03_instance_scores_by_context.png": "instance-score distributions of detections in two-/one-fish frames, "
                                         "before/after dropouts and after jumps.",
    "04_node_scores.png": "how often each node is missing and the score distribution of the placed nodes.",
    "05_tracklet_durations.png": "duration of SLEAP's track ids, by track and weighted by detections.",
    "06_jumps.png": "frame-to-frame step distribution of the tracks and the jump rate per recording (switch proxy).",
    "07_dropout_durations.png": "durations of runs with fewer than two fish, by count and by missing fish-time.",
    "08_dropout_locations_map.png": "maps of all detections, of where fish were last seen before a dropout and of "
                                    "where they reappear.",
    "09_dropout_location_shares.png": "share of losses / reappearances at the tube, a block, the edge, open water, "
                                      "against where the detections are.",
    "10_video_vs_recording_time.png": "the span of every video in recording time (LED map) against the electric "
                                      "recording and the dark phase.",
    "11_led_residuals.png": "LED-map residuals over the recording, per recording.",
    "12_minable_two_fish_frames.png": "frames per recording with two confident, well-separated fish (B2 mining yield).",
    "13_light_on.png": "the light-on step measured in the video against meta.csv's 180 min.",
    "14_examples_missed_but_visible.png": "example crops of fish that are visible but have no SLEAP instance.",
    "15_examples_hidden.png": "example crops where the missing fish is not visible (tube, blocks).",
    "16_examples_close_pairs.png": "example crops of two instances < 1 BL apart (mostly one fish detected twice).",
}


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f} %"


def stage_report() -> None:
    recs = pd.read_csv(TABLES / "recordings.csv")
    mine = pd.read_csv(TABLES / "pairs_and_mining.csv").set_index("recording")
    trk = pd.read_parquet(TABLES / "tracklets.parquet")
    drop = pd.read_parquet(TABLES / "dropouts.parquet")
    base = pd.read_csv(TABLES / "detections_by_location.csv")
    inv = pd.read_csv(TABLES / "sleap_tracking_inventory.csv")
    ex = pd.read_csv(EXAMPLES / "examples.csv")
    jd = pd.read_csv(EXAMPLES / "judgements.csv")
    labels = rec_labels()
    inst = load_instances(["phase", "step_px", "jump", "swap_like", "instance_score", "head_score", "center_score",
                           "tail_score"])
    S: dict[str, float] = {}  # every number in the report

    # --- pooled numbers
    for ph in ("dark", "light", "all"):
        w = recs[f"frames_{ph}"]
        for k in (0, 1, 2):
            S[f"share{k}_{ph}"] = float((recs[f"share{k}_{ph}"] * w).sum() / w.sum())
            S[f"share{k}_{ph}_min"] = float(recs[f"share{k}_{ph}"].min())
            S[f"share{k}_{ph}_max"] = float(recs[f"share{k}_{ph}"].max())
    for ph in ("dark", "light"):
        n2 = mine[f"n_two_frames_{ph}"].sum()
        S[f"dup_shared_{ph}"] = float(mine[f"n_two_frames_shared_node_{ph}"].sum() / n2)
        S[f"dup_close_{ph}"] = float(mine[f"n_two_frames_closer_0p5bl_{ph}"].sum() / n2)
        S[f"share2_distinct_{ph}"] = S[f"share2_{ph}"] * (1 - S[f"dup_shared_{ph}"])
    S["dup_shared_min"] = float(mine.share_two_frames_shared_node.min())
    S["dup_shared_max"] = float(mine.share_two_frames_shared_node.max())
    S["n_instances"] = float(recs.n_instances.sum())
    S["n_tracks"] = float(len(trk))
    S["tracks_per_h_min"], S["tracks_per_h_max"] = float(recs.tracks_per_video_hour.min()), float(recs.tracks_per_video_hour.max())
    S["track_median_s"] = float(trk.span_s.median())
    S["track_q25_s"], S["track_q75_s"] = float(trk.span_s.quantile(0.25)), float(trk.span_s.quantile(0.75))
    S["track_wmedian_s"] = weighted_median(trk.span_s, trk.n_instances)
    S["track_share_under_1s"] = float((trk.span_s < 1).mean())
    S["det_in_tracks_over_60s"] = float(trk.loc[trk.span_s > 60, "n_instances"].sum() / trk.n_instances.sum())
    S["track_max_s"] = float(trk.span_s.max())
    st = inst[np.isfinite(inst.step_px)]
    S["n_steps"] = float(len(st))
    S["jump_share"] = float(st.jump.mean())
    for ph, k in (("dark", 1), ("light", 2)):
        S[f"jumps_per_1000_{ph}"] = float(1000 * st.jump[st.phase == k].mean())
    S["swap_share"] = float(inst.swap_like.sum() / inst.jump.sum())
    S["bigjump_per_1000"] = float(1000 * recs.n_big_jumps.sum() / recs.n_steps.sum())
    d = drop[~drop.censored]
    S["n_dropouts"] = float(len(d))
    S["dropout_share_under_1s"] = float((d.dur_s < MIN_DROPOUT_S).mean())
    w = d.missing_fish_frames / d.missing_fish_frames.sum()
    S["missing_time_in_over_1s"] = float(w[d.dur_s >= 1].sum())
    S["missing_time_in_over_60s"] = float(w[d.dur_s >= 60].sum())
    S["longest_dropout_s"] = float(d.dur_s.max())
    S["dropout_over_1s_median_s"] = float(d.dur_s[d.dur_s >= 1].median())
    dl = d[(d.dur_s >= 1) & d.phase.isin([1, 2])]
    for ph, k in (("dark", 1), ("light", 2)):
        b = base[base.phase == k].groupby("loc").n.sum()
        b = b / b.sum()
        x = dl[dl.phase == k]
        lost = pd.concat([x.lost1_loc, x.lost2_loc])
        lost = lost[lost != ""].value_counts(normalize=True)
        for loc in LOCATIONS:
            S[f"det_at_{loc}_{ph}"] = float(b.get(loc, 0))
            S[f"lost_at_{loc}_{ph}"] = float(lost.get(loc, 0))
    tube_det = base[base["loc"] == "tube"]
    S["single_node_at_tube"] = float((tube_det.single_node * tube_det.n).sum() / tube_det.n.sum())
    S["led_med_min"], S["led_med_max"] = float(recs.led_resid_med_ms.min()), float(recs.led_resid_med_ms.max())
    S["led_p95_min"], S["led_p95_max"] = float(recs.led_resid_p95_ms.min()), float(recs.led_resid_p95_ms.max())
    S["video_start_min_s"], S["video_start_max_s"] = float(recs.t_first_frame_s.min()), float(recs.t_first_frame_s.max())
    S["video_end_after_rec_min_s"] = float((recs.t_last_frame_s - recs.rec_dur_s).min())
    S["video_end_after_rec_max_s"] = float((recs.t_last_frame_s - recs.rec_dur_s).max())
    S["n_led_ok"] = float((recs.led_status == "ok").sum())
    S["n_covers_dark"] = float(recs.video_covers_dark.sum())
    S["n_covers_recording"] = float(recs.video_covers_recording.sum())
    S["grid_pauses_matched_share"] = float(recs.grid_pauses_matched.sum() / recs.grid_pauses.sum())
    lon = (recs.light_on_measured_s - LIGHT_ON_S) / 60
    S["light_on_delay_min_min"], S["light_on_delay_max_min"] = float(lon[lon > 0].min()), float(lon[lon > 0].max())
    S["light_on_0705_min"] = float(lon[recs.recording == "2022-07-05-10_10"].iloc[0])
    S["bl_px_min"], S["bl_px_max"] = float(recs.body_length_px.min()), float(recs.body_length_px.max())
    S["n_mine"], S["n_mine_light"] = float(mine.n_frames_mine.sum()), float(mine.n_frames_mine_light.sum())
    S["n_clean"], S["n_clean_light"] = float(mine.n_frames_clean.sum()), float(mine.n_frames_clean_light.sum())
    S["n_mine_rec_min"], S["n_mine_rec_max"] = float(mine.n_frames_mine.min()), float(mine.n_frames_mine.max())
    S["n_clean_rec_min"], S["n_clean_rec_max"] = float(mine.n_frames_clean.min()), float(mine.n_frames_clean.max())
    S["n_rec_clean_under_100"] = float((mine.n_frames_clean < 100).sum())
    S["collapsed_at_tube"] = float(mine.n_collapsed_at_tube.sum() / mine.n_instances_head_tail_at_tube.sum())
    S["collapsed_all"] = float(mine.n_collapsed.sum() / mine.n_instances_head_tail.sum())
    S["frames_over_2"] = float((recs.share3plus_all * recs.frames_all).sum())
    S["video_h_min"], S["video_h_max"] = float(recs.video_h.min()), float(recs.video_h.max())
    S["rec_h"] = float(recs.rec_dur_s.iloc[0] / 3600)
    S["share_tracks_with_gaps"] = float((trk.n_gaps > 0).mean())
    S["max_gap_frames"] = float(trk.max_gap_frames.max())
    S["tracks_per_h_dark_min"], S["tracks_per_h_dark_max"] = float(recs.tracks_per_hour_dark.min()), float(recs.tracks_per_hour_dark.max())
    S["tracks_per_h_light_min"], S["tracks_per_h_light_max"] = float(recs.tracks_per_hour_light.min()), float(recs.tracks_per_hour_light.max())
    for n in NODES:
        for ph, k in (("dark", 1), ("light", 2)):
            S[f"missing_{n}_{ph}"] = float(inst.loc[inst.phase == k, f"{n}_score"].isna().mean())
    objs = pd.read_csv(TABLES / "arena_objects.csv")
    big = objs[objs.kind.isin(["tube", "block"])].copy()
    big["slot"] = big.groupby("recording").cx.rank().astype(int) * 10 + big.groupby("recording").cy.rank().astype(int)
    dev = big.groupby("slot")[["cx", "cy"]].transform(lambda v: v - v.median())
    S["object_shift_max_px"] = float(np.hypot(dev.cx, dev.cy).max())
    # visual judgements
    missing = jd[jd.fish.isin(["missing", "fish_a", "fish_b"])]
    cnt = missing.judgement.value_counts()
    S["n_judged_missing"] = float(len(missing))
    for k in ("hidden_tube", "hidden_block", "hidden_other", "visible_missed", "partly_visible_missed", "unclear"):
        S[f"judged_{k}"] = float(cnt.get(k, 0))
    close = jd[jd.example.isin(ex.example[ex.category == "two_close"])]
    conf = jd[jd.example.isin(ex.example[ex.category == "two_confident"])]
    S["n_close"], S["close_duplicates"] = float(len(close)), float((close.judgement == "duplicate_one_fish").sum())
    S["n_conf"] = float(len(conf))
    S["conf_partly_hidden"] = float((conf.judgement == "two_real_fish_partly_hidden").sum())
    lab = inv[inv["item"] == "labels.v000.slp"]
    S["hand_labelled_frames"] = float(lab.labelled_frames.sum())
    S["hand_user_instances"] = float(lab.user_instances.sum())
    pd.Series(S).to_csv(OUT / "summary_numbers.csv", header=["value"])

    # --- per-recording table
    rows = []
    for r in recs.itertuples():
        m = mine.loc[r.recording].to_dict()
        m = {k: int(v) if k.startswith("n_") else v for k, v in m.items()}
        rows.append(
            f"| {labels[r.recording]} | {r.t_first_frame_s:.0f} / {r.t_last_frame_s:.0f} | "
            f"{r.led_resid_med_ms:.0f} / {r.led_resid_p95_ms:.0f} | {(r.light_on_measured_s - LIGHT_ON_S) / 60:+.1f} | "
            f"{pct(r.share2_dark)} / {pct(r.share1_dark)} / {pct(r.share0_dark)} | "
            f"{pct(r.share2_light)} / {pct(r.share1_light)} / {pct(r.share0_light)} | "
            f"{pct(m['share_two_frames_shared_node'])} | {r.tracks_per_video_hour:.0f} | {r.track_span_s_median:.1f} / "
            f"{r.track_span_s_detection_weighted_median:.0f} | {r.jumps_per_1000_steps_dark:.1f} / "
            f"{r.jumps_per_1000_steps_light:.1f} | {r.n_dropouts_over_1s} | {m['n_frames_mine_dark']:,} / "
            f"{m['n_frames_mine_light']:,} | {m['n_frames_clean']:,} ({m['n_minutes_clean']} min) |")
    table = "\n".join([
        "| recording (pair) | video start / end (s, recording time) | LED residual median / p95 (ms) | light on "
        "vs 180 min (min) | dark: 2 / 1 / 0 fish | light: 2 / 1 / 0 fish | 2-instance frames with a shared "
        "keypoint | track ids per video hour | track duration median / detection-weighted median (s) | jumps per "
        "1000 steps dark / light | dropouts >= 1 s | minable frames dark / light | of which both fish away from "
        "tube and blocks |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|", *rows])

    jt = "\n".join([
        "| judgement of the missing fish | n |", "|---|---|",
        f"| hidden: not visible anywhere, last seen at the tube | {S['judged_hidden_tube']:.0f} |",
        f"| hidden: not visible anywhere, last seen at a block | {S['judged_hidden_block']:.0f} |",
        f"| hidden: not visible anywhere, last seen elsewhere (open water / wall, within ~1 BL of the tube or "
        f"a block in most) | {S['judged_hidden_other']:.0f} |",
        f"| visible (whole fish) but no instance | {S['judged_visible_missed']:.0f} |",
        f"| partly visible (sticking out from under a block) but no instance | {S['judged_partly_visible_missed']:.0f} |",
        f"| unclear | {S['judged_unclear']:.0f} |",
        f"| **total** | **{S['n_judged_missing']:.0f}** |"])
    n_hidden = S["judged_hidden_tube"] + S["judged_hidden_block"] + S["judged_hidden_other"]
    n_vis = S["judged_visible_missed"] + S["judged_partly_visible_missed"]
    n_one = int((ex.category == "one_fish").sum())
    n_none = int((ex.category == "no_fish").sum())
    figs = "\n".join(f"- `figures/{k}`: {v}" for k, v in FIGURE_NOTES.items() if (FIGURES / k).exists())
    vids = inv[inv["item"] == "labels.v000.slp"]
    vids = ", ".join(f"{v.video} ({v.labelled_frames:.0f})" for v in vids.itertuples() if v.labelled_frames > 0)
    models = inv[inv["item"] == "model"]
    n_ident = int(inv.loc[inv["item"] == "export", "identical_to_interrim"].sum())

    md = f"""# B1: assessment of the SLEAP export of the 2022 competition videos

Milestone B1 of `docs/spec_2022_dataset.md`. Code: `experiments/video2022/` (`sleap_nix.py` reader + test,
`recordings2022.py`, `b1_assess_sleap.py`). Every number below is computed by `b1_assess_sleap.py report` from the
tables in `tables/` (pooled numbers also in `summary_numbers.csv`), except the example-frame counts, which are my
visual judgement of the rendered frames (`examples/judgements.csv`, one line per fish, with a note).

Definitions: *recording time* = seconds of the electric recording (sample / 20 kHz) from the LED map; *dark* =
recording time 0-10800 s, *light* = 10800 s to the end of the electric recording (14420 s), as in meta.csv;
*BL* = body length, the median head-centre-tail length of confident instances ({S['bl_px_min']:.0f}-{S['bl_px_max']:.0f} px
depending on the recording); *jump* = a track moving > {JUMP_BL} BL between two consecutive frames (40 ms);
*dropout* = a run of frames with fewer than two instances; *at the tube / a block / the edge* = within
{LOC_MARGIN_BL} BL of the object's bounding box or of the bright floor's edge.

## In short

1. **Video time is solved.** The LED map (`paper/scripts/led_alignment.py`) passes `check_led_alignment` in
   {S['n_led_ok']:.0f} of 19 recordings: median |residual| {S['led_med_min']:.0f}-{S['led_med_max']:.0f} ms, 95th
   percentile {S['led_p95_min']:.0f}-{S['led_p95_max']:.0f} ms (one frame = 40 ms); {pct(S['grid_pauses_matched_share'])}
   of the 2-s pauses of the LED train line up, so no whole-second offset. Every video starts
   {-S['video_start_max_s'] / 60:.1f}-{-S['video_start_min_s'] / 60:.1f} min before the electric recording and ends
   {S['video_end_after_rec_min_s']:.0f}-{S['video_end_after_rec_max_s']:.0f} s after it: all {S['n_covers_recording']:.0f}
   videos cover the whole recording, including the whole 3-h dark phase. (The spec's "video shorter than the
   recording" is not the case: the videos are {S['video_h_min']:.2f}-{S['video_h_max']:.2f} h long, the electric
   recordings {S['rec_h']:.3f} h.)
2. **The light comes on later than meta.csv says**: {S['light_on_delay_min_min']:.1f}-{S['light_on_delay_max_min']:.1f} min
   after 180 min in 18 recordings (drifting later from May to July) and {-S['light_on_0705_min']:.1f} min *before* it in
   2022-07-05, which started at 10:10 instead of 10:00 (consistent with a light timer at a fixed clock time; my
   inference). Measured from the brightness step around the LED in `LED_val.npy` (figure 13).
3. **Coverage.** Dark phase: two fish in {pct(S['share2_dark'])} of the frames (recordings
   {pct(S['share2_dark_min'])}-{pct(S['share2_dark_max'])}), one in {pct(S['share1_dark'])}, none in {pct(S['share0_dark'])}.
   Light phase: two in {pct(S['share2_light'])} ({pct(S['share2_light_min'])}-{pct(S['share2_light_max'])}), one in
   {pct(S['share1_light'])}, none in {pct(S['share0_light'])}. Frames with more than two instances:
   {S['frames_over_2']:.0f} (apparently a two-instance limit in the SLEAP run). But some
   "two-fish" frames are one fish detected twice: {pct(S['dup_shared_dark'], 1)} (dark) and {pct(S['dup_shared_light'], 1)}
   (light) of the two-instance frames have two instances sharing a keypoint exactly (per recording
   {pct(S['dup_shared_min'])}-{pct(S['dup_shared_max'])}), a lower bound; {pct(S['dup_close_dark'], 1)} / {pct(S['dup_close_light'], 1)}
   have the two centroids < 0.5 BL apart. Without the shared-keypoint duplicates, two distinct fish are found in about
   {pct(S['share2_distinct_dark'])} of the dark and {pct(S['share2_distinct_light'])} of the light frames.
4. **Tracks are fragments.** {S['n_tracks']:,.0f} SLEAP track ids for 2 fish x 19 recordings
   ({S['tracks_per_h_min']:.0f}-{S['tracks_per_h_max']:.0f} new ids per video hour). Median track duration
   {S['track_median_s']:.1f} s (quartiles {S['track_q25_s']:.2f}-{S['track_q75_s']:.1f} s; {pct(S['track_share_under_1s'])}
   last < 1 s); half of all detections are in tracks shorter than {S['track_wmedian_s']:.0f} s; only
   {pct(S['det_in_tracks_over_60s'])} of the detections are in tracks longer than 1 min. New ids per hour:
   {S['tracks_per_h_dark_min']:.0f}-{S['tracks_per_h_dark_max']:.0f} in the dark, {S['tracks_per_h_light_min']:.0f}-{S['tracks_per_h_light_max']:.0f}
   in the light. {pct(S['share_tracks_with_gaps'])} of the tracks have gaps (frames without the fish inside the
   track), never longer than {S['max_gap_frames']:.0f} frames: SLEAP's tracker does not bridge longer gaps, it starts a
   new id.
5. **Switch proxy.** {pct(S['jump_share'], 1)} of the {S['n_steps']:,.0f} frame-to-frame steps of a track are jumps
   > {JUMP_BL} BL ({S['jumps_per_1000_dark']:.1f} per 1000 steps in the dark, {S['jumps_per_1000_light']:.1f} in the
   light; {S['bigjump_per_1000']:.1f} per 1000 are > 1 BL). {pct(S['swap_share'])} of the jumps land within 0.5 BL of
   where the other instance was one frame earlier, i.e. the track id moved onto the other fish (or its duplicate).
6. **Dropouts.** {S['n_dropouts']:,.0f} runs with fewer than two fish; {pct(S['dropout_share_under_1s'])} are flicker
   shorter than 1 s, but {pct(S['missing_time_in_over_1s'])} of the missing fish-time is in dropouts >= 1 s (median
   {S['dropout_over_1s_median_s']:.1f} s) and {pct(S['missing_time_in_over_60s'])} in dropouts >= 1 min (longest
   {S['longest_dropout_s']:.0f} s).
7. **Where fish are lost.** Dark: {pct(S['lost_at_tube_dark'])} of the losses (dropouts >= 1 s) are at the tube
   (where {pct(S['det_at_tube_dark'])} of the detections are), {pct(S['lost_at_edge_dark'])} at the arena edge
   ({pct(S['det_at_edge_dark'])} of the detections: the fish spend much time along the walls), {pct(S['lost_at_block_dark'])}
   at a block ({pct(S['det_at_block_dark'])}), {pct(S['lost_at_open_dark'])} in open water ({pct(S['det_at_open_dark'])}).
   Light: {pct(S['lost_at_tube_light'])} at the tube ({pct(S['det_at_tube_light'])} of detections),
   {pct(S['lost_at_block_light'])} at a block ({pct(S['det_at_block_light'])} of detections: in the light the fish sit
   under the blocks). Reappearances have the same distribution. {pct(S['single_node_at_tube'])} of the detections at the
   tube have a single node (a head or tail sticking out).
8. **Occlusion vs missed detection (visual judgement, n = {S['n_judged_missing']:.0f} missing fish in
   {n_one} one-fish + {n_none} no-fish frames, 0.5 s into dropouts >= 1 s, stratified over recordings, dark/light
   and the place of loss):** {n_hidden:.0f} hidden (no fish visible anywhere: {S['judged_hidden_tube']:.0f} last seen at
   the tube, {S['judged_hidden_block']:.0f} at a block, {S['judged_hidden_other']:.0f} elsewhere), {n_vis:.0f} visible
   but missed ({S['judged_visible_missed']:.0f} whole fish, mostly along the walls or in open water, several
   clearly visible; {S['judged_partly_visible_missed']:.0f} sticking out from under a block), {S['judged_unclear']:.0f}
   unclear. Of {S['n_close']:.0f} frames with two instances < 1 BL apart, {S['close_duplicates']:.0f} were one fish
   detected twice (in two of them the other fish was visible elsewhere and missed). Of {S['n_conf']:.0f}
   "two confident fish" frames all instances were on real fish, but in the {S['conf_partly_hidden']:.0f} light-phase
   ones both fish were partly hidden (under a block, or only the head out of the tube with head and tail keypoints
   on the same spot). In none of the judged frames did I see motion blur, darkness or reflections as the cause:
   the floor is bright and the fish are dark silhouettes in both phases.
9. **Scores.** The instance score (SLEAP's centroid confidence, >= 0.2) of detections right before a loss and right
   after a reappearance is low (figure 03), but low scores are common everywhere, so a threshold alone cannot clean
   the tracks: {pct(float((inst.loc[inst.phase == 1, 'instance_score'] < 0.3).mean()))} of all dark-phase detections
   score below 0.3. Nodes missing (dark / light): head {pct(S['missing_head_dark'])} / {pct(S['missing_head_light'])},
   centre {pct(S['missing_center_dark'])} / {pct(S['missing_center_light'])}, tail {pct(S['missing_tail_dark'])} /
   {pct(S['missing_tail_light'])}; placed head and tail nodes mostly score > 0.9 (figure 04).
10. **`sleap_tracking/`**: `labels.v000.slp` holds {S['hand_labelled_frames']:.0f} hand-labelled frames
    ({S['hand_user_instances']:.0f} user instances; videos: {vids}) with a **7-node skeleton** (snout, spine1-3, tail,
    fins), and the {len(models)} models in `models/` (n = 6, 31, 100 frames) use that skeleton. The NIX exports have
    a 3-node skeleton (head, center, tail) and name the project `/home/efish/2022_competition/competition.slp`,
    which is not in `sleap_tracking/`: **the model and the labels behind the exports are not in the dataset.**
    `predicted_videos/` holds byte-identical copies of the {n_ident} exports.

## Per recording

Dark/light coverage = share of frames with 2 / 1 / 0 instances. Track durations = first to last detection of a
track id. Minable frames: see "Implications for B2".

{table}

## Example frames (visual judgement)

Sampling (`select_examples`): {n_one} frames with one instance and {n_none} with none, 0.5 s after the start of a
dropout >= 1 s, stratified over dark/light and the place where the fish was last seen (tube, block, edge, open
water), recordings drawn round-robin; plus {int((ex.category == 'two_confident').sum())} frames with two confident
fish and {int((ex.category == 'two_close').sum())} with two instances < 1 BL apart. {len(ex)} frames in total from
{ex.recording.nunique()} recordings, read by seeking (`examples/frames/`). For each I looked at the three-panel
render (`examples/exNN_*.png`), a contrast-enhanced whole frame (`examples/scan/`) and, where needed, extra crops.

{jt}

So in this sample about half of the missing fish are hidden ({n_hidden:.0f} of {S['n_judged_missing']:.0f}), mostly in
the tube or under a block, and about two in five are in view and missed ({n_vis:.0f}); the missed visible fish are
typically low-contrast against the wall mesh or partly under a block, but some are fully visible in open water
(ex06, ex17, ex21). The sample is small (n = {S['n_judged_missing']:.0f}); the ratio is not a precise estimate.

## Figures

{figs}

## Implications for B2 (mining confident training data)

- **Yield.** {S['n_mine']:,.0f} frames have two minable fish (each: instance, head and tail score >= 0.5, head-tail
  >= 0.5 BL; the pair: centroids > 2 BL apart, no shared keypoint), {S['n_mine_rec_min']:.0f}-{S['n_mine_rec_max']:,.0f}
  per recording; only {S['n_mine_light']:.0f} of them are in the light. Requiring both fish away from the tube and the
  blocks leaves {S['n_clean']:,.0f} ({S['n_clean_rec_min']:.0f}-{S['n_clean_rec_max']:,.0f} per recording;
  {S['n_rec_clean_under_100']:.0f} recordings with < 100), {S['n_clean_light']:.0f} in the light. Consecutive frames are
  near duplicates, so the useful number is closer to the number of distinct minutes (table column "min"). Plenty for
  a first training set, spread over all 19 recordings, but almost entirely from the dark phase.
- **Do not mine on scores alone.** Two failure modes pass a score threshold: duplicates (two instances on one fish;
  remove frames with a shared keypoint or centroids < 0.5-1 BL apart) and collapsed poses (head and tail on the same
  spot: {pct(S['collapsed_at_tube'])} of the instances at the tube that have both head and tail have them
  < {COLLAPSED_BL} BL apart, {pct(S['collapsed_all'], 1)} of all such instances; require a minimum head-tail distance). Mined frames are also biased towards fish SLEAP already handles.
- **The centre node is unreliable** (missing in most light-phase detections); mined labels need the centre
  interpolated or must be filtered to all-three-node instances (fewer frames, see `recordings.csv`
  `n_frames_two_confident_separated_all3`).
- **What mining cannot cover (hand-annotation targets):** (1) the light phase as a whole, where fish sit under the
  blocks and in the tube; (2) fish partly under a block or with only the head out of the tube (how to label them:
  visible part only, or not at all, must be decided); (3) fish along the walls against the dark mesh (low contrast,
  often missed even when fully visible); (4) two fish close together or crossing (SLEAP mostly returns one fish twice
  there); (5) the fully visible open-water fish that SLEAP still misses. A useful labelling budget: a few hundred
  frames sampled from dropouts (figures 14-16 show the kind) plus close-pair frames, stratified by recording and
  phase.
- **Hidden fish are a real ceiling**: about half of the missing fish in the sample are not visible at all (tube,
  under blocks). No detector fixes that; tracklets will have to end and restart at the tube and the blocks, and
  analyses of chasing should treat "in the tube" as a state (dropout at the tube) rather than as missing data.

## Caveats and what was not done

- The exports name the unconverted video (`<date>_0.mp4`) as their source; only the `*converted.mp4` files are in
  `raw/`. Their frame counts agree with the export's metadata to within one frame, and in all {len(ex)} example
  frames the SLEAP keypoints lie on the fish, so the frame indices match to within a few frames; a one-frame offset
  cannot be excluded this way.
- The LED residuals show a saw-tooth of one-frame steps (figure 11); `fit_led_alignment` splits segments only at
  steps > 0.1 s, so the map is exact to about +-1 frame, which is enough for the paper's 0.5-s analyses.
- "Lost" and "reappeared" fish are identified by distance (no identities): the instance of the last two-fish frame
  farther from the instance that remains.
- Object positions come from one light-phase frame per recording (figure 00); the tube and block centres differ by
  at most {S['object_shift_max_px']:.0f} px between recordings, so the camera and the objects did not move.
- Body lengths are in pixels only; no px-to-cm calibration was attempted.
- The spec's "9,479 track ids" for 2022-05-10 is the length of the export's track map (including `none` and ids no
  instance uses); {int(trk[trk.recording == '2022-05-10-10_00'].track.nunique()):,} ids are used there. All track counts here are used ids.
- Dark/light follow meta.csv (10800 s). The video is still dark for the first 1-3 min of the "light" phase
  (figure 13; example ex12 is such a frame), and in 2022-07-05 the last 7 min of the "dark" phase are lit.
- The 2022-07-07 recording is the control pairing (meta2.csv / Versuchszettel: `2l4w`, lepto07 vs lepto16) and is
  included here; it is not in meta.csv.
- SLEAP was not re-run. Video frames decoded in total: {len(ex) + recs.shape[0] + 3 + 2} ({len(ex)} examples, {recs.shape[0]} background frames + 3 re-picked in the light, 2 test frames of 2022-05-10 while exploring), each by seeking.
"""
    (OUT / "report.md").write_text(md)
    print("wrote", OUT / "report.md", "and summary_numbers.csv")


# ----------------------------------------------------------------------------- sleap_tracking inventory
def stage_inventory() -> None:
    """What ``sleap_tracking/`` holds: hand-labelled frames, models, and whether the exports match."""
    import hashlib
    import json

    import h5py

    from paths import REC_2022

    root = REC_2022 / "sleap_tracking"
    rows = []
    with h5py.File(root / "labels.v000.slp", "r") as f:
        fr, ins = f["frames"][:], f["instances"][:]
        vids = [json.loads(v)["backend"]["filename"] for v in f["videos_json"][:]]
        nodes = [n["name"] for n in json.loads(f["metadata"].attrs["json"])["nodes"]]
        skel = json.loads(f["metadata"].attrs["json"])["skeletons"][0]["nodes"]
        user = ins[ins["instance_type"] == 0]
        n_user = np.bincount(user["frame_id"].astype(int), minlength=len(fr))
        for v, name in enumerate(vids):
            sel = fr["video"] == v
            rows.append(dict(item="labels.v000.slp", video=Path(name).name, labelled_frames=int(sel.sum()),
                             user_instances=int(n_user[fr["frame_id"][sel].astype(int)].sum()),
                             frames_with_more_than_2_user_instances=int(np.sum(n_user[fr["frame_id"][sel].astype(int)] > 2))))
    skeleton = [nodes[n["id"]] for n in skel]
    for m in sorted((root / "models").iterdir()):
        cfg = json.loads((m / "training_config.json").read_text())
        n_train = n_val = None
        for split in ("train", "val"):
            p = m / f"labels_gt.{split}.slp"
            if p.exists():
                with h5py.File(p, "r") as g:
                    if split == "train":
                        n_train = len(g["frames"])
                    else:
                        n_val = len(g["frames"])
        heads = [k for k, v in cfg["model"]["heads"].items() if v]
        rows.append(dict(item="model", video=m.name, model_head=";".join(heads), train_frames=n_train,
                         val_frames=n_val))

    def digest(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    for rec in list_recordings():
        copy = root / "predicted_videos" / rec.nix.name
        rows.append(dict(item="export", video=rec.nix.name, in_predicted_videos=copy.exists(),
                         identical_to_interrim=bool(copy.exists() and digest(copy) == digest(rec.nix))))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "sleap_tracking_inventory.csv", index=False)
    (TABLES / "sleap_tracking_skeleton.txt").write_text(
        "labels.v000.slp skeleton nodes: " + ", ".join(skeleton) + "\n"
        "all nodes in the file: " + ", ".join(nodes) + "\n"
        "NIX exports (interrim/*.analysis.nix): head, center, tail; project "
        "/home/efish/2022_competition/competition.slp (not in sleap_tracking/)\n")
    print(df.to_string())


# ----------------------------------------------------------------------------- main
STAGES = {"tables": stage_tables, "background": stage_background, "locations": stage_locations,
          "examples": stage_examples, "figures": stage_figures, "inventory": stage_inventory, "report": stage_report}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("only", nargs="*", help=f"figures stage only: some of {', '.join(FIGURE_FUNCS)}")
    args = ap.parse_args()
    if args.stage == "figures":
        stage_figures(args.only)
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
