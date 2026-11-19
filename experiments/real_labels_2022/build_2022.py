"""Build the 2022 labelling session real_v2_2022 (blind, real_v1's labeller and view).

Items:

1. **Candidates**: kept detections of ``detections/competition2022_v2_t001``
   with score >= 0.5, stratified by score bin (0.5-0.7, 0.7-0.8, 0.8-0.9,
   0.9-1) x the assigned fish's trial status (winner / loser / other: the
   ``Win_ID`` / ``Lose_ID`` track of ``raw/meta.csv``; unassigned rows, other
   tracks and 2022-07-07-10_00, which has no trial, are "other") x group
   (pair 3 = 2022-06-20/22/24, rest). Up to ``N_PER_CELL`` items per cell, as
   a simple random sample pooled over the cell's recordings (not spread over
   recordings as in real_v1), split into small and big chirps (decoder
   height < / >= 200 Hz, ``build_session.size_of``): each size gets its
   share of the cell's pool but at least ``MIN_PER_SIZE`` (in 2022 only about 4 %
   of these detections are big chirps; real_v1's even split would put half
   the items on them and cut a typical cell's effective sample size from
   about 12-15 to 8-10, see ``build_summary.md``). Sampling strata are cell
   x size, so every item's inclusion probability is ``n / N`` of its
   stratum.
2. **Recall windows**: 2-s zones from the frame's slots with >= 1 live track,
   a simple random sample per group (``N_WINDOWS``), uniform over the dark
   phase (not weighted by activity); ``pi = n / N`` per group.
3. **Hidden repeats**: ``N_REPEATS`` candidates again under new ids, each at
   least ``MIN_REPEAT_GAP`` items after its original.

Sampling frame: real_v1's (``experiments/real_labels/build_session.py``):
dark phase, item centre in [5 s, 10,795 s] (windows: the whole 5-s view),
>= 60 s from background-library segments, >= 5 s from the hand-labelled
regions, no box within 0.1 s (overlapping in frequency) of an earlier label.
The earlier labels now also include real_v1's 2022 items (its box items and
its window marks +-15 Hz), and every real_v1 item itself is excluded (boxes
by ``chirp_id``, windows by slot).

Writes (``$CHIRP_DATA/labels``): ``candidates/real_v2_2022.parquet`` (what
the labeller reads, in queue order), ``real_v2_2022/truth/``
(``real_v2_2022_truth.parquet``, ``pools.csv``, ``pool_by_recording.csv``,
``window_frame.csv``, ``window_frame_slots.parquet``, ``build.json``,
``build_summary.md``).

    .venv/bin/python experiments/real_labels_2022/build_2022.py
    # a trial build elsewhere (never overwrites the session):
    .venv/bin/python experiments/real_labels_2022/build_2022.py --dest /tmp/rv2_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design2022 as d

sys.path.insert(1, str(d.REAL_V1_CODE))
import build_session as v1
from common import DARK_PHASE_S, RUN_2022, VIEW, recording_dir

SEED = 20261119
N_PER_CELL = 15
MIN_PER_SIZE = 3
N_WINDOWS = {"pair3": 75, "rest": 75}
N_REPEATS = 20
MIN_REPEAT_GAP = 100
REPEAT_MARK_HALF_BAND_HZ = 15.0  # a real_v1 window mark as an earlier label: click +- this
TIME_PAUSE_S = 60.0  # real_v1 items that took longer are breaks (time estimate)
EXPECTED_MARK_SCORE = 0.8  # time estimate: detections >= this in a zone ~ marks to expect


def load_2022() -> pd.DataFrame:
    """Every stored row of the 2022 t001 run (19 recordings)."""
    manifest = json.loads((RUN_2022 / "run.json").read_text())
    frames = []
    for rec, info in sorted(manifest["recordings"].items()):
        if info.get("status") != "complete":
            msg = f"{rec}: run status {info.get('status')}"
            raise RuntimeError(msg)
        frames.append(pd.read_parquet(RUN_2022 / rec / "chirps_rtdetr.parquet",
                                      columns=["recording", *v1.MODEL_COLUMNS]))
    det = pd.concat(frames, ignore_index=True)
    det["dataset"] = "competition2022"
    return det


def real_v1_2022() -> tuple[set, set, pd.DataFrame]:
    """real_v1's 2022 items: its boxes as (recording, chirp_id), its windows as
    (recording, zone_t0_s), and the earlier-label boxes they add (every box
    item, every window mark +- REPEAT_MARK_HALF_BAND_HZ)."""
    t = pd.read_parquet(d.REAL_V1_TRUTH)
    t = t[t["dataset"] == "competition2022"]
    boxes = t[t["item_kind"] == "box"]
    wins = t[t["item_kind"] == "window"]
    box_ids = set(zip(boxes["recording"], boxes["chirp_id"].astype(int), strict=True))
    slots = set(zip(wins["recording"], wins["zone_t0_s"].astype(float), strict=True))
    cols = ["recording", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]
    parts = [boxes[cols]]
    for f in sorted((d.REAL_V1_LABELS / "labels").glob("2022-*.csv")):
        lab = pd.read_csv(f, dtype={"parent_item_id": str})
        mk = lab[lab["parent_item_id"].fillna("") != ""]
        parts.append(pd.DataFrame({
            "recording": f.stem, "t_start_s": mk["clicked_t_s"], "t_end_s": mk["clicked_t_s"],
            "f_low_hz": mk["clicked_hz"] - REPEAT_MARK_HALF_BAND_HZ,
            "f_high_hz": mk["clicked_hz"] + REPEAT_MARK_HALF_BAND_HZ}))
    return box_ids, slots, pd.concat(parts, ignore_index=True)


def candidate_pool(det: pd.DataFrame, regs: pd.DataFrame, bg: pd.DataFrame,
                   earlier: pd.DataFrame, v1_boxes: set) -> pd.DataFrame:
    """Kept rows with score >= 0.5 inside the frame, with bin, size, status, group."""
    c = det[det["score"] >= d.SCORE_EDGES[0]].reset_index(drop=True)
    c["score_bin"] = d.score_bin(c["score"])
    c["size"] = v1.size_of(c)
    c = v1.add_status(c).drop(columns=["status_stratum"])
    c["group"] = d.group_of(c["recording"])
    ok = v1.frame_mask(c, c["t_center_s"].to_numpy(), 0.0, regs, bg)
    ok &= ~np.array([(r, int(k)) in v1_boxes for r, k in zip(c["recording"], c["chirp_id"], strict=True)])
    ok[ok] &= ~v1.near_earlier(c[ok], earlier)
    c["in_frame"] = ok
    return c


def allocate(pool: pd.DataFrame) -> pd.DataFrame:
    """Items per stratum (score bin x status x group x size), see ``split_sizes``."""
    n = pool.groupby(["score_bin", "status", "group", "size"]).size()
    rows = []
    for b in d.SCORE_NAMES:
        for st in d.STATUSES:
            for g in d.GROUPS:
                ns, nb = int(n.get((b, st, g, "small"), 0)), int(n.get((b, st, g, "big"), 0))
                ks, kb = d.split_sizes(ns, nb, N_PER_CELL, MIN_PER_SIZE)
                es, eb = d.split_sizes(ns, nb, N_PER_CELL, N_PER_CELL // 2)  # even split
                cell = {"score_bin": b, "status": st, "group": g, "cell_N": ns + nb,
                        "cell_n": ks + kb, "n_eff": d.cell_n_eff(ns, nb, ks, kb),
                        "n_eff_even_split": d.cell_n_eff(ns, nb, es, eb)}
                rows.append({**cell, "size": "small", "N": ns, "n": ks})
                rows.append({**cell, "size": "big", "N": nb, "n": kb})
    return pd.DataFrame(rows)


def sample_strata(pool: pd.DataFrame, alloc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simple random sample per stratum, pooled over the recordings."""
    picked = []
    for a in alloc[alloc["n"] > 0].itertuples():
        g = pool[(pool["score_bin"] == a.score_bin) & (pool["status"] == a.status)
                 & (pool["group"] == a.group) & (pool["size"] == a.size)]
        assert len(g) == a.N
        take = g.iloc[np.sort(rng.choice(len(g), size=a.n, replace=False))].copy()
        take["cell_N"], take["cell_n"] = a.N, a.n  # the sampling stratum (cell x size)
        take["cell3_N"], take["cell3_n"] = a.cell_N, a.cell_n  # bin x status x group
        picked.append(take)
    out = pd.concat(picked, ignore_index=True)
    out["pi"] = out["cell_n"] / out["cell_N"]
    out["weight"] = 1.0 / out["pi"]
    out["stratum"] = ("main/" + out["score_bin"] + "/" + out["status"] + "/" + out["group"]
                      + "/" + out["size"])
    return out


def window_frame(regs: pd.DataFrame, bg: pd.DataFrame) -> pd.DataFrame:
    """Every 2-s slot of real_v1's window frame in the 2022 recordings."""
    from gridtools.io import load_tracks

    manifest = json.loads((RUN_2022 / "run.json").read_text())
    rows = []
    for rec, info in sorted(manifest["recordings"].items()):
        dur = float(info["duration_s"])
        starts = np.arange(v1.ACTIVE_START_S, min(dur, DARK_PHASE_S), v1.ZONE_S)
        slots = pd.DataFrame({"recording": rec, "dataset": "competition2022", "zone_t0_s": starts,
                              "zone_t1_s": starts + v1.ZONE_S})
        centre = starts + 0.5 * v1.ZONE_S
        half = 0.5 * VIEW["span_s"]  # the whole view in the frame
        slots = slots[v1.frame_mask(slots, centre, half, regs, bg) & (centre + half <= dur)].copy()
        tracks = load_tracks(v1.labeller_tracks(rec)[0])
        live = [tracks.frequencies_at(float(t), VIEW["max_gap_s"])
                for t in slots["zone_t0_s"].to_numpy() + 0.5 * v1.ZONE_S]
        slots["n_live"] = [len(v) for v in live]
        slots["f_min_hz"] = [min(v.values()) if v else np.nan for v in live]
        slots["f_max_hz"] = [max(v.values()) if v else np.nan for v in live]
        rows.append(slots[slots["n_live"] > 0])
    frame = pd.concat(rows, ignore_index=True)
    frame["group"] = d.group_of(frame["recording"])
    return frame


def same_as_real_v1(frame: pd.DataFrame) -> bool:
    """Whether the 2022 slots equal real_v1's window frame (same rule, same tracks)."""
    old = pd.read_parquet(d.REAL_V1_ROOT / "truth" / "window_frame_slots.parquet")
    old = old[old["dataset"] == "competition2022"]
    cols = ["recording", "zone_t0_s", "n_live", "f_min_hz", "f_max_hz"]
    a = frame[cols].sort_values(cols[:2]).reset_index(drop=True)
    b = old[cols].sort_values(cols[:2]).reset_index(drop=True)
    return a.shape == b.shape and bool(np.allclose(a[cols[1:]].to_numpy(float),
                                                   b[cols[1:]].to_numpy(float), equal_nan=True)
                                       and (a["recording"] == b["recording"]).all())


def sample_windows(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    picked = []
    for g, n in N_WINDOWS.items():
        f = frame[frame["group"] == g]
        take = f.iloc[np.sort(rng.choice(len(f), size=n, replace=False))].copy()
        take["cell_N"], take["cell_n"] = len(f), n
        picked.append(take)
    w = pd.concat(picked, ignore_index=True)
    w["pi"] = w["cell_n"] / w["cell_N"]
    w["weight"] = 1.0 / w["pi"]
    w["stratum"] = "window/" + w["group"]
    w["item_kind"] = "window"
    w["t_start_s"], w["t_end_s"] = w["zone_t0_s"], w["zone_t1_s"]
    w["t_center_s"] = 0.5 * (w["zone_t0_s"] + w["zone_t1_s"])
    w["f_low_hz"] = w["f_min_hz"]
    w["f_high_hz"] = w["f_max_hz"] + v1.WINDOW_HEADROOM_HZ
    return w


def new_ids(n: int, rng: np.random.Generator, used: set[str]) -> list[str]:
    out = []
    while len(out) < n:
        cid = "v" + "".join(rng.choice(list("0123456789abcdef"), size=9))
        if cid not in used:
            used.add(cid)
            out.append(cid)
    return out


def pace() -> dict:
    """The author's real_v1 times per item (``events.jsonl``)."""
    ev = pd.DataFrame([json.loads(line) for line in
                       (d.REAL_V1_LABELS / "events.jsonl").read_text().splitlines()])
    ev["parent_item_id"] = ev["parent_item_id"].fillna("")
    top = ev[(ev["parent_item_id"] == "") & (ev["type"] == "click")]
    cand = top[top["label"].isin([*d.POSITIVE, "not_chirp"])]["elapsed_s"]
    win = top[top["label"] == "window_done"].copy()
    marks = ev[(ev["parent_item_id"] != "") & (ev["type"] == "click")].groupby("parent_item_id").size()
    win["marks"] = win["item_id"].map(marks).fillna(0).clip(upper=3).astype(int)
    out = {"candidate_mean_s": float(cand.mean()),
           "candidate_mean_no_pauses_s": float(cand[cand <= TIME_PAUSE_S].mean()),
           "window_mean_s": float(win["elapsed_s"].mean()),
           "window_mean_no_pauses_s": float(win.loc[win["elapsed_s"] <= TIME_PAUSE_S, "elapsed_s"].mean()),
           "window_mean_by_marks_s": {}, "window_mean_by_marks_no_pauses_s": {}, "windows_by_marks": {}}
    for k, g in win.groupby("marks"):
        key = f"{k}+" if k == 3 else str(k)
        out["window_mean_by_marks_s"][key] = float(g["elapsed_s"].mean())
        out["window_mean_by_marks_no_pauses_s"][key] = float(
            g.loc[g["elapsed_s"] <= TIME_PAUSE_S, "elapsed_s"].mean())
        out["windows_by_marks"][key] = len(g)
    return out


def time_estimate(items: pd.DataFrame, det: pd.DataFrame, p: dict) -> dict:
    """Minutes the session should take at the author's real_v1 pace.

    Candidates at the real_v1 mean; windows either at the real_v1 mean (flat)
    or at the mean of real_v1 windows with as many marks as this window has
    kept detections >= EXPECTED_MARK_SCORE in its zone (a proxy; only used
    here)."""
    n_box = int((items["item_kind"] == "box").sum())
    w = items[items["item_kind"] == "window"]
    hi = det[det["kept"] & (det["score"] >= EXPECTED_MARK_SCORE)]
    expected = []
    for r in w.itertuples():
        dd = hi[hi["recording"] == r.recording]
        expected.append(int(((dd["t_center_s"] >= r.zone_t0_s) & (dd["t_center_s"] <= r.zone_t1_s)).sum()))
    keys = [f"{min(k, 3)}+" if k >= 3 else str(k) for k in expected]
    out = {"candidates_incl_repeats": n_box, "windows": len(w),
           "windows_by_expected_marks": pd.Series(keys).value_counts().sort_index().to_dict()}
    for tag, cm, wm, wk in (("with_pauses", p["candidate_mean_s"], p["window_mean_s"],
                             p["window_mean_by_marks_s"]),
                            ("without_pauses", p["candidate_mean_no_pauses_s"],
                             p["window_mean_no_pauses_s"], p["window_mean_by_marks_no_pauses_s"])):
        out[f"minutes_flat_{tag}"] = (n_box * cm + len(w) * wm) / 60.0
        out[f"minutes_by_marks_{tag}"] = (n_box * cm + sum(wk[k] for k in keys)) / 60.0
    return out


def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def summary_md(alloc: pd.DataFrame, wins: pd.DataFrame, frame: pd.DataFrame, te: dict,
               p: dict, info: dict) -> str:
    cells = alloc.groupby(["score_bin", "status", "group"], sort=False).agg(
        N=("N", "sum"), n=("n", "sum"), N_big=("N", "last"), n_big=("n", "last"),
        n_eff=("n_eff", "first"), n_eff_even_split=("n_eff_even_split", "first")).reset_index()
    wide = cells.pivot_table(index=["score_bin", "status"], columns="group", values=["n", "N"],
                             aggfunc="sum", sort=False)
    wide.columns = [f"{g} {v}" for v, g in wide.columns]
    wide = wide[["pair3 n", "pair3 N", "rest n", "rest N"]].reset_index()
    wtab = frame.groupby("group").size().rename("frame_slots").to_frame().join(
        wins.groupby("group").size().rename("windows")).reset_index()
    lines = [
        "# real_v2_2022: build summary (written by build_2022.py)", "",
        f"Created {info['created']}; seed {SEED}; {info['n_items']} items "
        f"({info['per_kind']['box']} candidates incl. {N_REPEATS} hidden repeats, "
        f"{info['per_kind']['window']} windows).", "",
        "## Candidates per cell (n sampled / N in the frame's pool), kept, score >= 0.5", "",
        _md(wide), "",
        "## Per cell: size split and effective sample size", "",
        "n_eff: Kish's effective sample size of the cell's estimate under this design; "
        "n_eff_even_split: the same if the cell were split evenly over small and big "
        "chirps (real_v1's rule).", "",
        _md(cells), "",
        "## Recall windows (uniform over the dark phase, slots with >= 1 live track)", "",
        _md(wtab), "",
        "## Time estimate (the author's real_v1 pace)", "",
        f"real_v1 means: {p['candidate_mean_s']:.2f} s per candidate, {p['window_mean_s']:.2f} s per "
        f"window (without items > {TIME_PAUSE_S:.0f} s: {p['candidate_mean_no_pauses_s']:.2f} s, "
        f"{p['window_mean_no_pauses_s']:.2f} s). Windows by number of marks (with pauses): "
        + ", ".join(f"{k}: {v:.2f} s (n={p['windows_by_marks'][k]})"
                    for k, v in p["window_mean_by_marks_s"].items()) + ".", "",
        f"This session: {te['candidates_incl_repeats']} candidates, {te['windows']} windows; "
        f"windows by kept detections >= {EXPECTED_MARK_SCORE} in the zone: "
        f"{te['windows_by_expected_marks']}.", "",
        f"- flat real_v1 means, with pauses: {te['minutes_flat_with_pauses']:.1f} min",
        f"- windows by expected marks, with pauses: {te['minutes_by_marks_with_pauses']:.1f} min",
        f"- windows by expected marks, without pauses: {te['minutes_by_marks_without_pauses']:.1f} min",
        "",
        f"Window frame identical to real_v1's 2022 slots: {info['window_frame_same_as_real_v1']}.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path, help="write candidates and truth here instead (trial build)")
    args = ap.parse_args()
    candidates_path, root = d.CANDIDATES, d.ROOT
    if args.dest is not None:
        candidates_path = args.dest / "candidates" / f"{d.SESSION}.parquet"
        root = args.dest / d.SESSION
    elif d.LABELS_OUT.exists() and any(d.LABELS_OUT.iterdir()):
        raise SystemExit(f"{d.LABELS_OUT} exists: the session has started; never rebuild it")
    rng = np.random.default_rng(SEED)
    det = load_2022()
    regs, bg = v1.regions(), v1.background_segments()
    v1_boxes, v1_slots, v1_earlier = real_v1_2022()
    earlier = pd.concat([v1.earlier_boxes(), v1_earlier], ignore_index=True)
    cand = candidate_pool(det, regs, bg, earlier, v1_boxes)
    pool = cand[cand["in_frame"] & cand["kept"]].reset_index(drop=True)
    print(f"{len(det)} stored rows; {len(cand)} with score >= 0.5; {len(pool)} kept in the frame")
    alloc = allocate(pool)
    boxes = sample_strata(pool, alloc, rng)
    boxes["item_kind"] = "box"

    frame = window_frame(regs, bg)
    same = same_as_real_v1(frame)
    print(f"window frame: {len(frame)} slots, identical to real_v1's: {same}")
    taken = np.array([(r, float(t)) in v1_slots for r, t in zip(frame["recording"], frame["zone_t0_s"],
                                                                 strict=True)])
    frame = frame[~taken].reset_index(drop=True)
    wins = sample_windows(frame, rng)

    items = pd.concat([boxes, wins], ignore_index=True)
    used: set[str] = set()
    items["candidate_id"] = new_ids(len(items), rng, used)
    items["repeat_of"] = ""
    rep = rng.choice(np.flatnonzero(items["item_kind"] == "box"), N_REPEATS, replace=False)
    reps = items.iloc[rep].copy()
    reps["repeat_of"] = reps["candidate_id"]
    reps["candidate_id"] = new_ids(len(reps), rng, used)
    reps["stratum"] = "repeat/" + reps["stratum"]
    n_base = len(items)
    items = pd.concat([items, reps], ignore_index=True)
    v1.MIN_REPEAT_GAP = MIN_REPEAT_GAP  # order_items reads its module's constant
    order = v1.order_items(n_base, [(n_base + k, int(o)) for k, o in enumerate(rep)], rng)
    items = items.iloc[order].reset_index(drop=True)
    items["queue_position"] = np.arange(len(items))
    trk = [v1.labeller_tracks(r) for r in items["recording"]]
    items["tracks"] = [str(p) for p, _ in trk]
    items["tracks_source"] = [s for _, s in trk]
    items["recording_dir"] = [str(recording_dir("competition2022", r)) for r in items["recording"]]

    # what the labeller reads: no score, no model emitter, no stratum
    visible = pd.DataFrame({
        "recording": items["recording_dir"], "t_start_s": items["t_start_s"].clip(lower=0.0),
        "t_end_s": items["t_end_s"], "f_low_hz": items["f_low_hz"], "f_high_hz": items["f_high_hz"],
        "candidate_id": items["candidate_id"], "tracks": items["tracks"],
        "item_kind": items["item_kind"]})
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    visible.to_parquet(candidates_path, index=False)

    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    keep = ["queue_position", "candidate_id", "item_kind", "stratum", "repeat_of", "dataset", "group",
            "recording", "recording_dir", "tracks", "tracks_source", "score_bin", "size",
            "cell_N", "cell_n", "cell3_N", "cell3_n", "pi", "weight", "zone_t0_s", "zone_t1_s",
            "n_live", "f_min_hz", "f_max_hz", "status", "status_detail", "status_track_id",
            "status_assignment", *v1.MODEL_COLUMNS]
    truth = items[[c for c in keep if c in items.columns]].copy()
    truth_path = truth_dir / f"{d.SESSION}_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    alloc.to_csv(truth_dir / "pools.csv", index=False)
    pool.groupby(["score_bin", "status", "group", "size", "recording"]).size().rename(
        "N").reset_index().to_csv(truth_dir / "pool_by_recording.csv", index=False)
    frame.groupby(["group", "recording"]).size().rename("slots").reset_index().to_csv(
        truth_dir / "window_frame.csv", index=False)
    frame.to_parquet(truth_dir / "window_frame_slots.parquet", index=False)

    p = pace()
    te = time_estimate(items, det, p)
    info = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "run": str(RUN_2022),
        "parameters": {"N_PER_CELL": N_PER_CELL, "MIN_PER_SIZE": MIN_PER_SIZE, "N_WINDOWS": N_WINDOWS,
                       "N_REPEATS": N_REPEATS, "MIN_REPEAT_GAP": MIN_REPEAT_GAP,
                       "SCORE_NAMES": d.SCORE_NAMES, "PAIR3": d.PAIR3, "SIZE_CUT_HZ": v1.SIZE_CUT_HZ,
                       "frame_rules": "real_v1 build_session.frame_mask / near_earlier",
                       "REPEAT_MARK_HALF_BAND_HZ": REPEAT_MARK_HALF_BAND_HZ},
        "view": VIEW,
        "labeller_args": "label-emitters --blind --span-s 5 --span-hz 600 --keep-order --serve",
        "n_items": len(truth),
        "per_kind": truth.groupby("item_kind").size().to_dict(),
        "per_stratum": truth.groupby("stratum").size().to_dict(),
        "candidates_file": str(candidates_path),
        "candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "frame": {"stored_rows": len(det), "score_ge_0.5": len(cand),
                  "score_ge_0.5_in_frame": int(cand["in_frame"].sum()),
                  "kept_in_frame": len(pool),
                  "flagged_in_frame": int((cand["in_frame"] & ~cand["kept"]).sum()),
                  "real_v1_boxes_excluded": len(v1_boxes),
                  "real_v1_window_slots_excluded": int(taken.sum()),
                  "window_slots": frame.groupby("group").size().to_dict()},
        "window_frame_same_as_real_v1": same,
        "pace_real_v1": p,
        "time_estimate": te,
    }
    (truth_dir / "build.json").write_text(json.dumps(info, indent=1, default=str))
    (truth_dir / "build_summary.md").write_text(summary_md(alloc, wins, frame, te, p, info))
    print((truth_dir / "build_summary.md").read_text())


if __name__ == "__main__":
    main()
