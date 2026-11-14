"""Build the real-data labelling session real_v1 (design approved 2026-11-14).

Items (blind browser labeller, the simulated benchmark's view):

1. Candidates (detections of the t001 runs, store threshold 0.01): kept rows
   (not flagged by post-processing), stratified by dataset (2019/2020,
   2022) x score bin x chirp size, ``N_PER_CELL`` per stratum (shortfalls of
   a stratum go to the other strata of the same score bin). Within a
   stratum the sample is spread over the recordings by water-filling (every
   recording gets the same number, up to what it has; dense recordings are
   capped at that level), and drawn at random within each recording. The
   sampling cells are therefore stratum x recording, and every item's
   inclusion probability is exact: ``pi = n_cell / N_cell``.
2. Flagged rows (post-processing: harmonic copies, duplicates) at score >=
   0.5, ``N_FLAGGED`` in all, by reason x dataset, cells as above.
3. Recall windows: 2-s zones (``ZONE_S``) on a grid of non-overlapping
   slots, drawn per dataset with probability proportional to
   ``c_m + WINDOW_UNIFORM * mean(c)``, where ``c_m`` is the number of chirps
   (kept, score >= 0.5) of the slot's recording minute (a defensive
   mixture: 80 % of the probability mass proportional to the minute's
   activity, 20 % uniform, so that quiet minutes, where missed chirps may
   hide, keep a chance). Chirping is sparse (5 % of the 2-s slots of 2022
   hold a detection), so a uniform draw would give few chirps; the
   weighting uses minute counts only, never the detections inside a slot. ``n`` draws with
   replacement per dataset (repeated draws dropped), so ``pi = 1 - (1 -
   p)^n``, recorded with ``p``.
4. Hidden repeats (test-retest): ``N_REPEAT_CANDIDATES`` candidates and
   ``N_REPEAT_WINDOWS`` windows again under new ids, each at least
   ``MIN_REPEAT_GAP`` items after its original.

Sampling frame (both datasets): the active (dark) phase, item centre in
``[ACTIVE_START_S, DARK_PHASE_S - ACTIVE_END_PAD_S]`` (windows: the whole
5-s view inside it), at least ``BACKGROUND_MARGIN_S`` from every
background-library segment (training data), at least ``REGION_MARGIN_S``
from the 6 hand-labelled regions, and no box within 0.1 s (overlapping in
frequency) of a box of an earlier real label set.

Writes (``$CHIRP_DATA``):

- ``labels/candidates/real_v1.parquet``: what the labeller reads (recording,
  box, candidate_id, tracks, item_kind); in queue order;
- ``labels/real_v1/truth/real_v1_truth.parquet``: per item the stratum,
  cell sizes, inclusion probability, weight, repeat link, and the model's
  outputs (never read by the labeller);
- ``labels/real_v1/truth/pools.csv``, ``window_frame.csv``: frame sizes;
- ``labels/real_v1/truth/build.json``: parameters, inputs, counts.

    uv run python experiments/real_labels/build_session.py
    # a trial build on the recordings finished so far, written elsewhere:
    uv run python experiments/real_labels/build_session.py --partial --dest /tmp/real_v1_test
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
from common import (  # noqa: E402
    BACKGROUND,
    CANDIDATES,
    DARK_PHASE_S,
    EMITTER_LABELS,
    GAPFILL_2019,
    PRODUCTION_2019,
    REAL_LABELS,
    ROOT,
    RUN_2019,
    RUN_2022,
    TRACKS_GAPFILL,
    VIEW,
    dataset_of,
    recording_dir,
    tracks_dir,
)

SEED = 20261114
SCORE_BINS = (0.01, 0.05, 0.2, 0.5, 0.7, 0.9, 1.0001)
SCORE_NAMES = ("0.01-0.05", "0.05-0.2", "0.2-0.5", "0.5-0.7", "0.7-0.9", "0.9-1")
SIZE_CUT_HZ = 200.0  # decoder height; valley of the bimodal height distribution
SIZE_CUT_BOX_HZ = 250.0  # rows without a decoder height: box top above the emitter
N_PER_CELL = 31  # 6 score bins x 2 sizes x 2 datasets = 24 strata -> ~744
N_FLAGGED = 50
FLAGGED_MIN_SCORE = 0.5
N_WINDOWS = {"competition2019": 125, "competition2022": 75}
ZONE_S = 2.0
WINDOW_UNIFORM = 0.25  # weight of the uniform part: c_m + 0.25 mean(c)
N_REPEAT_CANDIDATES = 40
N_REPEAT_WINDOWS = 10
MIN_REPEAT_GAP = 150
ACTIVE_START_S = 5.0
ACTIVE_END_PAD_S = 5.0
BACKGROUND_MARGIN_S = 60.0
REGION_MARGIN_S = 5.0
WINDOW_HEADROOM_HZ = 400.0  # window view: up to this far above the highest baseline
MODEL_COLUMNS = (
    "chirp_id", "t_start_s", "t_end_s", "t_center_s", "f_low_hz", "f_high_hz",
    "score", "operating", "kept", "chirp", "emitter_hz", "track_id", "track_hz",
    "emitter_track_df_hz", "assign_status", "assign_confidence", "nearest_track_id",
    "nearest_track_hz", "nearest_df_hz", "second_track_id", "second_track_hz",
    "second_df_hz", "assign_margin_hz", "n_live_tracks", "track_df_other_hz",
    "height_hz", "fwhm_s", "height_raw_hz", "params_ok", "params_flag",
    "pp_drop_reason", "pp_harmonic_of_track", "pp_duplicate_of", "pp_reattributed",
    "click_fraction_box",
)


PARTIAL = False  # --partial: only recordings that are complete (trial builds)


def _runs() -> dict[str, Path]:
    return {"competition2019": RUN_2019, "competition2022": RUN_2022}


def _complete(info: dict) -> bool:
    if info.get("status") == "complete":
        return True
    if PARTIAL:
        return False
    msg = f"recording status {info.get('status')} (use --partial for a trial build)"
    raise RuntimeError(msg)


def load_detections() -> pd.DataFrame:
    """Every stored row of both t001 runs, with dataset and run."""
    frames = []
    for ds, run in _runs().items():
        manifest = json.loads((run / "run.json").read_text())
        for rec, info in sorted(manifest["recordings"].items()):
            if not _complete(info):
                continue
            df = pd.read_parquet(
                run / rec / "chirps_rtdetr.parquet", columns=["recording", *MODEL_COLUMNS]
            )
            df["dataset"] = ds
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def regions() -> pd.DataFrame:
    rows = []
    for f in sorted(REAL_LABELS.glob("*/*/traces-grid1-editable-labels.csv")):
        lab = pd.read_csv(f)
        for r in lab[lab["category"] == "labeled_region"].itertuples():
            rows.append({"recording": f.parent.name, "t0": r.t_start_s, "t1": r.t_end_s})
    return pd.DataFrame(rows)


def earlier_boxes() -> pd.DataFrame:
    """Boxes of earlier real label sets (hand labels + the labeller's 58)."""
    rows = []
    for f in sorted(REAL_LABELS.glob("*/*/traces-grid1-editable-labels.csv")):
        lab = pd.read_csv(f)
        lab = lab[lab["category"] != "labeled_region"]
        rows.append(lab.assign(recording=f.parent.name))
    for f in sorted(EMITTER_LABELS.glob("*.csv")):
        rows.append(pd.read_csv(f).assign(recording=f.stem))
    cols = ["recording", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]
    return pd.concat([r[cols] for r in rows], ignore_index=True)


def background_segments() -> pd.DataFrame:
    m = pd.read_parquet(BACKGROUND / "manifest.parquet")
    m = m[m["dataset"].str.startswith("competition")]
    return m[["recording", "start_s", "stop_s"]]


def frame_mask(
    df: pd.DataFrame, t: np.ndarray, half_s: float, regs: pd.DataFrame, bg: pd.DataFrame
) -> np.ndarray:
    """Items (centre t, half-width half_s) inside the sampling frame."""
    ok = (t - half_s >= ACTIVE_START_S) & (t + half_s <= DARK_PHASE_S - ACTIVE_END_PAD_S)
    rec = df["recording"].to_numpy()
    for r in regs.itertuples():
        ok &= ~((rec == r.recording) & (t + half_s >= r.t0 - REGION_MARGIN_S)
                & (t - half_s <= r.t1 + REGION_MARGIN_S))
    for name, g in bg.groupby("recording"):
        sel = rec == name
        if not sel.any():
            continue
        for s in g.itertuples():
            ok[sel] &= ~((t[sel] + half_s >= s.start_s - BACKGROUND_MARGIN_S)
                         & (t[sel] - half_s <= s.stop_s + BACKGROUND_MARGIN_S))
    return ok


def near_earlier(df: pd.DataFrame, earlier: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=bool)
    tc = df["t_center_s"].to_numpy()
    for name, g in earlier.groupby("recording"):
        sel = np.flatnonzero(df["recording"].to_numpy() == name)
        if not sel.size:
            continue
        gc = 0.5 * (g["t_start_s"].to_numpy() + g["t_end_s"].to_numpy())
        for k in sel:
            close = np.abs(gc - tc[k]) <= 0.1
            close &= (g["f_low_hz"].to_numpy() < df["f_high_hz"].iat[k]) & (
                df["f_low_hz"].iat[k] < g["f_high_hz"].to_numpy())
            out[k] = close.any()
    return out


def size_of(df: pd.DataFrame) -> np.ndarray:
    h = df["height_hz"].to_numpy(float)
    box = (df["f_high_hz"] - df["emitter_hz"]).to_numpy(float)
    big = np.where(np.isfinite(h), h >= SIZE_CUT_HZ, box >= SIZE_CUT_BOX_HZ)
    return np.where(big, "big", "small")


def waterfill(counts: dict[str, int], n: int) -> dict[str, int]:
    """Spread n over cells as evenly as possible, at most counts[c] each."""
    alloc = dict.fromkeys(counts, 0)
    left = min(n, sum(counts.values()))
    open_cells = sorted(c for c in counts if counts[c] > 0)
    while left > 0 and open_cells:
        share, extra = divmod(left, len(open_cells))
        for k, c in enumerate(open_cells):
            give = min(counts[c] - alloc[c], share + (k < extra))
            alloc[c] += give
            left -= give
        open_cells = [c for c in open_cells if alloc[c] < counts[c]]
    return alloc


def allocate_strata(pool: pd.DataFrame) -> dict[tuple, int]:
    """n per stratum (dataset, score bin, size): N_PER_CELL, shortfalls of a
    stratum moved to the other strata of its score bin."""
    sizes = pool.groupby(["dataset", "score_bin", "size"]).size().to_dict()
    out: dict[tuple, int] = {}
    for b in SCORE_NAMES:
        cells = {k: v for k, v in sizes.items() if k[1] == b}
        target = N_PER_CELL * 4
        out |= waterfill(cells, target)
    return out


def sample_cells(pool: pd.DataFrame, strata: dict[tuple, int], keys: list[str],
                 rng: np.random.Generator) -> pd.DataFrame:
    """Within each stratum, water-fill over recordings, then SRS per cell."""
    picked = []
    for key, n in strata.items():
        if n == 0:
            continue
        sel = np.ones(len(pool), dtype=bool)
        for col, v in zip(keys, key, strict=True):
            sel &= pool[col].to_numpy() == v
        g = pool[sel]
        counts = g.groupby("recording").size().to_dict()
        alloc = waterfill(counts, n)
        for rec, k in alloc.items():
            if k == 0:
                continue
            cell = g[g["recording"] == rec]
            take = cell.iloc[np.sort(rng.choice(len(cell), size=k, replace=False))].copy()
            take["cell_N"] = len(cell)
            take["cell_n"] = k
            take["stratum_N"] = len(g)
            take["stratum_n"] = n
            picked.append(take)
    out = pd.concat(picked, ignore_index=True)
    out["pi"] = out["cell_n"] / out["cell_N"]
    out["weight"] = 1.0 / out["pi"]
    return out


def labeller_tracks(rec: str) -> tuple[Path, str]:
    gap = TRACKS_GAPFILL / rec
    if (gap / "ident_v.npy").exists():
        return gap, "gapfill"
    return tracks_dir(dataset_of(rec), rec), "original"


def window_frame(det: pd.DataFrame, regs: pd.DataFrame, bg: pd.DataFrame) -> pd.DataFrame:
    """Every 2-s slot of the frame with its minute's chirp count."""
    from gridtools.io import load_tracks

    rows = []
    for ds, run in _runs().items():
        manifest = json.loads((run / "run.json").read_text())
        for rec, info in sorted(manifest["recordings"].items()):
            if not _complete(info):
                continue
            dur = float(info["duration_s"])
            starts = np.arange(ACTIVE_START_S, min(dur, DARK_PHASE_S), ZONE_S)
            slots = pd.DataFrame({"recording": rec, "dataset": ds, "zone_t0_s": starts,
                                  "zone_t1_s": starts + ZONE_S})
            centre = slots["zone_t0_s"].to_numpy() + 0.5 * ZONE_S
            half = 0.5 * VIEW["span_s"]  # the whole view in the frame
            keep = frame_mask(slots, centre, half, regs, bg) & (centre + half <= dur)
            slots = slots[keep].copy()
            if slots.empty:
                continue
            trk_dir, _ = labeller_tracks(rec)
            tracks = load_tracks(trk_dir)
            centre = slots["zone_t0_s"].to_numpy() + 0.5 * ZONE_S
            live = [tracks.frequencies_at(float(t), VIEW["max_gap_s"]) for t in centre]
            slots["n_live"] = [len(v) for v in live]
            slots["f_min_hz"] = [min(v.values()) if v else np.nan for v in live]
            slots["f_max_hz"] = [max(v.values()) if v else np.nan for v in live]
            slots = slots[slots["n_live"] > 0]
            d = det[(det["recording"] == rec) & det["kept"] & (det["score"] >= 0.5)]
            minute = np.floor(d["t_center_s"].to_numpy() / 60.0).astype(int)
            per_min = pd.Series(minute).value_counts()
            slot_min = np.floor((slots["zone_t0_s"].to_numpy() + 0.5 * ZONE_S) / 60.0).astype(int)
            slots["minute"] = slot_min
            slots["minute_chirps"] = per_min.reindex(slot_min).fillna(0).to_numpy(int)
            rows.append(slots)
    return pd.concat(rows, ignore_index=True)


def sample_windows(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    picked = []
    for ds, n in N_WINDOWS.items():
        f = frame[frame["dataset"] == ds].reset_index(drop=True)
        c = f["minute_chirps"].to_numpy(float)
        size = c + WINDOW_UNIFORM * c.mean()
        p = size / size.sum()
        draws = rng.choice(len(f), size=n, replace=True, p=p)
        uniq = np.unique(draws)
        w = f.iloc[uniq].copy()
        w["p_draw"] = p[uniq]
        w["n_draws"] = n
        w["pi"] = 1.0 - (1.0 - p[uniq]) ** n
        w["weight"] = 1.0 / w["pi"]
        w["frame_slots"] = len(f)
        w["frame_mean_minute_chirps"] = c.mean()
        picked.append(w)
        print(f"windows {ds}: {len(uniq)} distinct of {n} draws, frame {len(f)} slots")
    return pd.concat(picked, ignore_index=True)


def _id(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        cid = "r" + "".join(rng.choice(list("0123456789abcdef"), size=9))
        if cid not in used:
            used.add(cid)
            return cid


def order_items(n_base: int, repeats: list[tuple[int, int]], rng: np.random.Generator) -> list[int]:
    """Random order of the base items, then each repeat inserted at random at
    least MIN_REPEAT_GAP after its original (originals too close to the end
    are first swapped with a random earlier item)."""
    order = [int(i) for i in rng.permutation(n_base)]
    originals = {o for _, o in repeats}
    limit = n_base - MIN_REPEAT_GAP
    for _, orig in repeats:
        p = order.index(orig)
        if p >= limit:
            free = [k for k in range(limit) if order[k] not in originals]
            k = int(rng.choice(free))
            order[p], order[k] = order[k], order[p]
    for j in rng.permutation(len(repeats)):
        rep, orig = repeats[int(j)]
        p = order.index(orig)
        order.insert(int(rng.integers(p + MIN_REPEAT_GAP, len(order) + 1)), rep)
    pos = {item: k for k, item in enumerate(order)}
    assert all(pos[r] - pos[o] >= MIN_REPEAT_GAP for r, o in repeats)
    assert sorted(order) == list(range(n_base + len(repeats)))
    return order


def production_match(sample: pd.DataFrame) -> pd.DataFrame:
    """Production (store 0.05) and gap-filled rows of the same boxes (2019)."""
    out = []
    for rec, g in sample[sample["dataset"] == "competition2019"].groupby("recording"):
        prod = pd.read_parquet(PRODUCTION_2019 / rec / "chirps_rtdetr.parquet")
        gpath = GAPFILL_2019 / rec / "chirps_rtdetr_gapfill.parquet"
        gap = pd.read_parquet(gpath) if gpath.exists() else None
        key = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"]
        m = g[["candidate_id", *key]].merge(
            prod[[*key, "chirp_id", "track_id", "assign_status", "pp_drop_reason"]].rename(
                columns={"chirp_id": "prod_chirp_id", "track_id": "prod_track_id",
                         "assign_status": "prod_assign_status",
                         "pp_drop_reason": "prod_pp_drop_reason"}),
            on=key, how="left")
        if gap is not None:
            m = m.merge(
                gap[[*key, "track_id", "assign_status", "pp_drop_reason"]].rename(
                    columns={"track_id": "gapfill_track_id",
                             "assign_status": "gapfill_assign_status",
                             "pp_drop_reason": "gapfill_pp_drop_reason"}),
                on=key, how="left")
        out.append(m.drop(columns=key))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["candidate_id"])


def main() -> None:
    global PARTIAL, CANDIDATES, ROOT  # noqa: PLW0603
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--partial", action="store_true", help="only complete recordings")
    ap.add_argument("--dest", type=Path, help="write candidates and truth here instead")
    args = ap.parse_args()
    PARTIAL = args.partial
    if args.dest is not None:
        CANDIDATES = args.dest / "candidates" / "real_v1.parquet"
        ROOT = args.dest / "real_v1"
    elif PARTIAL:
        raise SystemExit("--partial needs --dest (never overwrite the session)")
    rng = np.random.default_rng(SEED)
    det = load_detections()
    det["score_bin"] = pd.cut(det["score"], SCORE_BINS, right=False, labels=SCORE_NAMES).astype(str)
    det["size"] = size_of(det)
    regs, bg, earlier = regions(), background_segments(), earlier_boxes()
    tc = det["t_center_s"].to_numpy()
    in_frame = frame_mask(det, tc, 0.0, regs, bg)
    in_frame[in_frame] &= ~near_earlier(det[in_frame], earlier)
    det["in_frame"] = in_frame
    print(f"{len(det)} stored rows, {int(in_frame.sum())} in the frame")

    pool = det[det["in_frame"] & det["kept"]].reset_index(drop=True)
    strata = allocate_strata(pool)
    main_s = sample_cells(pool, strata, ["dataset", "score_bin", "size"], rng)
    main_s["stratum"] = ("main/" + main_s["dataset"].str.removeprefix("competition") + "/"
                         + main_s["score_bin"] + "/" + main_s["size"])
    flagged_pool = det[det["in_frame"] & ~det["kept"] & (det["score"] >= FLAGGED_MIN_SCORE)]
    flagged_pool = flagged_pool.reset_index(drop=True)
    fl_sizes = flagged_pool.groupby(["dataset", "pp_drop_reason"]).size().to_dict()
    fl_alloc = waterfill(fl_sizes, N_FLAGGED)
    flag_s = sample_cells(flagged_pool, fl_alloc, ["dataset", "pp_drop_reason"], rng)
    flag_s["stratum"] = ("flagged/" + flag_s["dataset"].str.removeprefix("competition") + "/"
                         + flag_s["pp_drop_reason"])
    cands = pd.concat([main_s, flag_s], ignore_index=True)
    cands["item_kind"] = "box"

    frame = window_frame(det, regs, bg)
    wins = sample_windows(frame, rng)
    wins["stratum"] = "window/" + wins["dataset"].str.removeprefix("competition")
    wins["item_kind"] = "window"
    wins["t_start_s"] = wins["zone_t0_s"]
    wins["t_end_s"] = wins["zone_t1_s"]
    wins["t_center_s"] = 0.5 * (wins["zone_t0_s"] + wins["zone_t1_s"])
    wins["f_low_hz"] = wins["f_min_hz"]
    wins["f_high_hz"] = wins["f_max_hz"] + WINDOW_HEADROOM_HZ

    items = pd.concat([cands, wins], ignore_index=True)
    used: set[str] = set()
    items["candidate_id"] = [_id(rng, used) for _ in range(len(items))]
    items["repeat_of"] = ""
    rep_c = rng.choice(np.flatnonzero(items["item_kind"] == "box"), N_REPEAT_CANDIDATES, replace=False)
    rep_w = rng.choice(np.flatnonzero(items["item_kind"] == "window"), N_REPEAT_WINDOWS, replace=False)
    reps = items.iloc[np.concatenate([rep_c, rep_w])].copy()
    reps["repeat_of"] = reps["candidate_id"]
    reps["candidate_id"] = [_id(rng, used) for _ in range(len(reps))]
    reps["stratum"] = "repeat/" + reps["stratum"]
    n_base = len(items)
    pairs = [(n_base + k, int(o)) for k, o in enumerate(np.concatenate([rep_c, rep_w]))]
    items = pd.concat([items, reps], ignore_index=True)
    order = order_items(n_base, pairs, rng)
    items = items.iloc[order].reset_index(drop=True)
    items["queue_position"] = np.arange(len(items))

    trk = [labeller_tracks(r) for r in items["recording"]]
    items["tracks"] = [str(p) for p, _ in trk]
    items["tracks_source"] = [s for _, s in trk]
    items["recording_dir"] = [str(recording_dir(ds, r)) for ds, r in
                              zip(items["dataset"], items["recording"], strict=True)]

    # what the labeller reads: no score, no model emitter, no stratum
    visible = pd.DataFrame({
        "recording": items["recording_dir"],
        "t_start_s": items["t_start_s"].clip(lower=0.0),
        "t_end_s": items["t_end_s"],
        "f_low_hz": items["f_low_hz"],
        "f_high_hz": items["f_high_hz"],
        "candidate_id": items["candidate_id"],
        "tracks": items["tracks"],
        "item_kind": items["item_kind"],
    })
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    visible.to_parquet(CANDIDATES, index=False)

    truth_dir = ROOT / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    keep = ["queue_position", "candidate_id", "item_kind", "stratum", "repeat_of", "dataset",
            "recording", "recording_dir", "tracks", "tracks_source", "score_bin", "size",
            "cell_N", "cell_n", "stratum_N", "stratum_n", "pi", "weight",
            "zone_t0_s", "zone_t1_s", "minute", "minute_chirps", "p_draw", "n_draws",
            "frame_slots", "frame_mean_minute_chirps", "n_live", "f_min_hz", "f_max_hz",
            *MODEL_COLUMNS]
    truth = items[[c for c in keep if c in items.columns]].copy()
    truth = truth.merge(production_match(items[items["item_kind"] == "box"]),
                        on="candidate_id", how="left")
    truth.to_parquet(truth_dir / "real_v1_truth.parquet", index=False)

    pools = det[det["in_frame"]].groupby(["dataset", "kept", "score_bin", "size"]).size()
    pools.rename("N").reset_index().to_csv(truth_dir / "pools.csv", index=False)
    cells = pool.groupby(["dataset", "score_bin", "size", "recording"]).size()
    cells.rename("N").reset_index().to_csv(truth_dir / "pool_cells.csv", index=False)
    flagged_pool.groupby(["dataset", "pp_drop_reason", "recording"]).size().rename(
        "N").reset_index().to_csv(truth_dir / "flagged_cells.csv", index=False)
    frame.groupby(["dataset", "recording"]).agg(
        slots=("zone_t0_s", "size"), chirps_in_minutes=("minute_chirps", "sum")
    ).reset_index().to_csv(truth_dir / "window_frame.csv", index=False)
    frame.to_parquet(truth_dir / "window_frame_slots.parquet", index=False)

    counts = truth.groupby("stratum").size().to_dict()
    build = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "runs": {k: str(v) for k, v in _runs().items()},
        "parameters": {k: v for k, v in globals().items() if k.isupper() and isinstance(
            v, (int, float, str, tuple, dict)) and k not in ("MODEL_COLUMNS",)},
        "view": VIEW,
        "n_items": len(truth),
        "per_stratum": counts,
        "candidates_file": str(CANDIDATES),
        "candidates_sha256": hashlib.sha256(CANDIDATES.read_bytes()).hexdigest(),
        "frame": {
            "stored_rows": int(len(det)),
            "in_frame": int(det["in_frame"].sum()),
            "kept_in_frame": int(len(pool)),
            "flagged_in_frame_score_ge_0.5": int(len(flagged_pool)),
            "window_slots": {k: int(v) for k, v in frame.groupby("dataset").size().items()},
        },
    }
    (truth_dir / "build.json").write_text(json.dumps(build, indent=1, default=str))
    print(json.dumps({"n_items": len(truth), "per_kind": truth.groupby("item_kind").size().to_dict()}))
    print(pd.Series(counts).to_string())


if __name__ == "__main__":
    main()
