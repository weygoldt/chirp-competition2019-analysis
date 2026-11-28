"""Real-data agreement of the single-window detector, reusing the real_v1 verdicts.

Production switched to the single-window detector on 2026-11-27
(``experiments/production_v3``; author's decision, labels: "match + short
session"). The expert's session real_v1 (``labels/real_v1``, README there) is
a stratified probability sample of the v2 detector's rows (the "old frame":
kept rows of ``competition2019_v2_t001`` in the session's sampling frame) plus
2-s recall windows labelled exhaustively. For the new detector (run
``competition2019_v3``, assignment of ``competition2019_v3_gapfill``), only on
the 2019/2020 recordings (the paper's; the new run has no 2022 recordings):

- **Recall** (windows): the marks do not depend on the detector; they are
  matched to the new run's kept detections with the session's rule
  (``score_session.match_marks``), window weights unchanged.
- **Precision**: the new frame (the new run's kept rows in the session's
  sampling frame) is split by a rule known for every row. ``P``: new rows
  matched one-to-one (time-frequency IoU >= ``MATCH_IOU``, greedy by IoU,
  against every stored row of the old run) to an old row of the old frame's
  main pool with an old score >= ``P_MIN_OLD_SCORE``. ``N``: every other new
  row. Over ``P`` the old sample is a probability sample: the old items whose
  row lies in ``P`` keep their weights (domain estimation), and the expert's
  verdict on the old box stands for the matched new box (the same chirp,
  IoU >= 0.5). ``N`` gets a fresh sample with real_v1's design (new score
  bin x size, in the bins >= 0.5 also the assigned fish's trial status;
  water-filled over recordings, ``pi = n_cell / N_cell``) labelled in a short
  session (``real_sw``, the same blind labeller and view). The two parts are
  disjoint, so their Horvitz-Thompson totals add up: ``P(t) = (sum_P w y +
  sum_N w y) / (sum_P w + sum_N w)`` over the items with new score >= t.
  Bootstrap: the old and the new strata resampled independently.
- **Emitter agreement**: the expert's emitter vs the new assignment
  (production and gap-filled), on the chirp items of both parts and on the
  window marks matched at 0.5.
- The labeller's test-retest is real_v1's; ``FILLERS`` old items of ``P``
  are shown again in the new session (blind, mixed in) for a cross-session
  check, not used in the estimates.

Steps (``$CHIRP_DATA/labels/...``)::

    python experiments/real_labels/sw_session.py frame   # -> real_v1/sw/frame.parquet, old_items.parquet, frame.json
    python experiments/real_labels/sw_session.py build   # -> candidates/real_sw.parquet, real_v1/sw/truth.parquet
    experiments/real_labels/serve_sw.sh                  # the author labels (port 8766)
    python experiments/real_labels/sw_session.py score   # -> real_v1/report_sw/ (the files of real_v1/report)

``--new-run`` / ``--new-gapfill`` / ``--dest`` for checks: with the v2
production run as the "new" run, every new row matches its own t001 row,
``N`` is empty and ``score`` reproduces real_v1's 2019 numbers at the
thresholds >= 0.05 (``--check-identity``).
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
import build_session as B  # noqa: E402
import score_session as S  # noqa: E402
from common import DATA, LABELS_OUT, ROOT, RUN_2019, VIEW, recording_dir  # noqa: E402

NEW_RUN = DATA / "detections" / "competition2019_v3"
NEW_GAPFILL = DATA / "detections" / "competition2019_v3_gapfill"
SW = ROOT / "sw"  # frame, truth of the new items, build record
CANDIDATES_SW = DATA / "labels" / "candidates" / "real_sw.parquet"
LABELS_SW = DATA / "labels" / "real_sw_labels"
REPORT_SW = ROOT / "report_sw"
OLD_TRUTH = ROOT / "truth" / "real_v1_truth.parquet"
OLD_REPORT = ROOT / "report"

DATASET = "competition2019"
SEED = 20261127
MATCH_IOU = 0.5
P_MIN_OLD_SCORE = 0.05
SCORE_BINS = (0.05, 0.2, 0.5, 0.7, 0.9, 1.0001)
SCORE_NAMES = ("0.05-0.2", "0.2-0.5", "0.5-0.7", "0.7-0.9", "0.9-1")
STATUS_BINS = B.STATUS_BINS
# items of N per score bin x status ("all" below 0.5); set after ``frame`` from N's size
TARGETS_N = {
    "0.05-0.2": {"all": 30}, "0.2-0.5": {"all": 30},
    "0.5-0.7": {"winner": 20, "loser": 15, "other": 6},
    "0.7-0.9": {"winner": 20, "loser": 15, "other": 6},
    "0.9-1": {"winner": 15, "loser": 15, "other": 6},
}
FILLERS = 20
THRESHOLDS = tuple(t for t in S.THRESHOLDS if t >= SCORE_BINS[0])
KEY = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"]
COLS = ["recording", "chirp_id", "t_start_s", "t_end_s", "t_center_s", "f_low_hz", "f_high_hz",
        "score", "kept", "emitter_hz", "track_id", "assign_status", "track_df_other_hz",
        "height_hz", "pp_drop_reason"]


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------
def _recordings(run: Path) -> list[str]:
    manifest = json.loads((run / "run.json").read_text())
    recs = sorted(manifest["recordings"])
    bad = [r for r in recs if manifest["recordings"][r].get("status") != "complete"]
    if bad:
        msg = f"{run}: recordings not complete: {bad}"
        raise SystemExit(msg)
    return recs


def load_run(run: Path, recs: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(run / r / "chirps_rtdetr.parquet", columns=COLS) for r in recs]
    return pd.concat(frames, ignore_index=True)


def in_frame(det: pd.DataFrame) -> np.ndarray:
    """The session's sampling frame (build_session: active phase, background
    and region margins, no earlier real label nearby)."""
    regs, bg, earlier = B.regions(), B.background_segments(), B.earlier_boxes()
    ok = B.frame_mask(det, det["t_center_s"].to_numpy(), 0.0, regs, bg)
    ok[ok] &= ~B.near_earlier(det[ok], earlier)
    return ok


def status_of(det: pd.DataFrame, gapfill_run: Path) -> pd.DataFrame:
    """The assigned fish's trial status (build_session.add_status: the
    gap-filled assignment where the gap-filled run has the row)."""
    parts = []
    for rec, g in det.groupby("recording", sort=False):
        path = gapfill_run / rec / "chirps_rtdetr_gapfill.parquet"
        if path.exists():
            gap = pd.read_parquet(path, columns=[*KEY, "track_id", "assign_status"]).rename(
                columns={"track_id": "gapfill_track_id", "assign_status": "gapfill_assign_status"})
            g = g.merge(gap, on=KEY, how="left", validate="many_to_one")
        else:
            g = g.assign(gapfill_track_id=np.nan, gapfill_assign_status=None)
        parts.append(g)
    det = pd.concat(parts, ignore_index=True)
    has_gap = det["gapfill_assign_status"].notna()
    track = np.where(has_gap, det["gapfill_track_id"], det["track_id"]).astype(float)
    src = np.where(has_gap, det["gapfill_assign_status"], det["assign_status"])
    assigned = np.isin(src, ["assigned", "reattributed"])
    status = np.full(len(det), "other", dtype=object)
    rec = det["recording"].to_numpy()
    for name, r in B.trial_roles().items():
        sel = rec == name
        if sel.any() and r["analysed"]:
            status[sel & assigned & (track == r["winner"])] = "winner"
            status[sel & assigned & (track == r["loser"])] = "loser"
    det["status_track_id"] = np.where(assigned, track, np.nan)
    det["status"] = status
    return det


def match_boxes(a: pd.DataFrame, b: pd.DataFrame, min_iou: float) -> tuple[np.ndarray, np.ndarray,
                                                                             np.ndarray]:
    """One-to-one matches (index into a, index into b, IoU) of the boxes of
    one recording: time-frequency IoU >= min_iou, greedy by IoU."""
    if a.empty or b.empty:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    order = np.argsort(b["t_start_s"].to_numpy(), kind="stable")
    bs = b["t_start_s"].to_numpy()[order]
    be, bl, bh = (b[c].to_numpy()[order] for c in ("t_end_s", "f_low_hz", "f_high_hz"))
    max_dur = float(np.max(be - bs))
    a0, a1, al, ah = (a[c].to_numpy() for c in ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz"))
    lo = np.searchsorted(bs, a0 - max_dur, side="left")
    hi = np.searchsorted(bs, a1, side="left")
    n = hi - lo
    ia = np.repeat(np.arange(len(a)), n)
    ib = np.concatenate([np.arange(x, y) for x, y in zip(lo, hi, strict=True)]) if n.sum() else \
        np.zeros(0, int)
    dt = np.clip(np.minimum(a1[ia], be[ib]) - np.maximum(a0[ia], bs[ib]), 0, None)
    df = np.clip(np.minimum(ah[ia], bh[ib]) - np.maximum(al[ia], bl[ib]), 0, None)
    inter = dt * df
    union = (a1[ia] - a0[ia]) * (ah[ia] - al[ia]) + (be[ib] - bs[ib]) * (bh[ib] - bl[ib]) - inter
    iou = np.where(union > 0, inter / union, 0.0)
    keep = iou >= min_iou
    ia, ib, iou = ia[keep], ib[keep], iou[keep]
    out_a, out_b, out_iou = [], [], []
    used_a, used_b = np.zeros(len(a), bool), np.zeros(len(b), bool)
    for k in np.argsort(-iou, kind="stable"):
        x, y = ia[k], ib[k]
        if used_a[x] or used_b[y]:
            continue
        used_a[x] = used_b[y] = True
        out_a.append(x)
        out_b.append(order[y])
        out_iou.append(iou[k])
    return np.asarray(out_a, int), np.asarray(out_b, int), np.asarray(out_iou)


def frame(args: argparse.Namespace) -> None:
    recs = _recordings(args.new_run)
    old_recs = _recordings(RUN_2019)
    if sorted(set(recs) - set(old_recs)):
        msg = f"new recordings without an old run: {sorted(set(recs) - set(old_recs))}"
        raise SystemExit(msg)
    old = load_run(RUN_2019, recs)
    old["in_frame"] = in_frame(old)
    old["pool"] = old["in_frame"] & old["kept"]
    # check: the old pool per recording is the session's (truth/pool_cells.csv, 2019)
    cells = pd.read_csv(ROOT / "truth" / "pool_cells.csv")
    want = cells[cells["dataset"] == DATASET].groupby("recording")["N"].sum()
    got = old[old["pool"]].groupby("recording").size()
    diff = (got.reindex(want.index).fillna(0) - want).abs()
    if diff.max() > 0:
        msg = f"old pool differs from the session's: {diff[diff > 0].to_dict()}"
        raise SystemExit(msg)
    new = load_run(args.new_run, recs)
    new["in_frame"] = in_frame(new)
    new = status_of(new, args.new_gapfill)
    new["score_bin"] = pd.cut(new["score"], SCORE_BINS, right=False, labels=SCORE_NAMES).astype(str)
    new["size"] = B.size_of(new)
    new["status_stratum"] = np.where(new["score_bin"].isin(STATUS_BINS), new["status"], "all")
    for c in ("old_chirp_id", "old_score", "old_pool", "old_kept", "iou"):
        new[c] = np.nan
    for rec in recs:
        ia, ib, iou = match_boxes(new[new["recording"] == rec], old[old["recording"] == rec], MATCH_IOU)
        ni = np.flatnonzero(new["recording"].to_numpy() == rec)[ia]
        oi = np.flatnonzero(old["recording"].to_numpy() == rec)[ib]
        new.loc[ni, "old_chirp_id"] = old["chirp_id"].to_numpy()[oi]
        new.loc[ni, "old_score"] = old["score"].to_numpy()[oi]
        new.loc[ni, "old_pool"] = old["pool"].to_numpy(float)[oi]  # 1.0 / 0.0, NaN unmatched
        new.loc[ni, "old_kept"] = old["kept"].to_numpy(float)[oi]
        new.loc[ni, "iou"] = iou
    frame_rows = new["in_frame"] & new["kept"]
    new["part"] = np.where(
        frame_rows & (new["old_pool"] == 1) & (new["old_score"] >= P_MIN_OLD_SCORE), "P",
        np.where(frame_rows, "N", ""))
    out = args.dest / "sw" if args.dest else SW
    out.mkdir(parents=True, exist_ok=True)
    new.to_parquet(out / "frame.parquet", index=False)
    # the old items (main strata, 2019) and their new rows
    truth = pd.read_parquet(OLD_TRUTH)
    items = truth[(truth["item_kind"] == "box") & (truth["dataset"] == DATASET)
                  & (truth["repeat_of"] == "")]
    m = new[new["old_chirp_id"].notna()].assign(old_chirp_id=lambda d: d["old_chirp_id"].astype(int))
    keep = ["recording", "old_chirp_id", "chirp_id", "score", "kept", "part", "iou", "track_id",
            "assign_status", "gapfill_track_id", "gapfill_assign_status", "status",
            "status_track_id", "score_bin", "size", "track_df_other_hz", "t_start_s", "t_end_s",
            "f_low_hz", "f_high_hz"]
    old_items = items[["candidate_id", "stratum", "recording", "chirp_id", "score"]].rename(
        columns={"chirp_id": "old_chirp_id", "score": "old_score"}).merge(
        m[keep].add_prefix("new_").rename(columns={"new_recording": "recording",
                                                   "new_old_chirp_id": "old_chirp_id"}),
        on=["recording", "old_chirp_id"], how="left")
    old_items.to_parquet(out / "old_items.parquet", index=False)
    summary = frame_summary(new, old_items)
    (out / "frame.json").write_text(json.dumps(
        {"created": datetime.now(UTC).isoformat(timespec="seconds"), "new_run": str(args.new_run),
         "new_gapfill": str(args.new_gapfill), "old_run": str(RUN_2019), "match_iou": MATCH_IOU,
         "p_min_old_score": P_MIN_OLD_SCORE, **summary}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


def frame_summary(new: pd.DataFrame, old_items: pd.DataFrame) -> dict:
    f = new[new["part"] != ""]
    by_bin = f.groupby(["score_bin", "part"]).size().unstack(fill_value=0)
    matched = f["old_chirp_id"].notna()
    why = np.select([~matched, f["old_pool"] != 1, f["old_score"] < P_MIN_OLD_SCORE],
                    ["no old box", "old box not in the pool (flagged or outside the frame)",
                     f"old score < {P_MIN_OLD_SCORE}"], "P")
    main = old_items[old_items["stratum"].str.startswith("main/")]
    return {
        "new_frame_rows": int(len(f)),
        "P": int((f["part"] == "P").sum()), "N": int((f["part"] == "N").sum()),
        "per_bin": {b: {k: int(v) for k, v in r.items()} for b, r in by_bin.iterrows()},
        "N_reasons": pd.Series(why[f["part"].to_numpy() == "N"]).value_counts().to_dict(),
        "N_share_score_ge_0.5": float(((f["part"] == "N") & (f["score"] >= 0.5)).sum()
                                      / max(1, (f["score"] >= 0.5).sum())),
        "iou_matched": {q: float(f.loc[matched, "iou"].quantile(q)) for q in (0.05, 0.5)},
        "old_main_items": int(len(main)),
        "old_main_items_in_P": int((main["new_part"] == "P").sum()),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(args: argparse.Namespace) -> None:
    base = args.dest / "sw" if args.dest else SW
    new = pd.read_parquet(base / "frame.parquet")
    pool = new[new["part"] == "N"].reset_index(drop=True)
    pool["dataset"] = DATASET
    rng = np.random.default_rng(SEED)
    counts = pool.groupby(["score_bin", "status_stratum", "size"]).size().to_dict()
    strata: dict[tuple, int] = {}
    for b, targets in TARGETS_N.items():
        scaled = {st: int(round(n * args.scale)) for st, n in targets.items()}
        avail = {st: sum(v for k, v in counts.items() if k[:2] == (b, st)) for st in scaled}
        for st, n in B.capped(scaled, avail).items():
            sizes = {k: v for k, v in counts.items() if k[:2] == (b, st)}
            strata |= B.waterfill(sizes, n)
    items = B.sample_cells(pool, strata, ["score_bin", "status_stratum", "size"], rng)
    items["stratum"] = ("new/2019/" + items["score_bin"] + "/" + items["status_stratum"] + "/"
                        + items["size"])
    items["repeat_of"] = ""
    # fillers: old items of P shown again (cross-session check only)
    old_items = pd.read_parquet(base / "old_items.parquet")
    truth = pd.read_parquet(OLD_TRUTH).set_index("candidate_id")
    pick = old_items[(old_items["new_part"] == "P") & old_items["stratum"].str.startswith("main/")]
    pick = pick.iloc[np.sort(rng.choice(len(pick), size=min(FILLERS, len(pick)), replace=False))]
    fill = truth.loc[pick["candidate_id"], ["recording", "t_start_s", "t_end_s", "t_center_s",
                                            "f_low_hz", "f_high_hz"]].reset_index()
    fill = fill.rename(columns={"candidate_id": "repeat_of"})
    fill["stratum"] = "filler/2019"
    allitems = pd.concat([items, fill], ignore_index=True)
    used: set[str] = set(truth.index)
    allitems["candidate_id"] = [B._id(rng, used) for _ in range(len(allitems))]  # noqa: SLF001
    allitems = allitems.iloc[rng.permutation(len(allitems))].reset_index(drop=True)
    allitems["queue_position"] = np.arange(len(allitems))
    allitems["item_kind"] = "box"
    allitems["dataset"] = DATASET
    trk = [B.labeller_tracks(r) for r in allitems["recording"]]
    allitems["tracks"] = [str(p) for p, _ in trk]
    allitems["tracks_source"] = [s for _, s in trk]
    allitems["recording_dir"] = [str(recording_dir(DATASET, r)) for r in allitems["recording"]]
    visible = pd.DataFrame({
        "recording": allitems["recording_dir"], "t_start_s": allitems["t_start_s"].clip(lower=0.0),
        "t_end_s": allitems["t_end_s"], "f_low_hz": allitems["f_low_hz"],
        "f_high_hz": allitems["f_high_hz"], "candidate_id": allitems["candidate_id"],
        "tracks": allitems["tracks"], "item_kind": allitems["item_kind"]})
    cand = args.dest / "candidates" / "real_sw.parquet" if args.dest else CANDIDATES_SW
    if cand.exists() and not args.dest:
        msg = f"{cand} exists: the session was built already (labels refer to its ids)"
        raise SystemExit(msg)
    cand.parent.mkdir(parents=True, exist_ok=True)
    visible.to_parquet(cand, index=False)
    allitems.to_parquet(base / "truth.parquet", index=False)
    rec = {"created": datetime.now(UTC).isoformat(timespec="seconds"), "seed": SEED,
           "scale": args.scale, "targets": TARGETS_N, "fillers": int(len(fill)), "view": VIEW,
           "n_items": int(len(allitems)), "per_stratum": items.groupby("stratum").size().to_dict(),
           "N_frame": int(len(pool)), "candidates_file": str(cand),
           "candidates_sha256": hashlib.sha256(cand.read_bytes()).hexdigest()}
    (base / "build.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k != "view"}, indent=1, default=str))


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def scored_items(base: Path, labels_sw: Path | None) -> tuple[pd.DataFrame, dict]:
    """One row per labelled item of P (old items) and N (new items): part,
    stratum, weight, verdict, the new row's score / status / tracks."""
    old_truth = pd.read_parquet(OLD_TRUTH)
    old = S.design_weights(S.candidate_table(old_truth, S.read_labels(LABELS_OUT)))
    old = old[old["stratum"].str.startswith("main/") & (old["dataset"] == DATASET) & old["labelled"]]
    link = pd.read_parquet(base / "old_items.parquet")
    old = old.merge(link[["candidate_id", "new_part", "new_score", "new_status", "new_track_id",
                          "new_gapfill_track_id", "new_track_df_other_hz", "new_chirp_id"]],
                    on="candidate_id", how="left")
    p = old[old["new_part"] == "P"]
    p = pd.DataFrame({
        "part": "P", "candidate_id": p["candidate_id"], "stratum": "old:" + p["stratum"],
        "recording": p["recording"], "w": p["w"], "y": p["y"], "label": p["label"],
        "emitter_id": p["emitter_id"], "delta_f_hz": p["delta_f_hz"],
        "score": p["new_score"], "status": p["new_status"], "track_id": p["new_track_id"],
        "gapfill_track_id": p["new_gapfill_track_id"], "tracks_source": p["tracks_source"],
        "track_df_other_hz": p["new_track_df_other_hz"], "chirp_id": p["new_chirp_id"]})
    info = {"old_items_labelled": int(len(old)), "old_items_in_P": int(len(p))}
    parts = [p]
    truth_path = base / "truth.parquet"
    if (labels_sw is not None and truth_path.exists()
            and any((labels_sw / "labels").glob("*.csv"))):  # the labeller makes the folder at start
        new_truth = pd.read_parquet(truth_path)
        t = S.candidate_table(new_truth, S.read_labels(labels_sw))
        info["new_items_total"] = int((t["stratum"].str.startswith("new/")).sum())
        info["new_items_labelled"] = int((t["stratum"].str.startswith("new/") & t["labelled"]).sum())
        info["cross_session"] = cross_session(t, old_truth)
        t = S.design_weights(t[t["stratum"].str.startswith("new/")])
        t = t[t["labelled"]]
        parts.append(pd.DataFrame({
            "part": "N", "candidate_id": t["candidate_id"], "stratum": "new:" + t["stratum"],
            "recording": t["recording"], "w": t["w"], "y": t["y"], "label": t["label"],
            "emitter_id": t["emitter_id"], "delta_f_hz": t["delta_f_hz"], "score": t["score"],
            "status": t["status"], "track_id": t["track_id"],
            "gapfill_track_id": t["gapfill_track_id"], "tracks_source": t["tracks_source"],
            "track_df_other_hz": t["track_df_other_hz"], "chirp_id": t["chirp_id"]}))
    else:
        info["new_items_labelled"] = 0
    items = pd.concat(parts, ignore_index=True)
    items["score_bin"] = pd.cut(items["score"], SCORE_BINS, right=False,
                                labels=SCORE_NAMES).astype(str)
    return items.reset_index(drop=True), info


def cross_session(t: pd.DataFrame, old_truth: pd.DataFrame) -> dict:
    """Fillers: the new session's verdict vs real_v1's on the same old item."""
    old = S.candidate_table(old_truth, S.read_labels(LABELS_OUT)).set_index("candidate_id")
    f = t[t["stratum"].str.startswith("filler/") & t["labelled"]]
    rows = [(o["label"], r.label, o["emitter_id"], r.emitter_id) for r in f.itertuples()
            if (o := old.loc[r.repeat_of])["labelled"]]
    if not rows:
        return {"pairs": 0}
    a = pd.DataFrame(rows, columns=["a", "b", "ea", "eb"])
    ya, yb = a["a"].isin(S.POSITIVE), a["b"].isin(S.POSITIVE)
    both = (a["a"] == "chirp") & (a["b"] == "chirp")
    return {"pairs": len(a), "agreement_chirp_vs_not": float((ya == yb).mean()),
            "emitter_pairs": int(both.sum()),
            "emitter_agreement": float((a.loc[both, "ea"] == a.loc[both, "eb"]).mean())
            if both.any() else None}


def precision_tables(items: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, ...]:
    index = S.boot_matrix([g.index.to_numpy() for _, g in items.groupby("stratum")], rng)
    w, y, s = (items[c].to_numpy(float) for c in ("w", "y", "score"))
    part = items["part"].to_numpy()

    def row(m: np.ndarray) -> dict:
        ci = S._boot_ratio(w * y * m, w * m, index)  # noqa: SLF001
        return {"n": int(m.sum()), "n_P": int((m & (part == "P")).sum()),
                "n_N": int((m & (part == "N")).sum()), "N_hat": float(w[m].sum()),
                "N_hat_N": float(w[m & (part == "N")].sum()),
                "precision": S._ratio(w[m] * y[m], w[m]), "ci_low": ci[0], "ci_high": ci[1]}  # noqa: SLF001

    by_bin = pd.DataFrame([{"dataset": "2019", "score_bin": b, **row(items["score_bin"].to_numpy() == b),
                            "mean_score": S._ratio(w[items["score_bin"] == b] * s[items["score_bin"] == b],  # noqa: SLF001
                                                   w[items["score_bin"] == b])}
                           for b in SCORE_NAMES])
    at_thr = pd.DataFrame([{"dataset": "2019", "threshold": t, **row(s >= t)} for t in THRESHOLDS])
    rows = []
    for status in ("winner", "loser", "other"):
        sm = items["status"].to_numpy() == status
        groups = [(b, items["score_bin"].to_numpy() == b) for b in SCORE_NAMES]
        groups += [(f">= {t}", s >= t) for t in (0.5, 0.7, 0.9)] + [("0.5-0.9", (s >= 0.5) & (s < 0.9))]
        for name, gm in groups:
            m = sm & gm
            if m.any():
                r = row(m)
                rows.append({"dataset": "2019", "status": status, "scores": name, **r,
                             "N_hat_chirps": float((w * y)[m].sum())})
    return by_bin, at_thr, pd.DataFrame(rows)


def recall_tables(new_run: Path, rng: np.random.Generator) -> dict:
    """real_v1's windows (2019) and marks against the new run's kept detections."""
    truth = pd.read_parquet(OLD_TRUTH)
    win, marks = S.window_table(truth, S.read_labels(LABELS_OUT))
    win = win[win["dataset"] == DATASET]
    saved = S.RUN_2019, S.THRESHOLDS
    try:  # score_session reads its module constants
        S.RUN_2019, S.THRESHOLDS = new_run, THRESHOLDS
        out = S.recall_tables(win, marks, rng)
    finally:
        S.RUN_2019, S.THRESHOLDS = saved
    if not out["recall"].empty:
        out["recall"] = out["recall"][out["recall"]["dataset"] == "2019"].reset_index(drop=True)
    return out


def emitter_table(items: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    c = items[(items["label"] == "chirp") & items["emitter_id"].notna()].copy()
    df = c["delta_f_hz"].where(c["delta_f_hz"].notna(), c["track_df_other_hz"])
    c["df_bin"] = pd.cut(df, S.DF_BINS, right=False, labels=S.DF_NAMES).astype(str)
    rows = []
    for model, col in (("production", "track_id"), ("gap-filled (2019 trials)", "gapfill_track_id")):
        sub = c if col == "track_id" else c[c["tracks_source"] == "gapfill"]
        for b, g in [("all", sub), *sub.groupby("df_bin")]:
            assigned = g[col].notna()
            agree = g.loc[assigned, col].astype(float) == g.loc[assigned, "emitter_id"].astype(float)
            ww = g.loc[assigned, "w"]
            rows.append({"source": "candidates", "model": model, "df_bin": b, "n": len(g),
                         "assigned": int(assigned.sum()),
                         "agreement": float(agree.mean()) if len(agree) else np.nan,
                         "agreement_weighted": S._ratio(ww * agree, ww) if len(agree) else np.nan})  # noqa: SLF001
    if not pairs.empty:
        p = pairs[(pairs["mark_label"] == "chirp") & pairs["mark_emitter"].notna()].copy()
        p["df_bin"] = pd.cut(p["mark_delta_f_hz"], S.DF_BINS, right=False, labels=S.DF_NAMES).astype(str)
        for b, g in [("all", p), *p.groupby("df_bin")]:
            assigned = g["det_track_id"].notna()
            agree = g.loc[assigned, "det_track_id"].astype(float) == g.loc[assigned, "mark_emitter"].astype(float)
            rows.append({"source": "window marks (t>=0.5)", "model": "production", "df_bin": b,
                         "n": len(g), "assigned": int(assigned.sum()),
                         "agreement": float(agree.mean()) if len(agree) else np.nan,
                         "agreement_weighted": S._ratio(g.loc[assigned, "weight"] * agree,  # noqa: SLF001
                                                        g.loc[assigned, "weight"]) if len(agree) else np.nan})
    return pd.DataFrame(rows)


def score(args: argparse.Namespace) -> None:
    base = args.dest / "sw" if args.dest else SW
    out = args.dest / "report_sw" if args.dest else REPORT_SW
    rng = np.random.default_rng(S.SEED)
    items, info = scored_items(base, args.labels_sw)
    by_bin, at_thr, by_status = precision_tables(items, rng)
    rec = recall_tables(args.new_run, rng)
    emit = emitter_table(items, rec["pairs"])
    old_metrics = json.loads((OLD_REPORT / "metrics.json").read_text())
    frame_info = json.loads((base / "frame.json").read_text())
    complete = info["new_items_labelled"] > 0 or frame_info["N"] == 0
    progress = {
        "candidates_labelled": old_metrics["progress"]["candidates_labelled"] + info["new_items_labelled"],
        "old_candidates_labelled_2019": info["old_items_labelled"], "old_items_in_P": info["old_items_in_P"],
        "new_items_labelled": info["new_items_labelled"], "new_items_total": info.get("new_items_total", 0),
        # session totals (real_v1, both datasets), as candidates_labelled; the
        # windows and marks of 2019/2020 that the recall uses are in recall.csv
        "windows_done": old_metrics["progress"]["windows_done"],
        "marks": old_metrics["progress"]["marks"],
        "windows_2019": int(rec["per_window"].shape[0]),
        "marks_2019": int(rec["per_window"]["marks"].sum()) if len(rec["per_window"]) else 0,
        "N_estimated": bool(complete),
    }
    metrics = {"progress": progress, "frame": {k: frame_info[k] for k in (
        "new_frame_rows", "P", "N", "N_share_score_ge_0.5", "match_iou", "p_min_old_score")},
        "retest": old_metrics["retest"], "cross_session": info.get("cross_session", {"pairs": 0}),
        "new_run": str(args.new_run)}
    out.mkdir(parents=True, exist_ok=True)
    by_bin.to_csv(out / "precision_by_bin.csv", index=False)
    at_thr.to_csv(out / "precision_at_threshold.csv", index=False)
    by_status.to_csv(out / "precision_by_status.csv", index=False)
    rec["recall"].to_csv(out / "recall.csv", index=False)
    rec["per_window"].to_csv(out / "windows.csv", index=False)
    emit.to_csv(out / "emitter_agreement.csv", index=False)
    items.to_csv(out / "candidates_scored.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    warn = [] if complete else ["**N not labelled yet: the precision covers P only (not the new detector's "
                                "whole frame).**", ""]
    md = ["# real_v1 scored for the single-window detector (P: real_v1 verdicts; N: session real_sw)", "",
          *warn, f"Progress: {json.dumps(progress)}", "", f"Frame: {json.dumps(metrics['frame'])}", "",
          "## Precision by new score bin", "", S._md(by_bin), "", "## Precision at thresholds", "",  # noqa: SLF001
          S._md(at_thr), "", "## Precision by the assigned fish's trial status", "", S._md(by_status),  # noqa: SLF001
          "", "## Recall (real_v1 windows, 2019/2020)", "", S._md(rec["recall"]), "",  # noqa: SLF001
          "## Emitter agreement", "", S._md(emit), "", "## Cross-session check (fillers)", "",  # noqa: SLF001
          "```", json.dumps(metrics["cross_session"], indent=1, default=float), "```", ""]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    if args.check_identity:
        check_identity(out)


def check_identity(out: Path) -> None:
    """With the v2 production run as the "new" run: real_v1's 2019 numbers at >= 0.05."""
    a = pd.read_csv(OLD_REPORT / "precision_at_threshold.csv")
    a = a[(a["dataset"] == "2019") & (a["threshold"] >= SCORE_BINS[0])].set_index("threshold")
    b = pd.read_csv(out / "precision_at_threshold.csv").set_index("threshold")
    ra = pd.read_csv(OLD_REPORT / "recall.csv")
    ra = ra[(ra["dataset"] == "2019") & (ra["threshold"] >= SCORE_BINS[0])].set_index("threshold")
    rb = pd.read_csv(out / "recall.csv").set_index("threshold")
    sa = pd.read_csv(OLD_REPORT / "precision_by_status.csv")
    sa = sa[(sa["dataset"] == "2019") & sa["scores"].isin(STATUS_BINS)].set_index(["status", "scores"])
    sb = pd.read_csv(out / "precision_by_status.csv").set_index(["status", "scores"])
    checks = {
        "precision": (a["precision"] - b.loc[a.index, "precision"]).abs().max(),
        "precision_n": (a["n"] - b.loc[a.index, "n"]).abs().max(),
        "recall": (ra["recall"] - rb.loc[ra.index, "recall"]).abs().max(),
        "recall_found": (ra["found"] - rb.loc[ra.index, "found"]).abs().max(),
        "status_precision": (sa["precision"] - sb.loc[sa.index, "precision"]).abs().max(),
    }
    print("identity check (max abs difference to real_v1/report, 2019):", checks)
    if max(checks.values()) > 1e-9:
        raise SystemExit("identity check failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["frame", "build", "score"])
    ap.add_argument("--new-run", type=Path, default=NEW_RUN)
    ap.add_argument("--new-gapfill", type=Path, default=NEW_GAPFILL)
    ap.add_argument("--dest", type=Path, help="write everything under DEST (checks)")
    ap.add_argument("--labels-sw", type=Path, default=LABELS_SW)
    ap.add_argument("--scale", type=float, default=1.0, help="build: multiply TARGETS_N")
    ap.add_argument("--check-identity", action="store_true")
    args = ap.parse_args()
    {"frame": frame, "build": build, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
