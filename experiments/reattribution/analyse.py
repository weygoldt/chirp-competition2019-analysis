"""Statistics of harmonic-copy handling (drop / re-attribution), production rules.

Production (chirpdetector v2 fde5e3e, pipeline/run.py `_postprocess`):
harmonic and duplicate flags on every stored row (commute with the
threshold), re-attribution on the rows at or above the operating
threshold. That equals `flag_detections(rows >= t, tracks,
PostprocessConfig())` with the pipeline's assignment as ``emitter_id``;
here the assignment is `assign_by_frequency` with the pipeline's tolerance
(20 Hz) and gap (10 s) for sim and the labelled recordings, and the stored
``track_id`` for the dry runs.

Inputs: detections of eval_postprocess.py (sim_v4 test tiles, the four
labelled recordings; score >= 0.05) and the dry run r3 tables.
Output: tables (CSV/JSON) in $CHIRP_DATA/runs/reattribution_report/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA, FAST, REC_2019  # noqa: E402


import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from gridtools.io import load_tracks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from eval_postprocess import LABEL_FILES, TRACK_DIRS  # noqa: E402

from chirpdetector.data import dataset_ground_truth, load_dataset  # noqa: E402
from chirpdetector.evaluation import (  # noqa: E402
    EvaluationConfig,
    GroundTruth,
    assign_by_frequency,
    evaluate,
    read_ground_truth,
)
from chirpdetector.postprocess import COLUMNS, PostprocessConfig, flag_detections  # noqa: E402

OUT = DATA / "runs/reattribution_report"
RUNS = {"dual-head": "dualhead_r18_sim_v4_r1", "YOLO26s": "yolo26s_sim_v4_r1"}
SIM = FAST / "datasets/sim_v4"
DRY = DATA / "detections/dryrun_20261113_r3"
DRY_RECS = {
    "2019-11-25-09_59": str(REC_2019 / "2019-11-25-09_59"),
    "2020-05-18-10_00": str(REC_2019 / "2020-05-18-10_00"),
}
THRESHOLDS = (0.3, 0.5, 0.7)
PROD = PostprocessConfig()
ON = replace(PROD, reattribute_harmonics=True)
OFF = replace(PROD, reattribute_harmonics=False)
ASSIGN_TOL, GAP = 20.0, 10.0
OLD = ("drop_reason", "harmonic_of_id", "harmonic_pair_of", "duplicate_of")


# ------------------------------------------------------------------ loading


def _strip(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in (*OLD, *COLUMNS, "emitter_id", "emitter_df_hz") if c in df.columns]
    return df.drop(columns=drop).reset_index(drop=True)


def _assign(preds: pd.DataFrame, tracks) -> pd.DataFrame:
    if "emitter_hz" not in preds.columns:
        return preds
    return assign_by_frequency(preds, tracks, ASSIGN_TOL, max_gap_s=GAP)


def load_sim(detector: str):
    run = RUNS[detector]
    raw = pd.read_parquet(f"{DATA}/runs/{run}/eval_postprocess/sim_test_predictions_flagged.parquet")
    ds = load_dataset(SIM, check_files=False)
    ids = list(ds.manifest.image_ids("test"))
    truth = dataset_ground_truth(ds, ids)
    cache, tracks = {}, {}
    for i in ids:
        s = ds.manifest.image(i).source
        if s not in cache:
            cache[s] = load_tracks(ds.tracks_dir(s))
        tracks[i] = cache[s]
    return _assign(_strip(raw), tracks), truth, tracks


def load_labelled(detector: str):
    run = RUNS[detector]
    raw = pd.read_parquet(f"{DATA}/runs/{run}/eval_postprocess/real_predictions_flagged.parquet")
    truth = GroundTruth.concat([read_ground_truth(p) for p in LABEL_FILES])
    tracks = {n: load_tracks(p) for n, p in TRACK_DIRS.items()}
    return _assign(_strip(raw), tracks), truth, tracks


def load_dry():
    """Dry-run tables, with the detected values restored and the pipeline's
    assignment as ``emitter_id`` (re-attributed rows: unassigned)."""
    frames, tracks, hours = [], {}, {}
    run = json.loads((DRY / "run.json").read_text())
    for rec, path in DRY_RECS.items():
        d = pd.read_parquet(DRY / rec / "chirps_rtdetr.parquet")
        again = d["pp_reattributed"].astype(str).to_numpy() == "harmonic"
        for col in ("emitter_hz", "f_low_hz", "f_high_hz"):
            d.loc[again, col] = d.loc[again, f"pp_harmonic_{col}"]
        ids = d["track_id"].astype("Float64").round().astype("Int64")
        ids[again] = pd.NA
        frames.append(
            pd.DataFrame(
                {
                    "recording": rec,
                    "t_start_s": d["t_start_s"],
                    "t_end_s": d["t_end_s"],
                    "f_low_hz": d["f_low_hz"],
                    "f_high_hz": d["f_high_hz"],
                    "score": d["score"],
                    "emitter_hz": d["emitter_hz"],
                    "emitter_id": ids,
                    "stored_drop_reason": d["pp_drop_reason"].astype(str),
                    "stored_reattributed": d["pp_reattributed"].astype(str),
                }
            )
        )
        tracks[rec] = load_tracks(path)
        hours[rec] = run["recordings"][rec]["duration_s"] / 3600
    return pd.concat(frames, ignore_index=True), tracks, hours


# ------------------------------------------------------------------ helpers


def at(preds: pd.DataFrame, t: float, tracks, config: PostprocessConfig) -> pd.DataFrame:
    sel = preds.loc[preds["score"].to_numpy() >= t].reset_index(drop=True)
    return flag_detections(sel, tracks, config)


def kept(flagged: pd.DataFrame) -> pd.DataFrame:
    return flagged.loc[flagged["drop_reason"].astype(str).to_numpy() == ""]


def is_copy(flagged: pd.DataFrame) -> np.ndarray:
    return (flagged["drop_reason"].astype(str).to_numpy() == "harmonic") | (
        flagged["reattributed"].astype(str).to_numpy() == "harmonic"
    )


def evaluate_at(rows: pd.DataFrame, truth, t: float):
    """Evaluate rows (index = row ids of the flagged table) at threshold t.

    Returns the pooled counts and, per row id, the matched outcome and
    label row (via items.pred_row = position in ``rows``)."""
    table = rows.drop(columns=[c for c in COLUMNS if c in rows.columns]).reset_index(drop=True)
    config = EvaluationConfig(thresholds=(t,), reference_threshold=t, n_bootstrap=0)
    result = evaluate(table, truth, config=config)
    pooled = result.summary[result.summary["level"] == "all"].iloc[0]
    counts = {k: int(pooled[k]) for k in ("labels", "tp", "fp", "fn")}
    counts.update({k: float(pooled[k]) for k in ("precision", "recall", "f1")})
    items = result.items
    m = items[items["pred_row"].notna()].drop_duplicates("pred_row")
    pos = m["pred_row"].astype(int).to_numpy()
    match = pd.DataFrame(
        {
            "outcome": m["outcome"].to_numpy(),
            "label_row": m["label_row"].to_numpy(),
        },
        index=rows.index.to_numpy()[pos],
    )
    labels = items[items["label_row"].notna() & items["outcome"].isin(["tp", "fn"])]
    lab = pd.Series(labels["outcome"].to_numpy(), index=labels["label_row"].astype(int).to_numpy())
    lab = lab[~lab.index.duplicated()]
    return counts, match, lab


def _f_at(tracks, k: int, t: float) -> float:
    try:
        return float(tracks.track(int(k)).frequency_at(np.array([t]), GAP)[0])
    except KeyError:
        return float("nan")


def fundamental_partner(flagged: pd.DataFrame, tracks_of, config=PROD) -> pd.DataFrame:
    """For every dropped copy: the kept detection that counts as its
    fundamental (temporal IoU >= reattribute_min_tiou, spanning f_k or
    assigned to k; best IoU), with Δt and scores."""
    rows = []
    kp = kept(flagged)
    drop = flagged.index[flagged["drop_reason"].astype(str).to_numpy() == "harmonic"]
    for r in drop:
        c = flagged.loc[r]
        k = int(c["harmonic_of_id"])
        rec = c["recording"]
        tc = 0.5 * (c["t_start_s"] + c["t_end_s"])
        f_k = _f_at(tracks_of[rec], k, tc)
        g = kp[kp["recording"] == rec]
        inter = np.clip(np.minimum(g["t_end_s"], c["t_end_s"]) - np.maximum(g["t_start_s"], c["t_start_s"]), 0, None)
        union = np.maximum(g["t_end_s"], c["t_end_s"]) - np.minimum(g["t_start_s"], c["t_start_s"])
        tiou = (inter / union).to_numpy()
        span = ((g["f_low_hz"] <= f_k) & (g["f_high_hz"] >= f_k)).to_numpy()
        own = np.zeros(len(g), bool)
        if "emitter_id" in g.columns:
            own = (g["emitter_id"].astype("Int64") == k).fillna(False).to_numpy(bool)
        else:
            w0, w1 = config.edge_fundamental_window_hz
            own = ((g["f_low_hz"] + w0 <= f_k) & (g["f_low_hz"] + w1 >= f_k)).to_numpy()
        ok = (tiou >= config.reattribute_min_tiou) & (span | own)
        rec_row = {"row": r, "recording": rec, "t_center_s": tc, "copy_score": float(c["score"]), "k": k, "f_k": f_k}
        if ok.any():
            j = int(np.argmax(np.where(ok, tiou, -1)))
            p = g.iloc[j]
            rec_row.update(
                {
                    "partner_row": int(g.index[j]),
                    "fund_score": float(p["score"]),
                    "dt_s": tc - 0.5 * (p["t_start_s"] + p["t_end_s"]),
                    "tiou": float(tiou[j]),
                    "fund_emitter_id": (int(p["emitter_id"]) if "emitter_id" in g.columns and pd.notna(p["emitter_id"]) else -1),
                    "fund_assigned_to_k": bool(own[j]) if "emitter_id" in g.columns else None,
                }
            )
        rows.append(rec_row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 1: triggers


def triggers(name: str, detector: str, preds: pd.DataFrame, tracks, hours=None) -> list[dict]:
    out = []
    for t in THRESHOLDS:
        prod = at(preds, t, tracks, PROD)
        n = len(prod)
        copies = is_copy(prod)
        again = prod["reattributed"].astype(str).to_numpy() == "harmonic"
        dropped = prod["drop_reason"].astype(str).to_numpy() == "harmonic"
        row = {
            "data": name,
            "detector": detector,
            "threshold": t,
            "detections": n,
            "copies": int(copies.sum()),
            "copies_per_1000": 1000 * copies.sum() / n if n else np.nan,
            "dropped": int(dropped.sum()),
            "reattributed": int(again.sum()),
            "share_reattributed": again.sum() / copies.sum() if copies.sum() else np.nan,
            "duplicates": int((prod["drop_reason"].astype(str) == "duplicate").sum()),
            "chirps_kept": int((prod["drop_reason"].astype(str) == "").sum()),
        }
        if "emitter_hz" not in preds.columns:
            on = at(preds, t, tracks, ON)
            row["reattributed_if_forced_on"] = int((on["reattributed"].astype(str) == "harmonic").sum())
        if hours is not None:
            for rec, h in hours.items():
                sel = prod["recording"].to_numpy() == rec
                row[f"{rec}_copies_per_h"] = float(np.sum(copies & sel) / h)
                row[f"{rec}_reattributed_per_h"] = float(np.sum(again & sel) / h)
                row[f"{rec}_chirps_per_h"] = float(np.sum((prod["drop_reason"].astype(str).to_numpy() == "") & sel) / h)
        out.append(row)
    return out


# ------------------------------------------------- 2-4: sim, truth known


def _gt_class(flagged: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Best-IoU ground-truth box of every detection: class, emitter, chirp."""
    gt, gt_iou, gt_em, gt_ch, index = [], [], [], [], []
    by_rec = dict(tuple(labels.groupby("recording")))
    for rec, g in flagged.groupby("recording", sort=False):
        b = by_rec.get(rec)
        for i, p in zip(g.index, g.itertuples(index=False), strict=True):
            index.append(i)
            if b is None:
                gt.append("none"); gt_iou.append(0.0); gt_em.append(-1); gt_ch.append(-1)
                continue
            it = np.clip(np.minimum(p.t_end_s, b["t_end_s"].to_numpy()) - np.maximum(p.t_start_s, b["t_start_s"].to_numpy()), 0, None)
            iff = np.clip(np.minimum(p.f_high_hz, b["f_high_hz"].to_numpy()) - np.maximum(p.f_low_hz, b["f_low_hz"].to_numpy()), 0, None)
            inter = it * iff
            union = (p.t_end_s - p.t_start_s) * (p.f_high_hz - p.f_low_hz) + ((b["t_end_s"] - b["t_start_s"]) * (b["f_high_hz"] - b["f_low_hz"])).to_numpy() - inter
            iou = inter / union
            j = int(np.argmax(iou))
            if iou[j] <= 0:
                gt.append("none"); gt_iou.append(0.0); gt_em.append(-1); gt_ch.append(-1)
                continue
            lab = b.iloc[j]
            gt.append("chirp" if lab["label"] == "chirp" else str(lab["reason"]))
            gt_iou.append(float(iou[j]))
            gt_em.append(int(lab["emitter_id"]) if pd.notna(lab["emitter_id"]) else -1)
            gt_ch.append(int(lab["chirp_id"]) if pd.notna(lab["chirp_id"]) else -1)
    res = pd.DataFrame({"gt": gt, "gt_iou": gt_iou, "gt_emitter": gt_em, "gt_chirp": gt_ch}, index=index)
    return res.loc[flagged.index]


def sim_job(args: tuple[str, float]) -> dict:
    detector, t = args
    cache = OUT / "tables" / f"sim_job_{detector}_{t}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = _sim_job(detector, t)
    cache.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return json.loads(cache.read_text())


def _sim_job(detector: str, t: float) -> dict:
    preds, truth, tracks = load_sim(detector)
    labels = truth.labels.reset_index(drop=True)
    prod = at(preds, t, tracks, PROD)
    head = "emitter_hz" in preds.columns
    out: dict = {"detector": detector, "threshold": t}
    # --- variants evaluated at t
    copies = prod["drop_reason"].astype(str).to_numpy() == "harmonic"
    again = prod["reattributed"].astype(str).to_numpy() == "harmonic"
    k_prod = kept(prod)
    c_prod, m_prod, l_prod = evaluate_at(k_prod, truth, t)
    with_copies = prod.loc[(prod["drop_reason"].astype(str).to_numpy() == "") | copies]
    c_back, m_back, l_back = evaluate_at(with_copies, truth, t)
    off = at(preds, t, tracks, OFF if head else PROD)
    c_off, m_off, l_off = evaluate_at(kept(off), truth, t)
    out["counts"] = {"production": c_prod, "copies_kept": c_back, "no_reattribution": c_off}
    # --- 2a: dropped copies: what they matched when kept
    mb = m_back.reindex(prod.index[copies])
    lab_reason = labels["reason"].astype(str)
    kinds = []
    for r, row in mb.iterrows():
        if pd.isna(row["outcome"]):
            kinds.append("unmatched")
        elif row["outcome"] == "tp":
            lab = labels.iloc[int(row["label_row"])]
            same = int(lab["emitter_id"]) == int(prod.loc[r, "harmonic_of_id"]) if pd.notna(lab["emitter_id"]) else False
            kinds.append("tp_label_of_same_fish" if same else "tp_label_of_other_fish")
        elif row["outcome"] == "ignored":
            kinds.append(f"ignored_{lab_reason.iloc[int(row['label_row'])]}")
        else:
            kinds.append(str(row["outcome"]))
    out["dropped_copies_matched_when_kept"] = pd.Series(kinds, dtype=object).value_counts().to_dict()
    lost = l_back.index[(l_back == "tp") & (l_prod.reindex(l_back.index) == "fn")]
    out["labels_lost_by_dropping"] = len(lost)
    lost_rows = []
    for li in lost:
        lab = labels.iloc[int(li)]
        lost_rows.append({"recording": lab["recording"], "t_center_s": float(lab["t_center_s"]), "label_f_low_hz": float(lab["f_low_hz"]), "label_f_high_hz": float(lab["f_high_hz"]), "label_emitter": lab["emitter_id"]})
    out["labels_lost_examples"] = lost_rows[:10]
    # --- 2b: re-attributed rows at the fundamental
    ra = prod.index[again]
    mr = m_prod.reindex(ra)
    verdict = []
    for r in ra:
        row = mr.loc[r]
        if pd.isna(row["outcome"]):
            verdict.append("unmatched")
        elif row["outcome"] == "tp":
            lab = labels.iloc[int(row["label_row"])]
            verdict.append("tp_right_fish" if pd.notna(lab["emitter_id"]) and int(lab["emitter_id"]) == int(prod.loc[r, "harmonic_of_id"]) else "tp_wrong_fish")
        else:
            verdict.append(str(row["outcome"]))
    out["reattributed"] = pd.Series(verdict, dtype=object).value_counts().to_dict()
    ex = prod.loc[ra, ["recording", "t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score", "harmonic_of_id", "harmonic_f_low_hz", "harmonic_f_high_hz"]].copy()
    ex["verdict"] = verdict
    out["reattributed_examples"] = ex.head(20).to_dict(orient="records")
    # if forced on for box-only
    if not head:
        on = at(preds, t, tracks, ON)
        c_on, m_on, _ = evaluate_at(kept(on), truth, t)
        out["counts"]["forced_on"] = c_on
        ra_on = on.index[on["reattributed"].astype(str).to_numpy() == "harmonic"]
        mo = m_on.reindex(ra_on)
        out["reattributed_if_forced_on"] = mo["outcome"].fillna("unmatched").value_counts().to_dict()
    # --- 3: upper bound
    chirps = labels.index[(labels["label"] == "chirp").to_numpy()]
    harm = labels[(labels["label"] == "ignore") & (labels["reason"].astype(str) == "harmonic")]
    hkey = {(r, c): i for i, r, c in zip(harm.index, harm["recording"], harm["chirp_id"], strict=True)}
    sel_all = preds.loc[preds["score"].to_numpy() >= t]
    by_rec = dict(tuple(sel_all.groupby("recording")))
    n_fn_off = n_harm_det = n_rec = 0
    for li in chirps:
        if l_off.get(li) != "fn":
            continue
        n_fn_off += 1
        lab = labels.loc[li]
        hi = hkey.get((lab["recording"], lab["chirp_id"]))
        if hi is None:
            continue
        h = labels.loc[hi]
        g = by_rec.get(lab["recording"])
        if g is None:
            continue
        hc = h["t_center_s"]
        gc = 0.5 * (g["t_start_s"] + g["t_end_s"])
        hit = (np.abs(gc - hc) <= 0.1) & (g["f_low_hz"] < h["f_high_hz"]) & (g["f_high_hz"] > h["f_low_hz"])
        if hit.any():
            n_harm_det += 1
            if l_prod.get(li) == "tp":
                n_rec += 1
    out["upper_bound"] = {
        "visible_chirp_labels": len(chirps),
        "fn_without_reattribution": n_fn_off,
        "fn_with_detected_harmonic": n_harm_det,
        "recovered_by_reattribution": n_rec,
        "fn_with_reattribution": int(np.sum(l_prod.reindex(chirps) == "fn")),
    }
    # --- 4: both detected; kept-by-mistake copies
    pairs = fundamental_partner(prod, tracks)
    if len(pairs):
        has = pairs["partner_row"].notna() if "partner_row" in pairs else pd.Series(False, index=pairs.index)
        p = pairs[has]
        out["pairs"] = {
            "dropped_copies": len(pairs),
            "with_partner": int(has.sum()),
            "dt_s_quantiles_5_50_95": np.nanquantile(p["dt_s"], [0.05, 0.5, 0.95]).round(4).tolist() if len(p) else [],
            "abs_dt_s_max": float(np.abs(p["dt_s"]).max()) if len(p) else None,
            "tiou_quantiles_5_50": np.nanquantile(p["tiou"], [0.05, 0.5]).round(3).tolist() if len(p) else [],
            "copy_score_higher": float(np.mean(p["copy_score"] > p["fund_score"])) if len(p) else None,
            "score_diff_quantiles_5_50_95": np.nanquantile(p["copy_score"] - p["fund_score"], [0.05, 0.5, 0.95]).round(3).tolist() if len(p) else [],
        }
        if head and len(p):
            nb = p[(p["fund_emitter_id"] >= 0) & (p["fund_emitter_id"] != p["k"])]
            una = p[p["fund_emitter_id"] < 0]
            truth_k = truth_j = 0
            if len(nb):
                prow = nb["partner_row"].astype(int).to_numpy()
                gtc = _gt_class(prod.loc[np.unique(prow)], labels)
                em = gtc.loc[prow, "gt_emitter"].to_numpy(int)
                truth_k = int(np.sum(em == nb["k"].to_numpy(int)))
                truth_j = int(np.sum(em == nb["fund_emitter_id"].to_numpy(int)))
            out["pairs"]["fundamental_assigned_to_other_fish"] = {
                "n": len(nb),
                "truth_is_copy_fish": truth_k,
                "truth_is_assigned_fish": truth_j,
                "neither": len(nb) - truth_k - truth_j,
            }
            out["pairs"]["fundamental_unassigned"] = len(una)
        pairs.to_csv(OUT / "tables" / f"sim_pairs_{detector}_{t}.csv", index=False)
    # kept copies: detections on a harmonic region that were not flagged
    gt = _gt_class(prod, labels)
    on_harm = (gt["gt"].to_numpy() == "harmonic") & (gt["gt_iou"].to_numpy(float) >= 0.3)
    flagged_copy = is_copy(prod)
    missed = prod.index[on_harm & ~flagged_copy & (prod["drop_reason"].astype(str).to_numpy() == "")]
    causes = []
    for r in missed:
        c = prod.loc[r]
        k = int(gt.loc[r, "gt_emitter"])
        tc = 0.5 * (c["t_start_s"] + c["t_end_s"])
        tr = tracks[c["recording"]]
        f_k = _f_at(tr, k, tc)
        live = tr.frequencies_at(tc, GAP)
        if not np.isfinite(f_k):
            causes.append("own track not live (gap / outside track)")
            continue
        if head:
            e = float(c["emitter_hz"])
            if abs(e - 2 * f_k) > PROD.harmonic_tolerance_hz:
                causes.append("emitter_hz > 15 Hz from 2 f_k")
            elif any(abs(e - f) <= PROD.fundamental_tolerance_hz for f in live.values()) or pd.notna(c.get("emitter_id")):
                causes.append("protected: a fish within 50 Hz of the copy (or assigned)")
            else:
                causes.append("other")
        else:
            lo = c["f_low_hz"]
            w0, w1 = PROD.edge_harmonic_window_hz
            f0, f1 = PROD.edge_fundamental_window_hz
            if not (lo + w0 <= 2 * f_k <= lo + w1):
                causes.append("2 f_k outside the lower-edge window")
            elif any(lo + f0 <= f <= lo + f1 for f in live.values()):
                causes.append("protected: a fish in the fundamental window")
            else:
                causes.append("other")
    out["copies_on_harmonic_regions"] = int(on_harm.sum())
    out["copies_flagged"] = int(np.sum(on_harm & flagged_copy))
    out["copies_kept_by_mistake"] = pd.Series(causes, dtype=object).value_counts().to_dict()
    # copies flagged but not on a harmonic region (possible wrong drops)
    wrong = prod.index[flagged_copy & ~on_harm]
    out["flagged_not_on_harmonic_region"] = gt.loc[wrong, "gt"].value_counts().to_dict()
    return out


# ------------------------------------------------------- real: pairs, kept


def real_pairs(name: str, preds, tracks, t: float = 0.5) -> dict:
    prod = at(preds, t, tracks, PROD)
    pairs = fundamental_partner(prod, tracks)
    out = {"data": name, "threshold": t, "dropped_copies": len(pairs)}
    if len(pairs) and "partner_row" in pairs:
        p = pairs[pairs["partner_row"].notna()]
        out.update(
            {
                "with_partner": len(p),
                "dt_s_quantiles_5_50_95": np.nanquantile(p["dt_s"], [0.05, 0.5, 0.95]).round(4).tolist(),
                "copy_score_higher": float(np.mean(p["copy_score"] > p["fund_score"])),
                "score_diff_quantiles_5_50_95": np.nanquantile(p["copy_score"] - p["fund_score"], [0.05, 0.5, 0.95]).round(3).tolist(),
                "fundamental_assigned_to_other_fish": int(np.sum((p["fund_emitter_id"] >= 0) & (p["fund_emitter_id"] != p["k"]))),
                "fundamental_unassigned": int(np.sum(p["fund_emitter_id"] < 0)),
            }
        )
    pairs.to_csv(OUT / "tables" / f"real_pairs_{name}.csv", index=False)
    # candidate kept copies: kept rows whose emitter lies within 15 Hz of twice a live track
    kp = kept(prod)
    cand = 0
    if "emitter_hz" in kp.columns:
        for p in kp.itertuples(index=False):
            live = tracks[p.recording].frequencies_at(0.5 * (p.t_start_s + p.t_end_s), GAP)
            if any(abs(p.emitter_hz - 2 * f) <= PROD.harmonic_tolerance_hz for f in live.values()):
                cand += 1
    out["kept_rows_at_twice_a_track"] = cand
    return out


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    summary: dict = {"production_config": PROD.to_dict(), "assignment": {"tolerance_hz": ASSIGN_TOL, "max_gap_s": GAP}}
    # 1: triggers (cached in tables/triggers.csv)
    cached = (OUT / "tables" / "triggers.csv").exists()
    trig = pd.read_csv(OUT / "tables" / "triggers.csv").to_dict(orient="records") if cached else []
    for det in RUNS:
        if not cached:
            preds, truth, tracks = load_sim(det)
            trig += triggers("sim_v4 test", det, preds, tracks)
        preds, truth, tracks = load_labelled(det)
        if not cached:
            trig += triggers("labelled real", det, preds, tracks)
        summary.setdefault("real_pairs", []).append(real_pairs(f"labelled_{det}", preds, tracks))
    dry, dtracks, hours = load_dry()
    if not cached:
        trig += triggers("dry runs (2 recordings)", "dual-head", dry, dtracks, hours)
    summary["real_pairs"].append(real_pairs("dryrun_dual-head", dry, dtracks))
    # stored flags agree with the re-run at 0.5?
    prod = at(dry, 0.5, dtracks, PROD)
    stored = dry.loc[dry["score"] >= 0.5].reset_index(drop=True)
    summary["dryrun_rerun_matches_stored"] = {
        "drop_reason": int(np.sum(prod["drop_reason"].astype(str).to_numpy() != stored["stored_drop_reason"].to_numpy())),
        "reattributed": int(np.sum(prod["reattributed"].astype(str).to_numpy() != stored["stored_reattributed"].to_numpy())),
    }
    pd.DataFrame(trig).to_csv(OUT / "tables" / "triggers.csv", index=False)
    summary["triggers"] = trig
    # 2-4: sim jobs
    jobs = [(d, t) for d in RUNS for t in THRESHOLDS]
    with ProcessPoolExecutor(6) as pool:
        summary["sim"] = list(pool.map(sim_job, jobs))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
