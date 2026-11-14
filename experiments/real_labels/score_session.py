"""Score the real-data labelling session real_v1 (run only once labels exist).

Reads the truth file (``labels/real_v1/truth/real_v1_truth.parquet``), the
labeller's label files (``labels/real_v1_labels/labels/*.csv``) and, for the
recall windows, the t001 runs' detections. Estimates:

1. **Precision by score bin and calibration** (candidates of the main
   strata, kept rows): ``P_b = sum(w y) / sum(w)`` with ``y = 1`` for
   ``chirp`` and ``chirp_emitter_unclear``, ``0`` for ``not_chirp``, and
   ``w = N_cell / n_labelled_cell`` (stratum x recording cells of the
   design; unlabelled items are taken as missing at random within their
   cell). Calibration: weighted mean score vs precision per bin.
2. **Precision at thresholds** 0.5-0.95: the same estimator over the main
   strata's items with ``score >= t`` (domain estimation within the cells),
   both datasets pooled and per dataset; ``N_hat(t) = sum(w)``.
   **By the assigned fish's trial status** (winner / loser / other, the
   strata of the bins 0.5-0.7, 0.7-0.9, 0.9-1; ``status`` in the truth):
   precision per dataset x score bin x status and at thresholds 0.5-0.9
   per status: whether the winner-assigned detections between 0.5 and 0.9
   are chirps decides the paper's winner counts.
3. **Recall** from the windows the labeller finished: every mark in the
   zone is a chirp; a mark is found at threshold ``t`` if a kept detection
   with ``score >= t`` matches it under the rule of
   ``chirpdetector.evaluation.matching`` (centres within 100 ms,
   overlapping frequency ranges, one-to-one, Hungarian); a mark's range is
   its click +-15 Hz (the snapping tolerance; marks are clicks on the
   emitter's baseline at the chirp, or on the chirp itself if the emitter
   is unclear). ``R(t) = sum_i w_i found_i(t) / sum_i w_i marks_i`` with
   the windows' design weights ``1 / pi_i``. Detections in the zone that
   match no mark give an independent (exhaustive-label) precision.
   ``recall_time_only`` drops the frequency condition (any detection
   within 100 ms): an upper bound that shows how much the frequency rule
   costs.
4. **Emitter agreement**: the labeller's emitter (track id of the
   labeller's tracks: gap-filled for the 2019 trials) vs the model's
   assigned track (t001 run = production assignment; gap-filled run for
   the 2019 trials), by ΔF of the two closest live tracks, weighted and
   unweighted; also on the matched window marks.
5. **Test-retest** on the hidden repeats: decision agreement and Cohen's
   kappa (chirp vs not), emitter agreement; for windows, marks matched
   between the two passes.
6. **Flagged rows** (post-processing): label distribution per reason; for
   duplicates whether the labeller's emitter differs from the kept twin's.

Confidence intervals: 95 % percentile bootstrap (2,000 resamples) within the
design's strata (candidates; items keep their cell weights) or datasets
(windows).

    uv run python experiments/real_labels/score_session.py [--labels DIR] [--out DIR]

Output: ``labels/real_v1/report/`` (``report.md``, ``metrics.json``, tables,
``calibration.png``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS_OUT, ROOT, RUN_2019, RUN_2022  # noqa: E402

THRESHOLDS = (0.01, 0.05, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DF_BINS = (0.0, 10.0, 25.0, 50.0, 100.0, np.inf)
DF_NAMES = ("0-10", "10-25", "25-50", "50-100", "100+")
MARK_HALF_BAND_HZ = 15.0
TIME_TOL_S = 0.1
N_BOOT = 2000
SEED = 7
POSITIVE = ("chirp", "chirp_emitter_unclear")


def read_labels(labels_dir: Path) -> pd.DataFrame:
    files = sorted((labels_dir / "labels").glob("*.csv"))
    if not files:
        msg = f"no label files in {labels_dir / 'labels'}"
        raise SystemExit(msg)
    df = pd.concat([pd.read_csv(f, dtype={"candidate_id": str, "parent_item_id": str,
                                           "item_id": str}) for f in files], ignore_index=True)
    df["parent_item_id"] = df["parent_item_id"].fillna("")
    df["labeled_at"] = pd.to_datetime(df["labeled_at"])
    return df


def boot_matrix(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """N_BOOT resamples (rows) of all items, each group resampled with
    replacement on its own (stratified bootstrap); shape (N_BOOT, n)."""
    parts = [g[rng.integers(0, len(g), size=(N_BOOT, len(g)))] for g in groups if len(g)]
    return np.hstack(parts) if parts else np.zeros((N_BOOT, 0), dtype=int)


def _boot_ratio(num: np.ndarray, den: np.ndarray, index: np.ndarray) -> tuple[float, float]:
    """Percentile CI of sum(num) / sum(den) over the resamples."""
    n, d = num[index].sum(axis=1), den[index].sum(axis=1)
    r = n[d > 0] / d[d > 0]
    if r.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def candidate_table(truth: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    box = labels[labels["parent_item_id"] == ""]
    box = box[box["label"].isin([*POSITIVE, "not_chirp"])]
    box = box.sort_values("labeled_at").groupby("candidate_id").tail(1)
    cols = ["candidate_id", "label", "emitter_id", "emitter_baseline_hz", "delta_f_hz",
            "elapsed_s", "labeled_at", "via"]
    t = truth[truth["item_kind"] == "box"].merge(box[cols], on="candidate_id", how="left")
    t["labelled"] = t["label"].notna()
    t["y"] = t["label"].isin(POSITIVE).astype(float)
    t.loc[~t["labelled"], "y"] = np.nan
    return t


def design_weights(t: pd.DataFrame) -> pd.DataFrame:
    """w = N_cell / n_labelled_cell (cells: stratum x recording)."""
    t = t.copy()
    lab = t[t["labelled"]]
    n_lab = lab.groupby(["stratum", "recording"]).size().rename("cell_n_labelled")
    t = t.merge(n_lab, left_on=["stratum", "recording"], right_index=True, how="left")
    t["w"] = np.where(t["labelled"], t["cell_N"] / t["cell_n_labelled"], 0.0)
    return t


def _ratio(num, den) -> float:  # noqa: ANN001
    d = float(np.sum(den))
    return float(np.sum(num)) / d if d > 0 else np.nan


def precision_tables(t: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = t[t["stratum"].str.startswith("main/") & t["labelled"]].reset_index(drop=True)
    # resample within strata (the recording cells often hold 1-2 items, which
    # a within-cell bootstrap would treat as fixed); items keep their weights
    cells = [g.index.to_numpy() for _, g in main.groupby("stratum")]
    index = boot_matrix(cells, rng)
    w, y, s = main["w"].to_numpy(), main["y"].to_numpy(), main["score"].to_numpy()
    rows = []
    for (ds_sel, ds_name) in ((None, "both"), ("competition2019", "2019"), ("competition2022", "2022")):
        dmask = np.ones(len(main), bool) if ds_sel is None else (main["dataset"] == ds_sel).to_numpy()
        for b, g in main[dmask].groupby("score_bin"):
            m = dmask & (main["score_bin"] == b).to_numpy()
            ci = _boot_ratio(w * y * m, w * m, index)
            rows.append({"dataset": ds_name, "score_bin": b, "n": int(m.sum()),
                         "N_hat": float(w[m].sum()), "mean_score": _ratio(w[m] * s[m], w[m]),
                         "precision": _ratio(w[m] * y[m], w[m]), "ci_low": ci[0], "ci_high": ci[1],
                         "precision_unweighted": float(np.mean(y[m]))})
            for size in ("small", "big"):
                ms = m & (main["size"] == size).to_numpy()
                if ms.any():
                    rows.append({"dataset": ds_name, "score_bin": f"{b} {size}", "n": int(ms.sum()),
                                 "N_hat": float(w[ms].sum()),
                                 "mean_score": _ratio(w[ms] * s[ms], w[ms]),
                                 "precision": _ratio(w[ms] * y[ms], w[ms]),
                                 "precision_unweighted": float(np.mean(y[ms]))})
    by_bin = pd.DataFrame(rows)
    rows = []
    for (ds_sel, ds_name) in ((None, "both"), ("competition2019", "2019"), ("competition2022", "2022")):
        dmask = np.ones(len(main), bool) if ds_sel is None else (main["dataset"] == ds_sel).to_numpy()
        for thr in THRESHOLDS:
            m = dmask & (s >= thr)
            ci = _boot_ratio(w * y * m, w * m, index)
            rows.append({"dataset": ds_name, "threshold": thr, "n": int(m.sum()),
                         "N_hat": float(w[m].sum()), "precision": _ratio(w[m] * y[m], w[m]),
                         "ci_low": ci[0], "ci_high": ci[1]})
    return by_bin, pd.DataFrame(rows)


def status_tables(t: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Precision by dataset x score bin x status and at thresholds by status."""
    main = t[t["stratum"].str.startswith("main/") & t["labelled"]].reset_index(drop=True)
    # resample within strata (the recording cells often hold 1-2 items, which
    # a within-cell bootstrap would treat as fixed); items keep their weights
    cells = [g.index.to_numpy() for _, g in main.groupby("stratum")]
    index = boot_matrix(cells, rng)
    w, y, s = main["w"].to_numpy(), main["y"].to_numpy(), main["score"].to_numpy()
    rows = []
    for ds_sel, ds_name in ((None, "both"), ("competition2019", "2019"), ("competition2022", "2022")):
        dm = np.ones(len(main), bool) if ds_sel is None else (main["dataset"] == ds_sel).to_numpy()
        for status in ("winner", "loser", "other"):
            sm = dm & (main["status"] == status).to_numpy()
            groups = [(b, (main["score_bin"] == b).to_numpy()) for b in sorted(main["score_bin"].unique())]
            groups += [(f">= {thr}", s >= thr) for thr in (0.5, 0.7, 0.9)]
            groups += [("0.5-0.9", (s >= 0.5) & (s < 0.9))]
            for name, gm in groups:
                m = sm & gm
                if not m.any():
                    continue
                ci = _boot_ratio(w * y * m, w * m, index)
                rows.append({"dataset": ds_name, "status": status, "scores": name, "n": int(m.sum()),
                             "N_hat": float(w[m].sum()), "precision": _ratio(w[m] * y[m], w[m]),
                             "ci_low": ci[0], "ci_high": ci[1],
                             "N_hat_chirps": float((w * y)[m].sum())})
    return pd.DataFrame(rows)


def window_detections(windows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    runs = {"competition2019": RUN_2019, "competition2022": RUN_2022}
    out = {}
    cols = ["t_start_s", "t_end_s", "t_center_s", "f_low_hz", "f_high_hz", "score", "kept",
            "track_id", "assign_status", "chirp_id"]
    for (ds, rec), _ in windows.groupby(["dataset", "recording"]):
        out[rec] = pd.read_parquet(runs[ds] / rec / "chirps_rtdetr.parquet", columns=cols)
    return out


def match_marks(
    marks: pd.DataFrame, det: pd.DataFrame, half_band_hz: float = MARK_HALF_BAND_HZ
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matches (mark index, detection index), evaluation rule."""
    from chirpdetector.evaluation.matching import admissible_pairs, assign

    if marks.empty or det.empty:
        return np.zeros(0, int), np.zeros(0, int)
    f = marks["clicked_hz"].to_numpy(float)
    li, pj, dt = admissible_pairs(
        marks["clicked_t_s"].to_numpy(float), (f - half_band_hz, f + half_band_hz),
        det["t_center_s"].to_numpy(float), (det["f_low_hz"].to_numpy(float), det["f_high_hz"].to_numpy(float)),
        time_tolerance_s=TIME_TOL_S, require_frequency_overlap=True)
    label_rank = np.argsort(np.argsort(marks["clicked_t_s"].to_numpy(float), kind="stable"))
    pred_rank = np.argsort(np.argsort(-det["score"].to_numpy(float), kind="stable"))
    sel = assign(li, pj, dt, label_rank, pred_rank)
    return li[sel], pj[sel]


def window_table(truth: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    win = truth[truth["item_kind"] == "window"].copy()
    done = labels[(labels["label"] == "window_done") & (labels["parent_item_id"] == "")]
    win = win.merge(done[["candidate_id", "elapsed_s"]].rename(columns={"elapsed_s": "elapsed_done_s"}),
                    on="candidate_id", how="left")
    win["done"] = win["elapsed_done_s"].notna() | win["candidate_id"].isin(done["candidate_id"])
    marks = labels[labels["parent_item_id"] != ""].copy()
    marks["window_candidate_id"] = marks["parent_item_id"].str.split("#").str[-1]
    return win, marks


def recall_tables(win: pd.DataFrame, marks: pd.DataFrame, rng: np.random.Generator) -> dict:
    base = win[win["done"] & (win["repeat_of"] == "")].reset_index(drop=True)
    dets = window_detections(base)
    per = []
    pairs_out = []
    for k, w in base.iterrows():
        mk = marks[marks["window_candidate_id"] == w["candidate_id"]].reset_index(drop=True)
        mk = mk[(mk["clicked_t_s"] >= w["zone_t0_s"]) & (mk["clicked_t_s"] <= w["zone_t1_s"])]
        mk = mk.reset_index(drop=True)
        d = dets[w["recording"]]
        d = d[d["kept"] & (d["t_center_s"] >= w["zone_t0_s"] - TIME_TOL_S)
              & (d["t_center_s"] <= w["zone_t1_s"] + TIME_TOL_S)].reset_index(drop=True)
        row = {"candidate_id": w["candidate_id"], "dataset": w["dataset"], "recording": w["recording"],
               "weight": w["weight"], "marks": len(mk)}
        in_zone = (d["t_center_s"] >= w["zone_t0_s"]) & (d["t_center_s"] <= w["zone_t1_s"])
        for thr in THRESHOLDS:
            dd = d[d["score"] >= thr].reset_index(drop=True)
            li, pj = match_marks(mk, dd)
            row[f"found_{thr}"] = len(li)
            row[f"found_time_only_{thr}"] = len(match_marks(mk, dd, half_band_hz=1e6)[0])
            z = in_zone[d["score"] >= thr].to_numpy()
            row[f"det_zone_{thr}"] = int(z.sum())
            row[f"det_zone_matched_{thr}"] = int(z[pj].sum()) if len(pj) else 0
            if thr == 0.5:
                for a, b in zip(li, pj, strict=True):
                    pairs_out.append({"candidate_id": w["candidate_id"], "dataset": w["dataset"],
                                      "weight": w["weight"],
                                      "mark_label": mk.at[a, "label"], "mark_emitter": mk.at[a, "emitter_id"],
                                      "mark_delta_f_hz": mk.at[a, "delta_f_hz"],
                                      "det_track_id": dd.at[b, "track_id"],
                                      "det_assign_status": dd.at[b, "assign_status"]})
        per.append(row)
    per = pd.DataFrame(per)
    out = {"per_window": per, "pairs": pd.DataFrame(pairs_out)}
    if per.empty:
        out["recall"] = pd.DataFrame()
        return out
    groups = [g.index.to_numpy() for _, g in per.groupby("dataset")]
    index = boot_matrix(groups, rng)
    wv, m = per["weight"].to_numpy(), per["marks"].to_numpy(float)
    rows = []
    for ds_sel, name in ((None, "both"), ("competition2019", "2019"), ("competition2022", "2022")):
        dm = np.ones(len(per), bool) if ds_sel is None else (per["dataset"] == ds_sel).to_numpy()
        for thr in THRESHOLDS:
            f = per[f"found_{thr}"].to_numpy(float)
            dz = per[f"det_zone_{thr}"].to_numpy(float)
            dzm = per[f"det_zone_matched_{thr}"].to_numpy(float)
            ci = _boot_ratio(wv * f * dm, wv * m * dm, index)
            ft = per[f"found_time_only_{thr}"].to_numpy(float)
            rows.append({"dataset": name, "threshold": thr, "windows": int(dm.sum()),
                         "marks": int(m[dm].sum()), "found": int(f[dm].sum()),
                         "recall": _ratio(wv[dm] * f[dm], wv[dm] * m[dm]), "ci_low": ci[0], "ci_high": ci[1],
                         "recall_unweighted": _ratio(f[dm], m[dm]),
                         "recall_time_only": _ratio(wv[dm] * ft[dm], wv[dm] * m[dm]),
                         "zone_detections": int(dz[dm].sum()),
                         "zone_precision": _ratio(wv[dm] * dzm[dm], wv[dm] * dz[dm])})
    out["recall"] = pd.DataFrame(rows)
    return out


def emitter_table(t: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    c = t[t["labelled"] & (t["label"] == "chirp") & t["emitter_id"].notna()
          & t["stratum"].str.startswith("main/")].copy()
    df = c["delta_f_hz"].where(c["delta_f_hz"].notna(), c["track_df_other_hz"])
    c["df_bin"] = pd.cut(df, DF_BINS, right=False, labels=DF_NAMES).astype(str)
    rows = []
    for model, col in (("t001 (= production)", "track_id"), ("gap-filled (2019 trials)", "gapfill_track_id")):
        if col not in c:
            continue
        sub = c if col == "track_id" else c[c["tracks_source"] == "gapfill"]
        for b, g in [("all", sub), *sub.groupby("df_bin")]:
            assigned = g[col].notna()
            agree = (g.loc[assigned, col].astype(float) == g.loc[assigned, "emitter_id"].astype(float))
            ww = g.loc[assigned, "w"]
            rows.append({"source": "candidates", "model": model, "df_bin": b, "n": len(g),
                         "assigned": int(assigned.sum()),
                         "agreement": float(agree.mean()) if len(agree) else np.nan,
                         "agreement_weighted": _ratio(ww * agree, ww) if len(agree) else np.nan})
    if not pairs.empty:
        p = pairs[(pairs["mark_label"] == "chirp") & pairs["mark_emitter"].notna()].copy()
        p["df_bin"] = pd.cut(p["mark_delta_f_hz"], DF_BINS, right=False, labels=DF_NAMES).astype(str)
        for b, g in [("all", p), *p.groupby("df_bin")]:
            assigned = g["det_track_id"].notna()
            agree = g.loc[assigned, "det_track_id"].astype(float) == g.loc[assigned, "mark_emitter"].astype(float)
            rows.append({"source": "window marks (t>=0.5)", "model": "t001 (= production)", "df_bin": b,
                         "n": len(g), "assigned": int(assigned.sum()),
                         "agreement": float(agree.mean()) if len(agree) else np.nan,
                         "agreement_weighted": _ratio(g.loc[assigned, "weight"] * agree, g.loc[assigned, "weight"])
                         if len(agree) else np.nan})
    return pd.DataFrame(rows)


def retest_table(t: pd.DataFrame, win: pd.DataFrame, marks: pd.DataFrame) -> dict:
    out = {}
    rep = t[t["repeat_of"] != ""]
    orig = t.set_index("candidate_id")
    rows = []
    for r in rep.itertuples():
        if r.repeat_of not in orig.index:
            continue
        o = orig.loc[r.repeat_of]
        if not (r.labelled and o["labelled"]):
            continue
        rows.append({"a": o["label"], "b": r.label, "ea": o["emitter_id"], "eb": r.emitter_id})
    p = pd.DataFrame(rows)
    if not p.empty:
        ya, yb = p["a"].isin(POSITIVE), p["b"].isin(POSITIVE)
        po = float((ya == yb).mean())
        pe = float(ya.mean() * yb.mean() + (1 - ya.mean()) * (1 - yb.mean()))
        both = (p["a"] == "chirp") & (p["b"] == "chirp") & p["ea"].notna() & p["eb"].notna()
        out["candidates"] = {"pairs": len(p), "agreement_chirp_vs_not": po,
                             "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
                             "agreement_3_labels": float((p["a"] == p["b"]).mean()),
                             "emitter_pairs": int(both.sum()),
                             "emitter_agreement": float((p.loc[both, "ea"] == p.loc[both, "eb"]).mean())
                             if both.any() else np.nan}
    wrows = []
    for r in win[win["repeat_of"] != ""].itertuples():
        o = win[win["candidate_id"] == r.repeat_of]
        if o.empty or not (r.done and o["done"].iloc[0]):
            continue
        ma = marks[marks["window_candidate_id"] == r.repeat_of].reset_index(drop=True)
        mb = marks[marks["window_candidate_id"] == r.candidate_id].reset_index(drop=True)
        fb = mb.rename(columns={"clicked_t_s": "t_center_s"}).assign(
            f_low_hz=mb["clicked_hz"] - MARK_HALF_BAND_HZ, f_high_hz=mb["clicked_hz"] + MARK_HALF_BAND_HZ,
            score=1.0)
        li, pj = match_marks(ma, fb)
        same = [ma.at[a, "emitter_id"] == mb.at[b, "emitter_id"] for a, b in zip(li, pj, strict=True)]
        wrows.append({"marks_first": len(ma), "marks_second": len(mb), "matched": len(li),
                      "matched_same_emitter": int(np.sum(same))})
    if wrows:
        w = pd.DataFrame(wrows)
        out["windows"] = {"pairs": len(w), **{k: int(w[k].sum()) for k in w.columns}}
    return out


def flagged_table(t: pd.DataFrame) -> pd.DataFrame:
    f = t[t["stratum"].str.startswith("flagged/") & t["labelled"]].copy()
    if f.empty:
        return pd.DataFrame()
    f["reason"] = f["stratum"].str.split("/").str[-1]
    tab = f.groupby(["reason", "label"]).size().unstack(fill_value=0)
    return tab.reset_index()


def _md(df: pd.DataFrame) -> str:
    """A pipe table (no tabulate in the environment)."""
    if df.empty:
        return "-"

    def cell(v: object) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def calibration_plot(by_bin: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.7", lw=1)
    for name, g in by_bin[~by_bin["score_bin"].str.contains(" ")].groupby("dataset"):
        ax.errorbar(g["mean_score"], g["precision"],
                    yerr=[g["precision"] - g["ci_low"], g["ci_high"] - g["precision"]],
                    marker="o", capsize=3, label=name)
    ax.set(xlabel="mean detector score in bin (weighted)", ylabel="precision (labeller: chirp)",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, default=LABELS_OUT)
    ap.add_argument("--truth", type=Path, default=ROOT / "truth" / "real_v1_truth.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "report")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    truth = pd.read_parquet(args.truth)
    labels = read_labels(args.labels)
    t = design_weights(candidate_table(truth, labels))
    by_bin, at_thr = precision_tables(t, rng)
    by_status = status_tables(t, rng)
    win, marks = window_table(truth, labels)
    rec = recall_tables(win, marks, rng)
    emit = emitter_table(t, rec["pairs"])
    retest = retest_table(t, win, marks)
    flagged = flagged_table(t)
    args.out.mkdir(parents=True, exist_ok=True)
    by_bin.to_csv(args.out / "precision_by_bin.csv", index=False)
    at_thr.to_csv(args.out / "precision_at_threshold.csv", index=False)
    by_status.to_csv(args.out / "precision_by_status.csv", index=False)
    rec["recall"].to_csv(args.out / "recall.csv", index=False)
    rec["per_window"].to_csv(args.out / "windows.csv", index=False)
    emit.to_csv(args.out / "emitter_agreement.csv", index=False)
    flagged.to_csv(args.out / "flagged.csv", index=False)
    t.to_csv(args.out / "candidates_scored.csv", index=False)
    if len(by_bin):
        calibration_plot(by_bin, args.out / "calibration.png")
    progress = {
        "candidates_labelled": int(t["labelled"].sum()), "candidates_total": len(t),
        "windows_done": int(win["done"].sum()), "windows_total": len(win),
        "marks": int(len(marks)),
        "median_elapsed_s_candidates": float(t["elapsed_s"].median()) if t["labelled"].any() else None,
        "contrast_toggle": "none (the labeller has no contrast control)",
    }
    metrics = {"progress": progress, "retest": retest}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    md = ["# real_v1: detector on real data, scored against the labeller", "",
          f"Progress: {json.dumps(progress)}", "",
          "## Precision by score bin (main strata, weighted)", "", _md(by_bin),
          "", "## Precision at thresholds", "",
          _md(at_thr),
          "", "## Precision by the assigned fish's trial status", "",
          "N_hat_chirps: estimated number of real chirps among the frame's detections of that cell.", "",
          _md(by_status),
          "", "## Recall (windows)", "", _md(rec["recall"])
          if len(rec["recall"]) else "(no finished windows)",
          "", "## Emitter agreement", "", _md(emit) if len(emit) else "-",
          "", "## Test-retest", "", "```", json.dumps(retest, indent=1, default=float), "```",
          "", "## Flagged rows", "", _md(flagged) if len(flagged) else "-", ""]
    (args.out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
