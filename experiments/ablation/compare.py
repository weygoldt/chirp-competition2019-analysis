"""Compare arm A (3 windows) and arm B (long window only) from their evaluation outputs.

    uv run python experiments/ablation/compare.py ROOT --sim-a SIM_V4 --sim-b SIM_V4_LONGONLY \
        [--n-boot 2000]

ROOT is the report folder with ``{A,B}/eval_postprocess`` (eval_postprocess.py
outputs, unchanged script) and ``real_labels_365/{A,B}`` (
eval_real_labels_ablation.sh). Everything here is read from those files;
nothing is re-predicted. Writes ROOT/compare/:

- ``sim_overall.csv``: sim_v4 test after post-processing at score 0.5
  (box-midpoint matching): P, R, F1 (counts from ``summary.csv``), AP (from
  ``pr_curve.csv``), assignment accuracy and end-to-end recall; per arm with
  Wilson 95% CIs, and B - A with a paired bootstrap over the 300 test scenes
  (scenes resampled with replacement, the same draws for both arms).
- ``sim_strata.csv``: recall and assignment accuracy per stratum: chirp
  family (small/big), duration (>= 0.3 s "long"), footprint SNR quartile,
  ΔF bin. Precision has no stratum (false positives have no label).
- ``emitter_error_by_df.csv``: emitter-head error (|predicted - true EODf|) of
  the true positives per ΔF bin (point values).
- ``real.csv``: the 335 original hand labels (eval_postprocess ``real_after``)
  and the 365 labels (335 + 30 confirmed), at 0.5: Wilson CIs per arm,
  Newcombe (hybrid score) CIs of B - A for P and R, an exact McNemar test on
  the labels found by one arm only.
- ``real_fp_review.csv``: each arm's false positives on the 365 labels by
  whether the author reviewed a box there (labels/emitters: the released
  dual-head's and YOLO26s' unmatched boxes at >= 0.5 were reviewed, arm B's
  never were).
- ``tables.md``, ``summary.json``, ``differences.png``.

Sanity checks (assertions): the two datasets give the same ground truth; the
pooled counts and AP recomputed here equal the evaluation's metrics.json;
assignment counts per ΔF bin equal assignment.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from chirpdetector.data import dataset_ground_truth, load_dataset
from chirpdetector.postprocess import COLUMNS

ARMS = ("A", "B")
NAMES = {"A": "A: 3 windows (released)", "B": "B: long window x3"}
REF = 0.5
DF_ORDER = ["single", "0-10", "10-25", "25-50", "50-100", "100+"]
Z = stats.norm.ppf(0.975)


# ------------------------------------------------------------------ helpers


def wilson(k: float, n: float) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return (centre - half, centre + half)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """95% CI of p2 - p1 (Newcombe's hybrid score method, independent samples)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (lo, hi)


def scene_of(tile: str) -> str:
    return tile.rsplit("_f", 1)[0]


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def ap_from_events(score: np.ndarray, dtp: np.ndarray, dfp: np.ndarray, labels: float) -> float:
    """Step-wise AP over distinct scores (chirpdetector.evaluation.detection)."""
    order = np.argsort(-score, kind="stable")
    s, tp, fp = score[order], np.cumsum(dtp[order]), np.cumsum(dfp[order])
    last = np.r_[s[1:] != s[:-1], True]  # last event of each distinct score
    tp, fp = tp[last], fp[last]
    recall = tp / labels
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    return float(np.nansum(np.diff(np.r_[0.0, recall]) * precision))


# ---------------------------------------------------------------- sim arm


class SimArm:
    def __init__(self, eval_dir: Path, scenes: list[str], truth_labels: pd.DataFrame):
        after = eval_dir / "sim_test_after"
        self.metrics = json.loads((after / "metrics.json").read_text())["metrics"]
        self.items = pd.read_csv(after / "items.csv")
        summary = pd.read_csv(after / "summary.csv")
        curve = pd.read_csv(after / "pr_curve.csv", dtype={"dataset": str, "recording": str})
        self.assign_csv = pd.read_csv(after / "assignment.csv")
        flagged = pd.read_parquet(eval_dir / "sim_test_predictions_flagged.parquet")
        keep = flagged["drop_reason"].astype(str).to_numpy() == ""
        self.kept = flagged.loc[keep].drop(columns=[c for c in COLUMNS if c in flagged]).reset_index(drop=True)
        self.scene_index = {s: k for k, s in enumerate(scenes)}
        n = len(scenes)

        # per-scene counts at the reference threshold
        rec = summary[(summary["level"] == "recording") & np.isclose(summary["threshold"], REF)]
        rec = rec.assign(scene=rec["recording"].map(scene_of))
        per = rec.groupby("scene")[["labels", "tp", "fp", "fn"]].sum()
        self.counts = {c: np.zeros(n) for c in ("labels", "tp", "fp", "fn")}
        for s, row in per.iterrows():
            for c in self.counts:
                self.counts[c][self.scene_index[s]] = row[c]
        det = self.metrics["detection"]["datasets"]["hybrid"]
        for c in ("labels", "tp", "fp", "fn"):
            assert int(self.counts[c].sum()) == int(det[c]), (c, self.counts[c].sum(), det[c])

        # AP events: per-tile cumulative counts at each distinct score -> increments
        rc = curve[curve["level"] == "recording"].copy()
        rc = rc.sort_values(["recording", "threshold"], ascending=[True, False])
        g = rc.groupby("recording", sort=False)
        rc["dtp"] = g["tp"].diff().fillna(rc["tp"])
        rc["dfp"] = g["fp"].diff().fillna(rc["fp"])
        # a tile's fp at the first score counts unmatched predictions; the
        # increments of fp+tp are the predictions not ignored
        self.ev_score = rc["threshold"].to_numpy(float)
        self.ev_tp = rc["dtp"].to_numpy(float)
        self.ev_fp = rc["dfp"].to_numpy(float)
        self.ev_scene = rc["recording"].map(scene_of).map(self.scene_index).to_numpy(int)
        self.labels_total = float(self.counts["labels"].sum())
        ap = ap_from_events(self.ev_score, self.ev_tp, self.ev_fp, self.labels_total)
        assert abs(ap - det["average_precision"]) < 1e-9, (ap, det["average_precision"])
        self.ap = ap

        # label-level outcomes with ground-truth metadata
        it = self.items
        lab = it[(it["label"] == "chirp") & it["outcome"].isin(["tp", "fn"])].copy()
        lab["label_row"] = lab["label_row"].astype(int)
        meta = truth_labels.iloc[lab["label_row"].to_numpy()]
        assert np.allclose(lab["label_t_center_s"].to_numpy(float), meta["t_center_s"].to_numpy(float))
        for c in ("family", "duration_s", "snr_db", "emitter_hz", "excursion_hz"):
            lab[c] = meta[c].to_numpy()
        lab["scene"] = lab["recording"].map(scene_of).map(self.scene_index)
        lab["found"] = (lab["outcome"] == "tp").astype(float)
        pred = lab["pred_row"].fillna(-1).astype(int).to_numpy()
        lab["correct"] = np.where(
            lab["outcome"] == "tp",
            (lab["pred_emitter_id"].astype("Int64").fillna(-1).to_numpy() == lab["true_emitter_id"].astype("Int64").fillna(-2).to_numpy()).astype(float),
            np.nan,
        )
        lab["pred_hz"] = np.where(pred >= 0, self.kept["emitter_hz"].to_numpy(float)[np.clip(pred, 0, None)], np.nan)
        lab["error_hz"] = np.abs(lab["pred_hz"] - lab["emitter_hz"].astype(float))
        self.labels = lab
        # assignment check against the evaluation's table
        tp = lab[lab["outcome"] == "tp"]
        for row in self.assign_csv.itertuples(index=False):
            sel = tp if row.df_bin == "all" else tp[tp["df_bin"] == row.df_bin]
            assert len(sel) == row.n and int(sel["correct"].sum()) == row.n_correct, (row.df_bin, len(sel), row.n)

    def scene_sums(self, mask: np.ndarray, column: str) -> np.ndarray:
        lab = self.labels[mask]
        return np.bincount(lab["scene"].to_numpy(int), weights=lab[column].to_numpy(float), minlength=len(self.scene_index))


# ------------------------------------------------------------------- main


def sim_section(arms: dict[str, SimArm], n_boot: int, seed: int = 0):
    n = len(next(iter(arms.values())).scene_index)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    W = np.stack([np.bincount(d, minlength=n) for d in draws]).astype(float)  # (B, S)

    def boot_ratio(num: dict, den: dict) -> dict:
        out = {}
        for a in ARMS:
            out[a] = ratio(W @ num[a], W @ den[a])
        return out

    def row(metric: str, stratum: str, k: dict, n_: dict, boot: dict) -> dict:
        r = {"metric": metric, "stratum": stratum}
        for a in ARMS:
            kk, nn = float(k[a].sum()), float(n_[a].sum())
            r[f"{a}_k"], r[f"{a}_n"] = int(kk), int(nn)
            r[f"{a}"] = kk / nn if nn else math.nan
            r[f"{a}_lo"], r[f"{a}_hi"] = wilson(kk, nn)
        d = boot["B"] - boot["A"]
        r["diff"] = r["B"] - r["A"]
        r["diff_lo"], r["diff_hi"] = np.nanpercentile(d, [2.5, 97.5])
        r["p_boot_B_gt_A"] = float(np.mean(d > 0))
        return r

    rows = []
    # overall detection
    c = {a: arms[a].counts for a in ARMS}
    tp = {a: c[a]["tp"] for a in ARMS}
    rows.append(row("precision", "all", tp, {a: c[a]["tp"] + c[a]["fp"] for a in ARMS}, boot_ratio(tp, {a: c[a]["tp"] + c[a]["fp"] for a in ARMS})))
    rows.append(row("recall", "all", tp, {a: c[a]["labels"] for a in ARMS}, boot_ratio(tp, {a: c[a]["labels"] for a in ARMS})))
    f1num = {a: 2 * c[a]["tp"] for a in ARMS}
    f1den = {a: 2 * c[a]["tp"] + c[a]["fp"] + c[a]["fn"] for a in ARMS}
    r = row("f1", "all", f1num, f1den, boot_ratio(f1num, f1den))
    for a in ARMS:  # Wilson does not apply to F1
        r[f"{a}_lo"] = r[f"{a}_hi"] = math.nan
        r[f"{a}_k"] = r[f"{a}_n"] = None
    rows.append(r)
    # AP, bootstrap over scenes
    ap_boot = {}
    for a in ARMS:
        arm = arms[a]
        vals = np.empty(n_boot)
        for b in range(n_boot):
            w = W[b, arm.ev_scene]
            vals[b] = ap_from_events(arm.ev_score, w * arm.ev_tp, w * arm.ev_fp, float(W[b] @ arm.counts["labels"]))
        ap_boot[a] = vals
    d = ap_boot["B"] - ap_boot["A"]
    r = {"metric": "average_precision", "stratum": "all"}
    for a in ARMS:
        r[a] = arms[a].ap
        r[f"{a}_lo"], r[f"{a}_hi"] = np.percentile(ap_boot[a], [2.5, 97.5])
        r[f"{a}_k"] = r[f"{a}_n"] = None
    r["diff"] = r["B"] - r["A"]
    r["diff_lo"], r["diff_hi"] = np.percentile(d, [2.5, 97.5])
    r["p_boot_B_gt_A"] = float(np.mean(d > 0))
    rows.append(r)
    # assignment and end-to-end, overall
    lab = {a: arms[a].labels for a in ARMS}
    tpm = {a: (lab[a]["outcome"] == "tp").to_numpy() for a in ARMS}
    corr = {a: arms[a].scene_sums(tpm[a], "correct") for a in ARMS}
    ntp = {a: arms[a].scene_sums(tpm[a], "found") for a in ARMS}
    rows.append(row("assignment_accuracy", "all", corr, ntp, boot_ratio(corr, ntp)))
    alll = {a: np.ones(len(lab[a]), bool) for a in ARMS}
    nlab = {a: np.bincount(lab[a]["scene"].to_numpy(int), minlength=n).astype(float) for a in ARMS}
    rows.append(row("end_to_end_recall", "all", corr, nlab, boot_ratio(corr, nlab)))
    overall = pd.DataFrame(rows)

    # strata
    strata = []
    snr_edges = np.nanquantile(lab["A"]["snr_db"].to_numpy(float), [0, 0.25, 0.5, 0.75, 1.0])

    def masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
        fam = df["family"].astype(str).to_numpy()
        dur = df["duration_s"].to_numpy(float)
        q = np.clip(np.searchsorted(snr_edges, df["snr_db"].to_numpy(float), side="right") - 1, 0, 3)
        m = {
            "family: small": fam == "small",
            "family: big": fam == "big",
            "big, duration < 0.3 s": (fam == "big") & (dur < 0.3),
            "long (duration >= 0.3 s)": dur >= 0.3,
        }
        for k in range(4):
            m[f"SNR Q{k + 1} [{snr_edges[k]:.1f}, {snr_edges[k + 1]:.1f}] dB"] = q == k
        for b in DF_ORDER:
            m[f"ΔF {b} Hz" if b != "single" else "ΔF single fish"] = df["df_bin"].astype(str).to_numpy() == b
        return m

    mk = {a: masks(lab[a]) for a in ARMS}
    for name in mk["A"]:
        # recall
        sel = {a: mk[a][name] for a in ARMS}
        found = {a: arms[a].scene_sums(sel[a], "found") for a in ARMS}
        total = {a: np.bincount(lab[a].loc[sel[a], "scene"].to_numpy(int), minlength=n).astype(float) for a in ARMS}
        strata.append(row("recall", name, found, total, boot_ratio(found, total)))
        # assignment accuracy of the true positives
        sel2 = {a: sel[a] & tpm[a] for a in ARMS}
        corr_s = {a: arms[a].scene_sums(sel2[a], "correct") for a in ARMS}
        ntp_s = {a: arms[a].scene_sums(sel2[a], "found") for a in ARMS}
        strata.append(row("assignment_accuracy", name, corr_s, ntp_s, boot_ratio(corr_s, ntp_s)))
        strata.append(row("end_to_end_recall", name, corr_s, total, boot_ratio(corr_s, total)))
    strata = pd.DataFrame(strata)

    # emitter error by ΔF bin (true positives)
    err = []
    for a in ARMS:
        t = lab[a][tpm[a]]
        for b in ["all", *DF_ORDER]:
            s = t if b == "all" else t[t["df_bin"].astype(str) == b]
            err.append(
                {
                    "arm": a,
                    "df_bin": b,
                    "n_tp": len(s),
                    "emitter_mae_hz": float(s["error_hz"].mean()),
                    "emitter_median_hz": float(s["error_hz"].median()),
                    "emitter_p90_hz": float(s["error_hz"].quantile(0.9)) if len(s) else math.nan,
                }
            )
    return overall, strata, pd.DataFrame(err), snr_edges


def real_section(root: Path, emitters_dir: Path):
    rows, mcnemar, review = [], [], []
    sets = {
        "335 hand labels": {a: root / a / "eval_postprocess" / "real_after" for a in ARMS},
        "365 labels (335 + 30 confirmed)": {a: root / "real_labels_365" / a for a in ARMS},
    }
    for set_name, dirs in sets.items():
        m = {a: json.loads((dirs[a] / "metrics.json").read_text())["metrics"]["detection"] for a in ARMS}
        items = {a: pd.read_csv(dirs[a] / "items.csv") for a in ARMS}
        pooled = {}
        for a in ARMS:
            # pooled over the recordings' datasets
            ds = m[a]["datasets"]
            tp = sum(v["tp"] for v in ds.values())
            fp = sum(v["fp"] for v in ds.values())
            fn = sum(v["fn"] for v in ds.values())
            labels = sum(v["labels"] for v in ds.values())
            it = items[a]
            assert int((it["outcome"] == "tp").sum()) == tp and int((it["outcome"] == "fn").sum()) == fn
            pooled[a] = {"tp": tp, "fp": fp, "fn": fn, "labels": labels}
        for metric, num, den in (("precision", "tp", None), ("recall", "tp", "labels")):
            r = {"label_set": set_name, "metric": metric}
            for a in ARMS:
                k = pooled[a]["tp"]
                nn = pooled[a]["tp"] + pooled[a]["fp"] if den is None else pooled[a][den]
                r[f"{a}_k"], r[f"{a}_n"], r[a] = k, nn, k / nn
                r[f"{a}_lo"], r[f"{a}_hi"] = wilson(k, nn)
            r["diff"] = r["B"] - r["A"]
            r["diff_lo"], r["diff_hi"] = newcombe(r["A_k"], r["A_n"], r["B_k"], r["B_n"])
            rows.append(r)
        r = {"label_set": set_name, "metric": "f1"}
        for a in ARMS:
            p = pooled[a]
            r[a] = 2 * p["tp"] / (2 * p["tp"] + p["fp"] + p["fn"])
            r[f"{a}_k"], r[f"{a}_n"] = p["fp"], p["fn"]  # (fp, fn) for the table
        r["diff"] = r["B"] - r["A"]
        rows.append(r)
        # paired recall: labels found by one arm only
        f = {}
        for a in ARMS:
            it = items[a]
            sel = it[it["outcome"].isin(["tp", "fn"])]
            f[a] = sel.set_index(sel["label_row"].astype(int))["outcome"].eq("tp")
        idx = f["A"].index.intersection(f["B"].index)
        only_a = int((f["A"][idx] & ~f["B"][idx]).sum())
        only_b = int((~f["A"][idx] & f["B"][idx]).sum())
        p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        mcnemar.append({"label_set": set_name, "labels": len(idx), "found_by_A_only": only_a, "found_by_B_only": only_b, "exact_mcnemar_p": p})
        # false positives at >= 0.5 vs the author's reviewed boxes (365 set only)
        if set_name.startswith("365"):
            reviewed = pd.concat(
                [pd.read_csv(p).assign(recording=p.stem) for p in sorted(emitters_dir.glob("*.csv"))], ignore_index=True
            )
            rc = 0.5 * (reviewed["t_start_s"] + reviewed["t_end_s"])
            for a in ARMS:
                it = items[a]
                fp = it[(it["outcome"] == "fp") & (it["score"] >= REF)]
                preds = pd.read_parquet(dirs[a] / "predictions_kept.parquet")
                counts: dict[str, int] = {}
                for prow in fp["pred_row"].astype(int):
                    pr = preds.iloc[prow]
                    c = 0.5 * (pr["t_start_s"] + pr["t_end_s"])
                    hit = reviewed[
                        (reviewed["recording"] == pr["recording"])
                        & ((rc - c).abs() <= 0.1)
                        & (reviewed["f_low_hz"] < pr["f_high_hz"])
                        & (pr["f_low_hz"] < reviewed["f_high_hz"])
                    ]
                    key = "unreviewed" if hit.empty else "reviewed: " + "/".join(sorted(set(hit["label"].astype(str))))
                    counts[key] = counts.get(key, 0) + 1
                for key, v in sorted(counts.items()):
                    review.append({"arm": a, "fp_at_0.5": len(fp), "status": key, "n": v})
    return pd.DataFrame(rows), pd.DataFrame(mcnemar), pd.DataFrame(review)


def benchmark_section(root: Path, n_boot: int, seed: int = 0) -> pd.DataFrame | None:
    """Blind benchmark (optional): both arms scored as models by label-benchmark score."""
    path = root / "benchmark" / "report" / "items.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path, low_memory=False)
    d = d[d["scored"].astype(bool)]
    true = d[d["truth"] == "chirp"]
    neg = d[d["truth"] != "chirp"]
    metrics = json.loads((root / "benchmark" / "report" / "metrics.json").read_text())["point"]
    rng = np.random.default_rng(seed)
    rows = []
    for name, table, col, key in (
        ("recall on true chirps", true, "chirp", "recall"),
        ("chirp found with the right emitter", true, "emitter_ok", "emitter_recall"),
        ("false-alarm rate, all negatives", neg, "false_alarm", "false_alarm"),
    ):
        x = {a: table[f"{a}_{col}"].astype(float).to_numpy() for a in ARMS}
        r = {"metric": name, "stratum": "all"}
        for a in ARMS:
            k, n = float(x[a].sum()), len(x[a])
            r[f"{a}_k"], r[f"{a}_n"], r[a] = int(k), n, k / n
            assert abs(r[a] - metrics[f"{key}_{a}"]) < 1e-12, (name, a, r[a], metrics[f"{key}_{a}"])
            r[f"{a}_lo"], r[f"{a}_hi"] = wilson(k, n)
        idx = rng.integers(0, len(table), size=(n_boot, len(table)))
        diff = x["B"][idx].mean(axis=1) - x["A"][idx].mean(axis=1)
        r["diff"] = r["B"] - r["A"]
        r["diff_lo"], r["diff_hi"] = np.percentile(diff, [2.5, 97.5])
        r["p_boot_B_gt_A"] = float(np.mean(diff > 0))
        only_a = int(((x["A"] == 1) & (x["B"] == 0)).sum())
        only_b = int(((x["A"] == 0) & (x["B"] == 1)).sum())
        r["A_only"], r["B_only"] = only_a, only_b
        r["exact_mcnemar_p"] = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        rows.append(r)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(v, digits=1):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.{digits}f}"


def md_rows(df: pd.DataFrame, label_cols: list[str], counts: bool = True) -> str:
    head = [*label_cols, "A", "B", "B - A (95% CI)"]
    if counts:
        head.insert(len(label_cols), "A k/n")
        head.insert(len(label_cols) + 2, "B k/n")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in df.to_dict(orient="records"):
        def arm(a):
            s = pct(r[a], 2 if r["metric"] in ("precision", "recall", "f1", "average_precision") and r.get("stratum", "") == "all" else 1)
            if not math.isnan(r.get(f"{a}_lo", math.nan) or math.nan):
                s += f" [{pct(r[f'{a}_lo'])}, {pct(r[f'{a}_hi'])}]"
            return s
        cells = [str(r[c]) for c in label_cols]
        a_cell, b_cell = arm("A"), arm("B")
        diff = f"{100 * r['diff']:+.2f}"
        if not math.isnan(r.get("diff_lo", math.nan) if r.get("diff_lo") is not None else math.nan):
            diff += f" [{100 * r['diff_lo']:+.2f}, {100 * r['diff_hi']:+.2f}]"
        if counts:
            ka = "" if r.get("A_k") is None or (isinstance(r.get("A_k"), float) and math.isnan(r["A_k"])) else f"{int(r['A_k'])}/{int(r['A_n'])}"
            kb = "" if r.get("B_k") is None or (isinstance(r.get("B_k"), float) and math.isnan(r["B_k"])) else f"{int(r['B_k'])}/{int(r['B_n'])}"
            lines.append("| " + " | ".join([*cells, ka, a_cell, kb, b_cell, diff]) + " |")
        else:
            lines.append("| " + " | ".join([*cells, a_cell, b_cell, diff]) + " |")
    return "\n".join(lines)


def figure(overall, strata, real, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, accent = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    entries = []
    names = {"precision": "precision", "recall": "recall", "f1": "F1", "average_precision": "AP", "assignment_accuracy": "assignment accuracy", "end_to_end_recall": "end-to-end recall"}
    for r in overall.to_dict(orient="records"):
        entries.append(("sim_v4 test, all chirps", names[r["metric"]], r))
    for r in strata[strata["metric"] == "recall"].to_dict(orient="records"):
        if r["stratum"].startswith(("family", "big,", "long", "SNR")):
            entries.append(("sim_v4 test, recall by stratum", r["stratum"], r))
    for r in strata[strata["metric"] == "assignment_accuracy"].to_dict(orient="records"):
        if r["stratum"].startswith("ΔF") or r["stratum"].startswith(("family", "long")):
            entries.append(("sim_v4 test, assignment accuracy", r["stratum"], r))
    for r in real[real["metric"].isin(["precision", "recall"])].to_dict(orient="records"):
        entries.append(("real recordings", f"{r['metric']}, {r['label_set'].split(' (')[0]}", r))
    fig, ax = plt.subplots(figsize=(8.6, 0.27 * len(entries) + 1.6), constrained_layout=True)
    y, ticks, labels, last = 0, [], [], None
    for group, label, r in entries:
        if group != last:
            y += 0.8 if last is not None else 0
            ax.text(-0.01, -y + 0.05, group, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, fontweight="bold", color=ink)
            y += 0.7
            last = group
        d, lo, hi = 100 * r["diff"], r.get("diff_lo"), r.get("diff_hi")
        if lo is not None and not (isinstance(lo, float) and math.isnan(lo)):
            ax.plot([100 * lo, 100 * hi], [-y, -y], color=accent, lw=2, solid_capstyle="round")
        ax.plot([d], [-y], "o", ms=6, color=accent, mec="white", mew=1.5, zorder=3)
        ticks.append(-y)
        n_a, n_b = r.get("A_n"), r.get("B_n")
        has_n = isinstance(n_a, (int, np.integer)) or (isinstance(n_a, float) and not math.isnan(n_a))
        n_txt = "" if not has_n else (f"  (n={int(n_a)})" if int(n_a) == int(n_b) else f"  (n={int(n_a)}/{int(n_b)})")
        labels.append(label + n_txt)
        y += 1
    ax.axvline(0, color=muted, lw=1)
    ax.set_yticks(ticks, labels, fontsize=8, color=ink)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("B (long window x3) minus A (3 windows, released), percentage points\n95% CI: paired bootstrap over test scenes (sim), Newcombe (real)", fontsize=8.5, color=muted)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.set_title("Front-end ablation: B - A (one training run per arm)", fontsize=10.5, loc="left", color=ink)
    ax.set_ylim(-y + 0.3, 0.8)
    fig.savefig(out, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--sim-a", type=Path, required=True)
    p.add_argument("--sim-b", type=Path, required=True)
    p.add_argument("--emitters", type=Path, default=Path("/mnt/data1/chirp_v2/labels/emitters/labels"))
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    out = args.root / "compare"
    out.mkdir(parents=True, exist_ok=True)

    da = load_dataset(args.sim_a, check_files=False)
    db = load_dataset(args.sim_b, check_files=False)
    ids = list(da.manifest.image_ids("test"))
    assert ids == list(db.manifest.image_ids("test"))
    ta = dataset_ground_truth(da, ids).labels
    tb = dataset_ground_truth(db, ids).labels
    pd.testing.assert_frame_equal(ta.reset_index(drop=True), tb.reset_index(drop=True))
    scenes = sorted({scene_of(i) for i in ids})
    arms = {a: SimArm(args.root / a / "eval_postprocess", scenes, ta) for a in ARMS}
    overall, strata, err, snr_edges = sim_section(arms, args.n_boot)
    real, mcn, review = real_section(args.root, args.emitters)

    overall.to_csv(out / "sim_overall.csv", index=False)
    strata.to_csv(out / "sim_strata.csv", index=False)
    err.to_csv(out / "emitter_error_by_df.csv", index=False)
    real.to_csv(out / "real.csv", index=False)
    mcn.to_csv(out / "real_mcnemar.csv", index=False)
    review.to_csv(out / "real_fp_review.csv", index=False)
    figure(overall, strata, real, out / "differences.png")

    md = ["## sim_v4 test (300 scenes, 8,100 tiles), after post-processing, score >= 0.5", ""]
    md.append("Values in %; per-arm CIs Wilson (AP: bootstrap), B - A in percentage points with a paired scene bootstrap (95%).")
    md += ["", md_rows(overall, ["metric"]), ""]
    for metric in ("recall", "assignment_accuracy", "end_to_end_recall"):
        md += [f"### {metric} by stratum", "", md_rows(strata[strata["metric"] == metric], ["stratum"]), ""]
    md += ["### emitter-head error of the true positives (Hz)", ""]
    piv = err.pivot(index="df_bin", columns="arm", values=["n_tp", "emitter_median_hz", "emitter_mae_hz"]).reindex(["all", *DF_ORDER])
    md.append("| ΔF bin | n A | n B | median A | median B | mean A | mean B |")
    md.append("|---|---|---|---|---|---|---|")
    for b, r in piv.iterrows():
        md.append(f"| {b} | {int(r[('n_tp', 'A')])} | {int(r[('n_tp', 'B')])} | {r[('emitter_median_hz', 'A')]:.2f} | {r[('emitter_median_hz', 'B')]:.2f} | {r[('emitter_mae_hz', 'A')]:.2f} | {r[('emitter_mae_hz', 'B')]:.2f} |")
    md += ["", "## Real recordings (4 hand-labelled recordings), after post-processing, score >= 0.5", ""]
    md.append("Per-arm CIs Wilson; B - A Newcombe hybrid score (treats the arms as independent); F1 row: k/n columns hold FP/FN.")
    md += ["", md_rows(real, ["label_set", "metric"]), ""]
    md += ["Labels found by one arm only (exact McNemar):", "", md_table(mcn), ""]
    md += ["False positives at >= 0.5 on the 365 labels vs the author's reviewed boxes:", "", md_table(review), ""]
    bench = benchmark_section(args.root, args.n_boot)
    if bench is not None:
        bench.to_csv(out / "benchmark.csv", index=False)
        md += ["## Blind benchmark (1,010 simulated items; scored true chirps and negatives), score >= 0.5", ""]
        md.append("Per-arm CIs Wilson; B - A bootstrap over items (paired). A_only / B_only: items where only that arm has the event of the row (detected, right emitter, false alarm); exact McNemar.")
        md += ["", md_rows(bench, ["metric"]), ""]
        md += [md_table(bench[["metric", "A_only", "B_only", "exact_mcnemar_p"]]), ""]
    (out / "tables.md").write_text("\n".join(md) + "\n")
    summary = {
        "n_boot": args.n_boot,
        "bootstrap": "paired, 300 test scenes resampled with replacement, seed 0",
        "snr_quartile_edges_db": snr_edges.tolist(),
        "sim_overall": overall.to_dict(orient="records"),
        "real": real.to_dict(orient="records"),
        "real_mcnemar": mcn.to_dict(orient="records"),
        "real_fp_review": review.to_dict(orient="records"),
        "benchmark": None if bench is None else bench.to_dict(orient="records"),
        "inputs": {a: str(args.root / a) for a in ARMS},
        "datasets": {"A": str(args.sim_a), "B": str(args.sim_b)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
