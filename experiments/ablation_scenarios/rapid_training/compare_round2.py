"""Compare the front ends of round 2 across seeds (run by run_round2.sh compare).

    python experiments/ablation_scenarios/rapid_training/compare_round2.py [--root R]

Per model (``R/runs/<arm>_s<seed>``) from its evaluations (run_round2.sh eval):

- breakdown benchmark (``R/bench/<name>/after_tol{025,100}/items.csv``; labels'
  sweep, interval and ΔF from ``scenarios/bench/analysis_A_Bc16/labels.parquet``):
  recall for chirps < 125 ms, 125-150 ms and >= 150 ms from the fish's nearest
  chirp (matching within 25 ms), correct emitter of the model's own true
  positives overall and below 4 Hz (100 ms), at score 0.5 and 0.8 (the
  operating threshold; true positives at 0.5 with a score >= 0.8, as Fig. 10);
- simulated test set (sim_v4 test = train_v4 test, ``R/eval/<name>/sim_test_after``):
  precision, recall, correct emitter at 0.5 and 0.8;
- 365 real labels (``R/real_labels_365/<name>/summary.csv``): precision and
  recall at 0.5 and 0.8;
- the real chirp train of Fig. S4C (``R/s4c/<name>.json``): chirps matched.

Writes ``R/compare/per_model.csv`` (one row per model and metric), ``summary.csv``
(per arm: mean, SD, n seeds; single minus three windows: difference of means
with a Welch 95 % CI over seeds) and ``tables.md``. Round 1's models (the
released detector, the single-window detector at epoch 16, both trained on
train_v4 with seed 0) are added as reference rows where their outputs exist.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

ROOT = DATA / "runs/ablation_frontend/round2"
ROUND1 = DATA / "runs/ablation_frontend"
BENCH_LABELS = ROUND1 / "scenarios/bench/analysis_A_Bc16/labels.parquet"
THRESHOLDS = {"t05": 0.5, "t08": 0.8}
ARM_NAMES = {"A": "three windows", "B": "single window"}


def items(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["outcome", "label", "label_row", "score", "true_emitter_id", "pred_emitter_id"])


def bench_metrics(bench_dir: Path, meta: pd.DataFrame) -> dict:
    out = {}
    i25, i100 = items(bench_dir / "after_tol025/items.csv"), items(bench_dir / "after_tol100/items.csv")
    for tag, thr in THRESHOLDS.items():
        lab = i25[(i25.label == "chirp") & i25.outcome.isin(["tp", "fn"])].set_index("label_row")
        lab = lab.loc[meta.index]
        found = (lab.outcome == "tp").to_numpy() & (lab.score.fillna(-1).to_numpy() >= thr)
        rate = (meta.sweep == "rate").to_numpy()
        dt = meta.dt_same_s.to_numpy()
        for key, lo, hi in (("below_125_ms", 0, 0.125), ("125_to_150_ms", 0.125, 0.15), ("150_ms_up", 0.15, np.inf)):
            m = rate & (dt >= lo) & (dt < hi)
            out[f"bench_recall_{key}_{tag}"] = float(found[m].mean())
        lab = i100[(i100.label == "chirp") & i100.outcome.isin(["tp", "fn"])].set_index("label_row").loc[meta.index]
        tp = (lab.outcome == "tp").to_numpy() & (lab.score.fillna(-1).to_numpy() >= thr)
        correct = tp & (lab.pred_emitter_id.astype("Int64").fillna(-1).to_numpy()
                        == lab.true_emitter_id.astype("Int64").fillna(-2).to_numpy())
        df_sweep = (meta.sweep == "df").to_numpy()
        df = meta.emitter_df_hz.to_numpy()
        for key, m in (("all", df_sweep), ("below_4_hz", df_sweep & (df < 4.0))):
            out[f"bench_assign_{key}_{tag}"] = float(correct[m & tp].sum() / max(tp[m].sum(), 1))
    return out


def sim_metrics(eval_dir: Path) -> dict:
    it = items(eval_dir / "sim_test_after/items.csv")
    out = {}
    lab = it[(it.label == "chirp") & it.outcome.isin(["tp", "fn"])]
    for tag, thr in THRESHOLDS.items():
        tp = (lab.outcome == "tp") & (lab.score >= thr)
        fp = ((it.outcome == "fp") & (it.score >= thr)).sum()
        correct = tp & (lab.pred_emitter_id.astype("Int64").fillna(-1) == lab.true_emitter_id.astype("Int64").fillna(-2))
        out[f"sim_recall_{tag}"] = float(tp.mean())
        out[f"sim_precision_{tag}"] = float(tp.sum() / (tp.sum() + fp))
        out[f"sim_assign_{tag}"] = float(correct.sum() / tp.sum())
    return out


def real_metrics(summary_csv: Path) -> dict:
    s = pd.read_csv(summary_csv)
    if "level" in s.columns:
        s = s[s.level == "all"]
    out = {}
    for tag, thr in THRESHOLDS.items():
        row = s.iloc[(s.threshold - thr).abs().argmin()]
        out[f"real365_precision_{tag}"] = float(row.precision)
        out[f"real365_recall_{tag}"] = float(row.recall)
    return out


def s4c_metrics(path: Path) -> dict:
    j = json.loads(path.read_text())
    return {"s4c_matched": float(j["n_matched"]), "s4c_fraction_matched": float(j["fraction_matched"])}


def model_rows(name: str, arm: str, seed: str, paths: dict, meta: pd.DataFrame) -> list[dict]:
    values: dict = {}
    if paths.get("bench") and (paths["bench"] / "after_tol100/items.csv").exists():
        values.update(bench_metrics(paths["bench"], meta))
    if paths.get("eval") and (paths["eval"] / "sim_test_after/items.csv").exists():
        values.update(sim_metrics(paths["eval"]))
    if paths.get("real") and paths["real"].exists():
        values.update(real_metrics(paths["real"]))
    if paths.get("s4c") and paths["s4c"].exists():
        values.update(s4c_metrics(paths["s4c"]))
    return [{"model": name, "arm": arm, "seed": seed, "metric": k, "value": v} for k, v in values.items()]


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Difference of means b - a with a Welch 95 % CI (NaN with < 2 per arm)."""
    d = b.mean() - a.mean()
    if len(a) < 2 or len(b) < 2:
        return d, math.nan, math.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return d, d, d
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    q = stats.t.ppf(0.975, dof)
    return d, d - q * se, d + q * se


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    R = args.root
    meta = pd.read_parquet(BENCH_LABELS)
    rows = []
    for run in sorted((R / "runs").glob("*_s*")):
        m = re.fullmatch(r"([AB])_s(\d+)", run.name)
        if not m or not (run / "DONE").exists():
            continue
        name = run.name
        rows += model_rows(name, m.group(1), m.group(2), {
            "bench": R / "bench" / name, "eval": R / "eval" / name,
            "real": R / "real_labels_365" / name / "summary.csv", "s4c": R / "s4c" / f"{name}.json"}, meta)
    reference = {
        "round1_A_released": ("A", {"bench": ROUND1 / "scenarios/bench/A", "eval": ROUND1 / "A/eval_postprocess",
                                    "real": ROUND1 / "real_labels_365/A/summary.csv"}),
        "round1_B_epoch16": ("B", {"bench": ROUND1 / "scenarios/bench/Bc16"}),
    }
    for name, (arm, paths) in reference.items():
        rows += model_rows(name, arm, "reference", paths, meta)
    per = pd.DataFrame(rows)
    out = R / "compare"
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_model.csv", index=False)

    seeds = per[per.seed != "reference"]
    summary = []
    for metric in sorted(seeds.metric.unique()):
        s = seeds[seeds.metric == metric]
        a = s[s.arm == "A"].value.to_numpy(float)
        b = s[s.arm == "B"].value.to_numpy(float)
        d, lo, hi = welch(a, b) if len(a) and len(b) else (math.nan,) * 3
        summary.append({"metric": metric,
                        "A_mean": a.mean() if len(a) else math.nan, "A_sd": a.std(ddof=1) if len(a) > 1 else math.nan,
                        "A_n": len(a),
                        "B_mean": b.mean() if len(b) else math.nan, "B_sd": b.std(ddof=1) if len(b) > 1 else math.nan,
                        "B_n": len(b), "B_minus_A": d, "B_minus_A_lo": lo, "B_minus_A_hi": hi})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    def f(x, pct=True):
        return "" if not np.isfinite(x) else (f"{100 * x:.1f}" if pct else f"{x:.0f}")

    lines = ["# Front-end ablation round 2: three windows (A) vs single window (B)", "",
             "Trained on train_v4r (train_v4 + rapid-chirping scenes), 30 epochs, no early stopping, "
             "best epoch by validation. Values in % (S4C: chirps matched of 109). B - A: difference of "
             "means over seeds, Welch 95 % CI.", "",
             "| metric | A mean ± SD (n) | B mean ± SD (n) | B - A [95 % CI] |", "|---|---|---|---|"]
    for r in summary.itertuples():
        pct = not r.metric.startswith("s4c_matched")
        lines.append(f"| {r.metric} | {f(r.A_mean, pct)} ± {f(r.A_sd, pct)} ({r.A_n}) | "
                     f"{f(r.B_mean, pct)} ± {f(r.B_sd, pct)} ({r.B_n}) | "
                     f"{f(r.B_minus_A, pct)} [{f(r.B_minus_A_lo, pct)}, {f(r.B_minus_A_hi, pct)}] |")
    wide = per.pivot_table(index="metric", columns="model", values="value")
    lines += ["", "## Per model", "", "| metric | " + " | ".join(wide.columns) + " |",
              "|---|" + "---|" * len(wide.columns)]
    for metric, row in wide.iterrows():
        cells = ["" if not np.isfinite(v) else (f"{100 * v:.1f}" if abs(v) <= 1 else f"{v:.0f}") for v in row]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
