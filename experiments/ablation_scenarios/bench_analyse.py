"""Score the breakdown benchmark: arm A (3 windows) vs arm B (long window x3).

    uv run python experiments/ablation_scenarios/bench_analyse.py [--arms A B] [--n-boot 4000]

Reads ``$ROOT/bench/<arm>/`` (bench_predict.py on the arm's benchmark
dataset), the two datasets' ground truth (identical boxes, asserted) and the
scenes' chirp tables. One row per chirp label (``labels.parquet``) with:

- the sweep and design level (scene name),
- ``emitter_df_hz``: the emitter's ΔF to the nearest other live fish at the
  chirp centre (``min_df_live_hz``; simulated and real background fish),
- ``dt_same_s``: centre-to-centre interval to the nearest other chirp of the
  same fish (every scheduled chirp, visible or not, in the whole scene),
- per arm and stage (``before`` / ``after`` post-processing) and matching
  tolerance (100 / 25 ms): found, emitter correct, emitter error (Hz).

Tables (score >= 0.5): sweep (a) by emitter ΔF bin and by design level:
assignment accuracy (labels both arms found, and per arm on its own true
positives), end-to-end recall, recall, median emitter error; sweep (b) by
same-fish interval bin and by design level: recall, precision (by design
level; false positives have no interval), assignment, and what
post-processing removed. B - A with a paired bootstrap over scenes (the same
draws for every bin). Writes ``tables.md``, ``*.csv`` and the figure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

from chirpdetector.data import dataset_ground_truth, load_dataset  # noqa: E402

ROOT = DATA / "runs/ablation_frontend/scenarios"
SCENES = DATA / "scenes/ablation_bench_v1"
DATASETS = {"A": DATA / "datasets/ablation_bench_v1_A", "B": DATA / "datasets/ablation_bench_v1_B"}
REF = 0.5
Z = stats.norm.ppf(0.975)
DF_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, math.inf]
DT_EDGES = [0.0, 0.08, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0, math.inf]


def names(edges, unit: str, scale: float = 1.0) -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        a = f"{lo * scale:g}"
        out.append(f">= {a} {unit}" if math.isinf(hi) else f"{a}-{hi * scale:g} {unit}")
    return out


DF_BINS = names(DF_EDGES, "Hz")
DT_BINS = names(DT_EDGES, "ms", 1000.0)


def wilson(k: float, n: float) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return (centre - half, centre + half)


def scene_of(tile: str) -> str:
    return tile.rsplit("_f", 1)[0]


def scene_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per scene: sweep, level; per chirp: same-fish interval."""
    meta, chirps = [], []
    for d in sorted(p for p in SCENES.iterdir() if (p / "scene.json").is_file()):
        name = json.loads((d / "scene.json").read_text())["name"]
        level = name.removeprefix("ablation_bench_v1_")
        sweep, value = level.split("_", 1)
        level_value = float(value.replace("p", ".")) if sweep == "df" else float(value.removesuffix("ms")) / 1000.0
        group = f"{name}-{d.name}"
        meta.append({"scene": group, "sweep": sweep, "level": level, "level_value": level_value})
        c = pd.read_parquet(d / "chirps.parquet", columns=["chirp_id", "emitter_id", "t_center_s"])
        c = c.sort_values(["emitter_id", "t_center_s"])
        t = c["t_center_s"].to_numpy()
        same = c["emitter_id"].to_numpy()
        prev = np.r_[np.inf, np.where(same[1:] == same[:-1], np.diff(t), np.inf)]
        nxt = np.r_[np.where(same[1:] == same[:-1], np.diff(t), np.inf), np.inf]
        chirps.append(pd.DataFrame({"scene": group, "chirp_id": c["chirp_id"].to_numpy(), "dt_same_s": np.minimum(prev, nxt)}))
    return pd.DataFrame(meta), pd.concat(chirps, ignore_index=True)


def arm_outcomes(arm: str, stage: str, tol: str, n_labels: int) -> pd.DataFrame:
    base = ROOT / "bench" / arm
    items = pd.read_csv(base / f"{stage}_tol{tol}/items.csv")
    flagged = pd.read_parquet(base / "predictions_flagged.parquet")
    rows = flagged if stage == "before" else flagged[flagged["drop_reason"].astype(str) == ""]
    rows = rows.reset_index(drop=True)
    lab = items[(items["label"] == "chirp") & items["outcome"].isin(["tp", "fn"])]
    out = pd.DataFrame(index=pd.RangeIndex(n_labels), columns=["found", "correct", "pred_hz"])
    out["found"] = False
    out["correct"] = False
    out["pred_hz"] = np.nan
    lr = lab["label_row"].astype(int).to_numpy()
    found = (lab["outcome"] == "tp").to_numpy()
    pred = lab["pred_row"].fillna(-1).astype(int).to_numpy()
    tid = lab["true_emitter_id"].astype("Int64").fillna(-2).to_numpy()
    pid = lab["pred_emitter_id"].astype("Int64").fillna(-1).to_numpy()
    out.loc[lr, "found"] = found
    out.loc[lr, "correct"] = found & (pid == tid)
    out.loc[lr, "pred_hz"] = np.where(pred >= 0, rows["emitter_hz"].to_numpy(float)[np.clip(pred, 0, None)], np.nan)
    # false positives at the reference threshold, per tile
    fp = items[(items["outcome"] == "fp") & (items["score"] >= REF)]
    out.attrs["fp_per_tile"] = fp.groupby("recording").size()
    return out


class Boot:
    """Paired scene bootstrap of ratios (the same draws for every quantity)."""

    def __init__(self, scenes: np.ndarray, n_boot: int, seed: int = 0):
        self.levels, self.idx = np.unique(scenes, return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.levels), size=(n_boot, len(self.levels)))
        self.W = np.stack([np.bincount(d, minlength=len(self.levels)) for d in draws]).astype(float)

    def sums(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.bincount(self.idx[mask], weights=values[mask].astype(float), minlength=len(self.levels))

    def diff_ci(self, num_a, den_a, num_b, den_b, mask) -> tuple[float, float]:
        with np.errstate(invalid="ignore", divide="ignore"):
            ra = (self.W @ self.sums(num_a, mask)) / (self.W @ self.sums(den_a, mask))
            rb = (self.W @ self.sums(num_b, mask)) / (self.W @ self.sums(den_b, mask))
        lo, hi = np.nanpercentile(rb - ra, [2.5, 97.5])
        return float(lo), float(hi)


def rate_row(boot: Boot, mask, num: dict, den: dict, arms, r: dict, metric: str) -> dict:
    for arm in arms:
        k, n = float(num[arm][mask].sum()), float(den[arm][mask].sum())
        r[f"{metric}_{arm}"] = k / n if n else math.nan
        r[f"{metric}_{arm}_lo"], r[f"{metric}_{arm}_hi"] = wilson(k, n)
        r[f"{metric}_{arm}_n"] = int(n)
    if len(arms) >= 2 and mask.any():
        a, b = arms[0], arms[1]
        r[f"{metric}_d"] = r[f"{metric}_{b}"] - r[f"{metric}_{a}"]
        r[f"{metric}_d_lo"], r[f"{metric}_d_hi"] = boot.diff_ci(num[a], den[a], num[b], den[b], mask)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["A", "B"])
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--out", type=Path, default=ROOT / "bench/analysis")
    args = p.parse_args()
    arms = args.arms
    args.out.mkdir(parents=True, exist_ok=True)

    truths = {}
    for arm in ("A", "B"):
        ds = load_dataset(DATASETS[arm], check_files=False)
        truths[arm] = dataset_ground_truth(ds, ds.manifest.image_ids("test")).labels.reset_index(drop=True)
    pd.testing.assert_frame_equal(truths["A"], truths["B"])
    truth = truths["A"]
    meta, intervals = scene_tables()
    lab = truth[truth["label"] == "chirp"].copy()
    lab["scene"] = lab["recording"].map(scene_of)
    lab = lab.merge(meta, on="scene", how="left", validate="many_to_one")
    lab = lab.merge(intervals, on=["scene", "chirp_id"], how="left", validate="many_to_one")
    lab.index = truth.index[truth["label"] == "chirp"]
    assert lab["sweep"].notna().all() and lab["dt_same_s"].notna().all()
    lab["emitter_df_hz"] = lab["min_df_live_hz"].astype(float)
    lab["df_bin"] = pd.cut(lab["emitter_df_hz"], DF_EDGES, right=False, labels=DF_BINS).astype(object)
    lab["dt_bin"] = pd.cut(lab["dt_same_s"], DT_EDGES, right=False, labels=DT_BINS).astype(object)

    for arm in arms:
        for stage in ("before", "after"):
            for tol in ("100", "025"):
                o = arm_outcomes(arm, stage, tol, len(truth)).loc[lab.index]
                lab[f"found_{arm}_{stage}_{tol}"] = o["found"].to_numpy(bool)
                lab[f"correct_{arm}_{stage}_{tol}"] = o["correct"].to_numpy(bool)
                if stage == "after" and tol == "100":
                    lab[f"err_{arm}"] = np.abs(o["pred_hz"].to_numpy(float) - lab["emitter_hz"].to_numpy(float))
    lab.to_parquet(args.out / "labels.parquet")
    ones = np.ones(len(lab))
    boot = Boot(lab["scene"].to_numpy(), args.n_boot)

    # ---- sweep (a): ΔEODf
    a = lab["sweep"].to_numpy() == "df"
    st, tol = "after", "100"
    found = {arm: lab[f"found_{arm}_{st}_{tol}"].to_numpy() for arm in arms}
    correct = {arm: lab[f"correct_{arm}_{st}_{tol}"].to_numpy() for arm in arms}
    all_found = np.logical_and.reduce([found[arm] for arm in arms])
    df_rows = []
    for key, col, order in (("emitter ΔF", "df_bin", DF_BINS), ("design level", "level", sorted(lab.loc[a, "level"].unique()))):
        for name in ["all", *order]:
            m = a & ((lab[col] == name).to_numpy() if name != "all" else True)
            if not m.any():
                continue
            r = {"by": key, "bin": name, "n_labels": int(m.sum()), "n_scenes": int(lab.loc[m, "scene"].nunique()),
                 "median_df_hz": float(np.median(lab.loc[m, "emitter_df_hz"]))}
            r = rate_row(boot, m & all_found, correct, {k: ones for k in arms}, arms, r, "assign_both")
            r = rate_row(boot, m, correct, found, arms, r, "assign_own")
            r = rate_row(boot, m, correct, {k: ones for k in arms}, arms, r, "end_to_end")
            r = rate_row(boot, m, found, {k: ones for k in arms}, arms, r, "recall")
            for arm in arms:
                r[f"err_median_{arm}"] = float(np.nanmedian(lab.loc[m & all_found, f"err_{arm}"]))
            df_rows.append(r)
    df_table = pd.DataFrame(df_rows)
    df_table.to_csv(args.out / "sweep_df.csv", index=False)

    # ---- sweep (b): chirp rate
    b = lab["sweep"].to_numpy() == "rate"
    lab_tiles = lab.loc[b, "recording"]
    rate_rows = []
    for tol in ("025", "100"):
        for stage in ("after", "before"):
            found = {arm: lab[f"found_{arm}_{stage}_{tol}"].to_numpy() for arm in arms}
            correct = {arm: lab[f"correct_{arm}_{stage}_{tol}"].to_numpy() for arm in arms}
            fp = {arm: arm_outcomes(arm, stage, tol, len(truth)).attrs["fp_per_tile"] for arm in arms}
            for key, col, order in (("same-fish interval", "dt_bin", DT_BINS), ("design level", "level", sorted(lab.loc[b, "level"].unique(), reverse=True))):
                for name in ["all", *order]:
                    m = b & ((lab[col] == name).to_numpy() if name != "all" else True)
                    if not m.any():
                        continue
                    r = {"tol_ms": int(tol), "stage": stage, "by": key, "bin": name, "n_labels": int(m.sum()),
                         "n_scenes": int(lab.loc[m, "scene"].nunique()), "median_dt_ms": float(1000 * np.median(lab.loc[m, "dt_same_s"]))}
                    r = rate_row(boot, m, found, {k: ones for k in arms}, arms, r, "recall")
                    r = rate_row(boot, m, correct, found, arms, r, "assign_own")
                    if key == "design level":
                        tiles = set(lab_tiles[(lab.loc[b, col] == name).to_numpy()] if name != "all" else lab_tiles)
                        for arm in arms:
                            tp = int(found[arm][m].sum())
                            nfp = int(fp[arm][fp[arm].index.isin(tiles)].sum())
                            r[f"precision_{arm}"] = tp / (tp + nfp) if tp + nfp else math.nan
                            r[f"fp_{arm}"] = nfp
                    rate_rows.append(r)
    rate_table = pd.DataFrame(rate_rows)
    rate_table.to_csv(args.out / "sweep_rate.csv", index=False)

    # ---- markdown
    def pct(x):
        return "" if not np.isfinite(x) else f"{100 * x:.1f}"

    def diff(r, metric):
        if f"{metric}_d" not in r or not np.isfinite(r[f"{metric}_d"]):
            return ""
        return f"{100 * r[f'{metric}_d']:+.1f} [{100 * r[f'{metric}_d_lo']:+.1f}, {100 * r[f'{metric}_d_hi']:+.1f}]"

    A, B = arms[0], arms[1]
    md = [f"# Breakdown benchmark: {A} vs {B}", "", "Score >= 0.5, after post-processing; B - A with a paired scene bootstrap (95 %).", ""]
    for key in ("emitter ΔF", "design level"):
        md += [f"## Sweep (a), by {key}", "",
               f"| {key} | n | median ΔF | assign {A} | assign {B} | {B}-{A} (both found) | end-to-end {A} | {B} | {B}-{A} | median err {A} / {B} (Hz) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in df_table[df_table["by"] == key].to_dict("records"):
            md.append(f"| {r['bin']} | {r['n_labels']} | {r['median_df_hz']:.2f} | {pct(r[f'assign_both_{A}'])} | {pct(r[f'assign_both_{B}'])} | {diff(r, 'assign_both')} | "
                      f"{pct(r[f'end_to_end_{A}'])} | {pct(r[f'end_to_end_{B}'])} | {diff(r, 'end_to_end')} | {r[f'err_median_{A}']:.2f} / {r[f'err_median_{B}']:.2f} |")
        md.append("")
    for tol in (25, 100):
        for key in ("same-fish interval", "design level"):
            sel = rate_table[(rate_table["tol_ms"] == tol) & (rate_table["by"] == key)]
            after, before = sel[sel["stage"] == "after"], sel[sel["stage"] == "before"].set_index("bin")
            md += [f"## Sweep (b), by {key}, matching within {tol} ms", "",
                   f"| {key} | n | recall {A} | recall {B} | {B}-{A} | recall before pp {A} / {B} | assign {A} / {B} |" + (f" precision {A} / {B} |" if key == "design level" else ""),
                   "|---|---|---|---|---|---|---|" + ("---|" if key == "design level" else "")]
            for r in after.to_dict("records"):
                rb = before.loc[r["bin"]]
                line = (f"| {r['bin']} | {r['n_labels']} | {pct(r[f'recall_{A}'])} | {pct(r[f'recall_{B}'])} | {diff(r, 'recall')} | "
                        f"{pct(rb[f'recall_{A}'])} / {pct(rb[f'recall_{B}'])} | {pct(r[f'assign_own_{A}'])} / {pct(r[f'assign_own_{B}'])} |")
                if key == "design level":
                    line += f" {pct(r[f'precision_{A}'])} / {pct(r[f'precision_{B}'])} |"
                md.append(line)
            md.append("")
    (args.out / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
