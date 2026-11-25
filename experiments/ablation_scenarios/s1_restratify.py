"""Scenario 1a: emitter assignment of arms A and B by the EMITTER's own ΔF.

    uv run python experiments/ablation_scenarios/s1_restratify.py [--out DIR] [--n-boot 4000]

The front-end ablation (experiments/ablation, report
$CHIRP_DATA/runs/ablation_frontend/report.md) stratified assignment by the
evaluation's ``df_bin``, which is the closest pair of ANY two live tracks in
the tile, not the emitter's distance to its nearest neighbour (most test
scenes hold 3-7 fish, so that pair often excludes the emitter). Here the same
stored outputs (``{A,B}/eval_postprocess/sim_test_after/items.csv``, nothing
re-predicted) are re-binned by ``min_df_live_hz`` of each label: the
emitter's ΔF to the nearest other live fish (simulated or real background
fish) at the chirp centre.

Per bin and arm: n labels, recall, assignment accuracy of the true positives,
end-to-end (found with the right emitter), the median emitter-head error; B - A
with a paired bootstrap over the 300 test scenes. Also a "both found" variant
of the assignment accuracy (labels both arms detected), which removes the
selection difference between the arms' true-positive sets.

Writes ``by_emitter_df.csv``, ``labels.parquet`` (one row per label, both
arms), ``table.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

from chirpdetector.data import dataset_ground_truth, load_dataset  # noqa: E402

ROOT = DATA / "runs/ablation_frontend"
SIM = DATA / "datasets/sim_v4"
EDGES = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, np.inf]


def bin_names(edges: list[float]) -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        out.append(f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}")
    return out


BINS = bin_names(EDGES)


def scene_of(tile: str) -> str:
    return tile.rsplit("_f", 1)[0]


def arm_labels(arm: str, truth: pd.DataFrame) -> pd.DataFrame:
    ev = ROOT / arm / "eval_postprocess"
    items = pd.read_csv(ev / "sim_test_after/items.csv")
    lab = items[(items["label"] == "chirp") & items["outcome"].isin(["tp", "fn"])].copy()
    lab["label_row"] = lab["label_row"].astype(int)
    meta = truth.iloc[lab["label_row"].to_numpy()]
    assert np.allclose(lab["label_t_center_s"].to_numpy(float), meta["t_center_s"].to_numpy(float))
    flagged = pd.read_parquet(ev / "sim_test_predictions_flagged.parquet")
    kept = flagged[flagged["drop_reason"].astype(str) == ""].reset_index(drop=True)
    pred = lab["pred_row"].fillna(-1).astype(int).to_numpy()
    found = (lab["outcome"] == "tp").to_numpy()
    pred_hz = np.where(pred >= 0, kept["emitter_hz"].to_numpy(float)[np.clip(pred, 0, None)], np.nan)
    true_id = lab["true_emitter_id"].astype("Int64").fillna(-2).to_numpy()
    pred_id = lab["pred_emitter_id"].astype("Int64").fillna(-1).to_numpy()
    out = pd.DataFrame(
        {
            "label_row": lab["label_row"].to_numpy(),
            "tile": lab["recording"].to_numpy(),
            "found": found,
            "correct": np.where(found, (pred_id == true_id), False),
            "pred_hz": pred_hz,
        }
    )
    return out.set_index("label_row")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "scenarios/s1_restratify")
    p.add_argument("--n-boot", type=int, default=4000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(SIM, check_files=False)
    ids = ds.manifest.image_ids("test")
    truth = dataset_ground_truth(ds, ids).labels.reset_index(drop=True)
    a, b = arm_labels("A", truth), arm_labels("B", truth)
    assert set(a.index) == set(b.index), "arms must score the same labels"
    rows = a.index.to_numpy()
    t = truth.iloc[rows]
    lab = pd.DataFrame(
        {
            "tile": a["tile"].to_numpy(),
            "scene": [scene_of(s) for s in a["tile"]],
            "emitter_hz": t["emitter_hz"].to_numpy(float),
            "emitter_df_hz": t["min_df_live_hz"].to_numpy(float),
            "family": t["family"].to_numpy(),
            "duration_s": t["duration_s"].to_numpy(float),
            "snr_db": t["snr_db"].to_numpy(float),
        },
        index=rows,
    )
    for arm, d in (("A", a), ("B", b.loc[rows])):
        lab[f"found_{arm}"] = d["found"].to_numpy()
        lab[f"correct_{arm}"] = d["correct"].to_numpy()
        lab[f"err_{arm}"] = np.abs(d["pred_hz"].to_numpy() - lab["emitter_hz"].to_numpy())
    single = ~np.isfinite(lab["emitter_df_hz"].to_numpy())
    lab["bin"] = pd.cut(lab["emitter_df_hz"], EDGES, right=False, labels=BINS).astype(object)
    lab.loc[single, "bin"] = "single"
    lab.to_parquet(args.out / "labels.parquet")

    scenes = sorted(lab["scene"].unique())
    sidx = lab["scene"].map({s: k for k, s in enumerate(scenes)}).to_numpy()
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(scenes), size=(args.n_boot, len(scenes)))
    W = np.stack([np.bincount(d, minlength=len(scenes)) for d in draws]).astype(float)

    def boot_diff(num_a, num_b, den_a, den_b, mask):
        def sums(v):
            return np.bincount(sidx[mask], weights=v[mask].astype(float), minlength=len(scenes))

        with np.errstate(invalid="ignore", divide="ignore"):
            ra = (W @ sums(num_a)) / (W @ sums(den_a))
            rb = (W @ sums(num_b)) / (W @ sums(den_b))
        d = rb - ra
        return np.nanpercentile(d, [2.5, 97.5])

    out = []
    fa, fb = lab["found_A"].to_numpy(), lab["found_B"].to_numpy()
    ca, cb = lab["correct_A"].to_numpy(), lab["correct_B"].to_numpy()
    both = fa & fb
    for name in ["all", "single", *BINS]:
        m = np.ones(len(lab), bool) if name == "all" else (lab["bin"] == name).to_numpy()
        if m.sum() == 0:
            continue
        r = {"bin": name, "n_labels": int(m.sum()), "n_scenes": int(len(set(sidx[m])))}
        ones = np.ones(len(lab))
        for metric, num_a, num_b, den_a, den_b, mm in (
            ("recall", fa, fb, ones, ones, m),
            ("assign", ca, cb, fa, fb, m),
            ("assign_both", ca, cb, ones, ones, m & both),
            ("end_to_end", ca, cb, ones, ones, m),
        ):
            va = num_a[mm].sum() / max(den_a[mm].sum(), 1)
            vb = num_b[mm].sum() / max(den_b[mm].sum(), 1)
            lo, hi = boot_diff(num_a, num_b, den_a, den_b, mm)
            r[f"{metric}_A"], r[f"{metric}_B"] = va, vb
            r[f"{metric}_dBA"], r[f"{metric}_lo"], r[f"{metric}_hi"] = vb - va, lo, hi
        r["n_both"] = int((m & both).sum())
        r["only_A_correct"] = int((m & both & ca & ~cb).sum())
        r["only_B_correct"] = int((m & both & cb & ~ca).sum())
        r["err_med_A"] = float(np.nanmedian(lab.loc[m & both, "err_A"]))
        r["err_med_B"] = float(np.nanmedian(lab.loc[m & both, "err_B"]))
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(args.out / "by_emitter_df.csv", index=False)

    def pct(x):
        return f"{100 * x:.1f}"

    lines = [
        "| emitter ΔF (Hz) | n | assign A | assign B | B-A [95% CI] (both found) | A-only / B-only correct | end-to-end A | B | B-A [95% CI] | median err A / B (Hz) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res.itertuples():
        lines.append(
            f"| {r.bin} | {r.n_labels} | {pct(r.assign_both_A)} | {pct(r.assign_both_B)} | "
            f"{100 * r.assign_both_dBA:+.1f} [{100 * r.assign_both_lo:+.1f}, {100 * r.assign_both_hi:+.1f}] | "
            f"{r.only_A_correct} / {r.only_B_correct} | {pct(r.end_to_end_A)} | {pct(r.end_to_end_B)} | "
            f"{100 * r.end_to_end_dBA:+.1f} [{100 * r.end_to_end_lo:+.1f}, {100 * r.end_to_end_hi:+.1f}] | "
            f"{r.err_med_A:.2f} / {r.err_med_B:.2f} |"
        )
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
