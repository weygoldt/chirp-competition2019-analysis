"""Training report of a detector run: curves, metric tables, example tiles.

Usage::

    uv run python make_report.py RUN_DIR --eval EVAL_DIR

Writes RUN_DIR/report.md and RUN_DIR/report/*.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import REC_2019, REC_2022  # noqa: E402


import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gridtools.io import open_recording
from matplotlib.patches import Rectangle

from chirpdetector.bundle import load_bundle
from chirpdetector.evaluation import GroundTruth, read_ground_truth
from chirpdetector.inference import render_span

LABEL_FILES = (
    str(REC_2019 / "2019-12-09-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2019 / "2020-06-03-10_10/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-05-12-10_00/traces-grid1-editable-labels.csv"),
    str(REC_2022 / "raw/2022-06-01-10_00/traces-grid1-editable-labels.csv"),
)
V1 = {"precision": 1.000, "recall": 0.764, "f1": 0.866, "labels": 335}
THRESHOLD = 0.5


def plot_curves(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
    for name in ("box_loss", "cls_loss", "l1_loss"):
        axes[0].plot(curves["epoch"], curves[f"train/{name}"], label=f"train {name}")
        axes[0].plot(curves["epoch"], curves[f"val/{name}"], "--", label=f"val {name}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=6)
    for name in ("precision", "recall", "mAP50", "mAP50-95"):
        axes[1].plot(curves["epoch"], curves[f"metrics/{name}(B)"], label=name)
    axes[1].set_xlabel("epoch")
    axes[1].set_title("validation (val groups)", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].plot(curves["epoch"], curves["lr/pg0"])
    axes[2].set_xlabel("epoch")
    axes[2].set_title("learning rate", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def table(df: pd.DataFrame, columns: list[str], floats: str = "{:.3f}") -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        cells = [
            floats.format(v) if isinstance(v, float) else str(v) for v in record.values()
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def example_tiles(
    bundle, items: pd.DataFrame, truth: GroundTruth, preds: pd.DataFrame, out: Path
) -> list[Path]:
    """Spectrogram tiles around a few TPs, misses and false positives."""
    recordings = {Path(p).parent.name: Path(p).parent for p in LABEL_FILES}
    rng = np.random.default_rng(0)
    paths = []
    for outcome, title in (("tp", "true positives"), ("fn", "misses"), ("fp", "false positives")):
        chosen = items[items["outcome"] == outcome]
        if not len(chosen):
            continue
        chosen = chosen.iloc[rng.permutation(len(chosen))[:6]]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
        for ax, item in zip(axes.flat, chosen.to_dict(orient="records"), strict=False):
            name = item["recording"]
            centre = item["label_t_center_s"] if outcome != "fp" else item["pred_t_center_s"]
            recording = open_recording(recordings[name])
            tiles = render_span(bundle, recording, centre - 1.6, centre + 1.6, device="cuda")
            image, g = tiles.images[0], tiles.geometries[0]
            t0, t1 = g.time_extent_s
            f0, f1 = g.freq_extent_hz
            ax.imshow(image[:, :, 0], aspect="auto", cmap="magma", extent=(t0, t1, f0, f1))
            lab = truth.labels[(truth.labels["recording"] == name)]
            for row in lab.itertuples():
                if row.t_end_s >= t0 and row.t_start_s <= t1:
                    ax.add_patch(Rectangle((row.t_start_s, row.f_low_hz), row.t_end_s - row.t_start_s, row.f_high_hz - row.f_low_hz, fill=False, ec="lime", lw=1.2))
            pr = preds[(preds["recording"] == name) & (preds["score"] >= THRESHOLD)]
            for row in pr.itertuples():
                if row.t_end_s >= t0 and row.t_start_s <= t1:
                    ax.add_patch(Rectangle((row.t_start_s, row.f_low_hz), row.t_end_s - row.t_start_s, row.f_high_hz - row.f_low_hz, fill=False, ec="cyan", lw=0.8, ls="--"))
                    ax.text(row.t_start_s, row.f_high_hz, f"{row.score:.2f}", color="cyan", fontsize=6)
            ax.axvline(centre, color="w", lw=0.5, alpha=0.5)
            ax.set_title(f"{name} {centre:.2f} s", fontsize=8)
            ax.set_xlabel("time (s)", fontsize=7)
            ax.set_ylabel("Hz", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"real data, {title} at threshold {THRESHOLD} (green: labels, cyan: detections; channel 0 = 205 ms window)", fontsize=9)
        path = out / f"examples_{outcome}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--eval", type=Path, required=True)
    args = parser.parse_args()
    run, ev = args.run, args.eval
    out = run / "report"
    out.mkdir(exist_ok=True)
    result = json.loads((run / "run.json").read_text())
    final = next(f for f in result["fits"] if f["name"] == "final")
    bundle = load_bundle(final["bundle"])
    curves = pd.read_csv(final["curves"])
    curves.columns = [c.strip() for c in curves.columns]
    plot_curves(curves, out / "curves.png")
    config = json.loads((run / "config.json").read_text())
    prov = json.loads((run / "provenance.json").read_text())

    real_summary = pd.read_csv(ev / "real" / "summary.csv")
    pooled = real_summary[real_summary["level"] == "all"].copy()
    at = pooled[np.isclose(pooled["threshold"], THRESHOLD)].iloc[0]
    best = pooled.loc[pooled["f1"].idxmax()]
    per_level = real_summary[np.isclose(real_summary["threshold"], THRESHOLD)]
    sim_summary = pd.read_csv(ev / "sim_test" / "summary.csv")
    sim_pooled = sim_summary[sim_summary["level"] == "all"]
    sim_at = sim_pooled[np.isclose(sim_pooled["threshold"], THRESHOLD)].iloc[0]
    strata = pd.read_csv(ev / "sim_test" / "recall_by_stratum.csv")
    items = pd.read_csv(ev / "real" / "items.csv")
    preds = pd.read_parquet(ev / "real_predictions.parquet")
    truth = GroundTruth.concat(read_ground_truth(p) for p in LABEL_FILES)
    examples = example_tiles(bundle, items, truth, preds, out)
    ci = pd.read_csv(ev / "real" / "intervals.csv", index_col=0)
    ap = pd.read_csv(ev / "real" / "average_precision.csv")

    lines = [
        f"# Detector run {result['run_id']}",
        "",
        f"Bundle `{final['bundle']}`, weights sha256 `{bundle.weights_sha256}`; "
        f"dataset `{result['dataset_sha256'][:16]}…` (sim_v1); "
        f"chirpdetector `{prov['code']['chirpdetector']['sha'][:10]}` "
        f"(dirty: {prov['code']['chirpdetector']['dirty']}), gridtools "
        f"`{(prov['code']['gridtools']['sha'] or '?')[:10]}`.",
        "",
        "## Training",
        "",
        f"- Model {config['model']['architecture']} (COCO-pretrained: {config['model']['pretrained']}), "
        f"input {bundle.model['input_shape']} (205 x 640 tiles zero-padded), NMS-free one-to-one head.",
        f"- {config['optim']['optimizer']}, lr {config['optim']['lr0']} cosine to x{config['optim']['lrf']}, "
        f"batch {config['optim']['batch']}, AMP {config['optim']['amp']}, patience {config['optim']['patience']}; "
        f"augmentation: none (time_flip {config['augment']['time_flip']}).",
        f"- {final['n_train_images']} training tiles ({len(final['train_groups'])} scenes), "
        f"{final['n_val_images']} validation tiles ({len(final['val_groups'])} scenes); "
        f"{final['epochs_run']} epochs, best epoch {final['best_epoch']}. No cross-validation in this run.",
        "",
        "![curves](report/curves.png)",
        "",
        "Validation metrics of the selected epoch (ultralytics, val scenes):",
        "",
        "| metric | value |\n|---|---|\n"
        + "\n".join(f"| {k} | {v:.3f} |" for k, v in final["validation_metrics"].items() if k.startswith("metrics/")),
        "",
        "## Simulated test split",
        "",
        f"At threshold {THRESHOLD}: precision {sim_at['precision']:.3f}, recall {sim_at['recall']:.3f}, "
        f"F1 {sim_at['f1']:.3f} ({int(sim_at['labels'])} chirps, {int(sim_at['tp'])} TP, "
        f"{int(sim_at['fp'])} FP; ignore regions neither count nor penalise).",
        "",
        table(sim_pooled, ["threshold", "labels", "tp", "fp", "fn", "precision", "recall", "f1"]),
        "Recall by stratum (threshold 0.5):",
        "",
        table(strata, ["stratum", "value", "labels", "recall"]),
        "## Real hand labels (v1 benchmark rules)",
        "",
        f"Four recordings, {int(at['labels'])} labelled chirps in "
        f"{int(ci.loc['all', 'n_regions']) if 'n_regions' in ci.columns else '?'} regions; "
        "matching: centres within 100 ms, overlapping bands, one-to-one; predictions "
        "outside a region's band dropped (harmonics).",
        "",
        f"| | precision | recall | F1 |\n|---|---|---|---|\n"
        f"| v1 (YOLOv8l, threshold 0.5) | {V1['precision']:.3f} | {V1['recall']:.3f} | {V1['f1']:.3f} |\n"
        f"| this run, threshold 0.5 | {at['precision']:.3f} | {at['recall']:.3f} | {at['f1']:.3f} |\n"
        f"| this run, best F1 (threshold {best['threshold']:.2f}) | {best['precision']:.3f} | {best['recall']:.3f} | {best['f1']:.3f} |",
        "",
        f"95% CIs at 0.5 (exact binomial): precision [{ci.loc['all', 'precision_lo']:.3f}, {ci.loc['all', 'precision_hi']:.3f}], "
        f"recall [{ci.loc['all', 'recall_lo']:.3f}, {ci.loc['all', 'recall_hi']:.3f}]. "
        f"AP (pooled): {ap[ap['level'] == 'all']['average_precision'].iloc[0]:.3f}.",
        "",
        "Threshold sweep (pooled):",
        "",
        table(pooled, ["threshold", "labels", "tp", "fp", "fn", "precision", "recall", "f1"]),
        "By recording at 0.5:",
        "",
        table(per_level[per_level["level"] != "all"].fillna(""), ["level", "dataset", "recording", "labels", "tp", "fp", "fn", "precision", "recall", "f1"]),
        "Caveat: the threshold sweep is on the locked test set; choosing the operating "
        "point from it is test-set selection. The 0.5 row is the pre-registered one.",
        "",
        "## Examples (real data)",
        "",
        *[f"![{p.stem}](report/{p.name})" for p in examples],
        "",
    ]
    (run / "report.md").write_text("\n".join(lines))
    print(run / "report.md")


if __name__ == "__main__":
    main()
