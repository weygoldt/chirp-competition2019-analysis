"""Screen background windows with a chirpdetector v2 model bundle.

Second chirp screen of the background library (the first is the v1
production detector, ``scripts/v1_screen.py``): a v2 YOLO26 model trained on
hybrid simulations finds chirps v1 misses. Two commands:

``bundle RUN_DIR OUT``
    Make a model bundle from a training run that was stopped before it
    wrote one: the run's ``final/ultralytics/weights/best.pt`` (the epoch
    with the best ultralytics fitness, 0.1 mAP50 + 0.9 mAP50-95, on the
    validation groups) is exported exactly as the trainer exports it and
    bundled with the dataset's front end, tiling and preprocessing; the
    training record says the run was stopped and which epoch was kept.
``screen BUNDLE OUT RECORDING...``
    Run the bundle over the last ``--last-s`` seconds of each recording, in
    chunks (each chunk owns the detections centred in it), and write the
    screen format read by ``gridtools.background.screen``::

        OUT/screen.json                    detector, weights sha256, thresholds
        OUT/<dataset>/<recording>.csv      t_start_s, t_end_s, f_low_hz,
                                           f_high_hz, score (>= store threshold)
        OUT/<dataset>/<recording>.json     covered_s, run info

Runs in the workspace environment (``uv run``), through ``scripts/guarded``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger("v2_screen")
BOX_COLUMNS = ["t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score"]


def fitness(row: dict[str, str]) -> float:
    """ultralytics' default fitness of a results.csv row."""
    return 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(
        row["metrics/mAP50-95(B)"]
    )


def make_bundle(run_dir: Path, out: Path, dataset_dir: Path | None) -> Path:
    from chirpdetector.bundle import write_bundle
    from chirpdetector.data.contract import load_dataset
    from chirpdetector.models._yolo_engine import (
        PAD_VALUE,
        ULTRALYTICS_VERSION,
        export_weights,
    )
    from chirpdetector.models.yolo import YoloModelConfig, input_shape
    from chirpdetector.training.detector import (
        FINAL,
        DetectorTrainingConfig,
        plan_fits,
    )

    run = json.loads((run_dir / "run.json").read_text())
    provenance = json.loads((run_dir / FINAL / "provenance.json").read_text())
    if dataset_dir is None:
        command = provenance["command"]
        dataset_dir = Path(command[command.index("--dataset") + 1])
    dataset = load_dataset(dataset_dir, check_files=False)
    manifest = dataset.manifest
    if manifest.content_sha256 != run["dataset_sha256"]:
        msg = f"{dataset_dir} is not the run's dataset"
        raise ValueError(msg)
    config = DetectorTrainingConfig.from_toml(run_dir / "config.toml")
    _, fits = plan_fits(dataset, config, FINAL)
    _, train_groups, val_groups = fits[0]
    results = run_dir / FINAL / "ultralytics" / "results.csv"
    with results.open() as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    best = max(rows, key=fitness)
    model_cfg = YoloModelConfig(**config.model)
    checkpoint = run_dir / FINAL / "ultralytics" / "weights" / "best.pt"
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        weights = Path(tmp) / "weights.pt"
        export_weights(checkpoint, weights)
        bundle = write_bundle(
            out,
            weights=weights,
            model_type=config.model_type,
            model={
                "architecture": model_cfg.architecture,
                "input_shape": list(
                    input_shape(manifest.tiling.height_px, manifest.tiling.width_px)
                ),
                "pad_value": PAD_VALUE,
                "padding": "bottom-right",
                "channels": "window order (0 = reference window)",
                "head": "one2one",
                "max_det": model_cfg.max_det,
                "ultralytics": ULTRALYTICS_VERSION,
            },
            spectrogram=manifest.spectrogram,
            tiling=manifest.tiling,
            preprocessing=manifest.preprocessing,
            score_threshold=config.score_threshold,
            training={
                "run_id": run["run_id"],
                "fit": FINAL,
                "dataset_sha256": manifest.content_sha256,
                "train_groups": list(train_groups),
                "val_groups": list(val_groups),
                "best_epoch": int(float(best["epoch"])),
                "epochs_run": len(rows),
                "validation_metrics": {
                    k: float(v) for k, v in best.items() if k.startswith("metrics/")
                },
                "config": config.to_dict(),
                "code": provenance["code"],
                "uv_lock": provenance["uv_lock"],
                "note": (
                    f"run stopped after {len(rows)} of {config.optim.epochs} epochs "
                    "without a bundle; made afterwards by scripts/v2_screen.py from "
                    f"{checkpoint} (best fitness at epoch {best['epoch']})"
                ),
                "bundled_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "checkpoint_sha256": _sha256(checkpoint),
            },
        )
    log.info("bundle %s (weights %s)", bundle.path, bundle.weights_sha256[:12])
    return bundle.path


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def screen(
    bundle_dir: Path,
    out: Path,
    recordings: list[Path],
    *,
    dataset: str,
    last_s: float | None,
    lead_s: float,
    chunk_s: float,
    pad_s: float,
    store_threshold: float,
    overwrite: bool,
) -> None:
    import torch
    from chirpdetector.bundle import load_bundle
    from chirpdetector.inference.spans import predict_span
    from chirpdetector.prediction import load_detector
    from gridtools.io import open_recording

    bundle = load_bundle(bundle_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = load_detector(bundle, device=device)
    meta = {
        "name": "v2",
        "detector": f"chirpdetector v2 bundle, {bundle.model.get('architecture')} "
        f"(run {bundle.training.get('run_id')})",
        "bundle": str(bundle.path),
        "weights_sha256": bundle.weights_sha256,
        "store_threshold": store_threshold,
        "bundle_score_threshold": bundle.score_threshold,
        "chunk_s": chunk_s,
        "pad_s": pad_s,
        "tables": "predict_span over chunks; each chunk keeps the detections "
        "centred in its own part",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "script": str(Path(__file__).resolve()),
    }
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "screen.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ("weights_sha256", "store_threshold"):
            if old.get(key) != meta[key]:
                msg = f"{meta_path}: {key} differs ({old.get(key)} vs {meta[key]})"
                raise RuntimeError(msg)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    for path in recordings:
        target = out / dataset / f"{path.name}.csv"
        if target.exists() and not overwrite:
            log.info("%s: done, skipped", path.name)
            continue
        t0 = time.perf_counter()
        rec = open_recording(path)
        duration = rec.duration_s
        lo = 0.0 if last_s is None else max(0.0, duration - last_s - lead_s)
        hi = duration
        frames = []
        n_chunks = math.ceil((hi - lo) / chunk_s)
        for k in range(n_chunks):
            own_a = lo + k * chunk_s
            own_b = min(hi, own_a + chunk_s)
            a, b = max(lo, own_a - pad_s), min(hi, own_b + pad_s)
            det = predict_span(
                bundle,
                rec,
                a,
                b,
                name=path.name,
                score_threshold=store_threshold,
                detector=detector,
                device=device,
            )
            if len(det):
                centre = 0.5 * (det["t_start_s"] + det["t_end_s"])
                last = k == n_chunks - 1
                keep = (centre >= own_a) & (
                    (centre <= own_b) if last else (centre < own_b)
                )
                frames.append(pd.DataFrame(det.loc[keep.to_numpy(), BOX_COLUMNS]))
        boxes = (
            pd.concat(frames, ignore_index=True).sort_values("t_start_s")
            if frames
            else pd.DataFrame(columns=BOX_COLUMNS)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".csv.part")
        boxes.to_csv(part, index=False, float_format="%.12g")
        info = {
            "recording": path.name,
            "dataset": dataset,
            "path": str(path),
            "samplerate_hz": rec.samplerate_hz,
            "n_samples": rec.n_samples,
            "duration_s": duration,
            "requested_last_s": last_s,
            "covered_s": [[lo, hi]],
            "n_chunks": n_chunks,
            "n_boxes": len(boxes),
            "n_boxes_ge_0.2": int((boxes["score"] >= 0.2).sum()) if len(boxes) else 0,
            "n_boxes_ge_0.5": int((boxes["score"] >= 0.5).sum()) if len(boxes) else 0,
            "runtime_s": round(time.perf_counter() - t0, 1),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (target.parent / f"{path.name}.json").write_text(
            json.dumps(info, indent=1) + "\n"
        )
        part.rename(target)
        log.info(
            "%s: %d boxes (%d >= 0.2) over %.0f s in %.0f s",
            path.name,
            info["n_boxes"],
            info["n_boxes_ge_0.2"],
            hi - lo,
            info["runtime_s"],
        )
        if device == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle", help="bundle a stopped training run")
    b.add_argument("run_dir", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--dataset", type=Path, default=None)
    s = sub.add_parser("screen", help="screen recordings")
    s.add_argument("bundle", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("recordings", nargs="+", type=Path)
    s.add_argument("--dataset", required=True)
    s.add_argument("--last-s", type=float, default=None)
    s.add_argument(
        "--lead-s",
        type=float,
        default=10.0,
        help="also screen this much before the window (margins of its first candidates)",
    )
    s.add_argument("--chunk-s", type=float, default=300.0)
    s.add_argument("--pad-s", type=float, default=3.0)
    s.add_argument("--store-threshold", type=float, default=0.05)
    s.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if args.command == "bundle":
        make_bundle(args.run_dir, args.out, args.dataset)
    else:
        if os.environ.get("CHIRP_MAX_VRAM_FRACTION") is None:
            log.warning("GPU memory not capped: run through scripts/guarded")
        screen(
            args.bundle,
            args.out,
            args.recordings,
            dataset=args.dataset,
            last_s=args.last_s,
            lead_s=args.lead_s,
            chunk_s=args.chunk_s,
            pad_s=args.pad_s,
            store_threshold=args.store_threshold,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
