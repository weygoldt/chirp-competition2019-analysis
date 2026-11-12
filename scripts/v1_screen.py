"""Re-run the v1 production chirp detector at a low threshold, for screening.

The paper's chirps were detected on 2024-06-16 by chirpdetector tag
``paper-chirp-competition-v1`` (3447105) with gridtools ``df7f504``, weights
``yolov8l.pt`` (sha256 d7830183c04264c7...) and
``/mnt/data2/2019_tube_competition/chirpdetector.toml`` (one tile 500-1500 Hz,
15-s windows + 1 s overlap, 12 windows per batch, threshold 0.5). The saved
``chirpdetector_bboxes.csv`` stop at score 0.5. This script replays that
detection loop exactly (same windows and batches over the whole recording,
same spectrograms, same model call) but asks ultralytics for every box down to
``--store-threshold`` (default 0.05), and writes the boxes before and after the
pipeline's own non-maximum suppression (torchvision NMS, IoU 0.2, per batch).
Both NMS stages are greedy by score, so thresholding the stored post-NMS table
at any t >= store threshold gives exactly what v1 would have produced at
threshold t (up to ultralytics' 300-boxes-per-image cap, which is checked).

Only detection runs: tracks are never loaded and the emitter network is not
run; its peak search, which drops boxes, is repeated (``assignable``). Same approach as
``~/wrk/analyses/chirp_competition/detector_benchmark/detect.py``, which
reproduced v1's benchmark (P 1.000, R 0.764) exactly.

Environment: the frozen benchmark environment (Python 3.11, ultralytics 8.1.8,
torch 2.1.2), e.g.

.. code-block:: sh

    UV_PROJECT_ENVIRONMENT=$TMP/v1env uv sync --frozen \\
        --project ~/wrk/analyses/chirp_competition/detector_benchmark
    scripts/guarded --name bgscreen-v1 --mem 16G --vram 0.25 -- \\
        $TMP/v1env/bin/python scripts/v1_screen.py \\
        /mnt/data1/chirp_v2/background/screen/v1 --dataset competition2019 \\
        /mnt/data2/2019_tube_competition/2019-11-25-09_59 ...

Output (the format read by ``gridtools.background.screen``)::

    OUT/screen.json                        detector, weights sha256, thresholds
    OUT/<dataset>/<recording>.csv          post-NMS boxes, score >= store threshold
    OUT/<dataset>/<recording>.prenms.csv   boxes before the pipeline's NMS
    OUT/<dataset>/<recording>.json         covered spans, batches, checks, runtime

Columns of the box tables: ``t_start_s, t_end_s, f_low_hz, f_high_hz, score,
batch, window, assignable`` (recording time in s, frequency in Hz).
``assignable`` (post-NMS table) is False for boxes v1's emitter assigner
dropped: ``MLPAssigner.assign`` ends with ``dropna()``, so a box without a
peak in its power profile (no candidate EODf) never reached
``chirpdetector_bboxes.csv``. The saved v1 output is therefore the boxes with
``score >= 0.5`` and ``assignable``; screening can use all boxes (a
superset). Runs extend: batches
already in a recording's tables are not run again (``--overwrite`` starts
over), so ``--last-s`` can screen the library windows first and a later run
without it completes the whole recordings. Nothing is written next to the raw
data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import logging
import os
import pathlib
import platform
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

PRODUCTION_CONFIG = pathlib.Path("/mnt/data2/2019_tube_competition/chirpdetector.toml")
WEIGHTS = pathlib.Path(
    "~/wrk/analyses/chirp_competition/models/yolov8l.pt"
).expanduser()
WEIGHTS_SHA256 = "d7830183c04264c75ae6bb920c0209a5a0c94fff45ad92bca87ebd41e33ae00c"
FROZEN = {
    "chirpdetector": (
        os.environ.get("CHIRPDETECTOR_REPO", "~/wrk/tools/chirpdetector"),
        "paper-chirp-competition-v1",
    ),
    "gridtools": (
        os.environ.get("GRIDTOOLS_REPO", "~/wrk/tools/gridtools"),
        "df7f504b6ca7960381a5bd34962c6e56dc4dfb40",
    ),
}
NMS_IOU = 0.2  # hard-coded in ChirpDetector.detect (v1)
MAX_DET = 300  # ultralytics' per-image cap
RAW_FILE = "traces-grid1.raw"
COLUMNS = [
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "score",
    "batch",
    "window",
    "assignable",
]

log = logging.getLogger("v1_screen")


def export_frozen(cache: pathlib.Path) -> dict[str, dict[str, str]]:
    """``git archive`` both packages at their frozen refs; put them on sys.path."""
    info = {}
    for name, (repo, ref) in FROZEN.items():
        repo_path = pathlib.Path(repo).expanduser()
        sha = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", f"{ref}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        target = cache / f"{name}-{sha[:10]}"
        if not target.exists():
            archive = subprocess.run(
                ["git", "-C", str(repo_path), "archive", sha],
                check=True,
                capture_output=True,
            ).stdout
            tmp = target.with_name(target.name + ".part")
            tmp.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(tmp, filter="data")
            tmp.rename(target)
        sys.path.insert(0, str(target))
        info[name] = {"repo": str(repo_path), "ref": ref, "commit": sha}
    return info


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


class _LowConf:
    """The ultralytics model, asked for every box down to ``conf``."""

    def __init__(self, model, conf: float) -> None:
        self.model = model
        self.conf = conf

    def predict(self, batch, **kwargs):
        return self.model.predict(batch, conf=self.conf, verbose=False, **kwargs)


class _Warnings(logging.Handler):
    """Collects ultralytics warnings (e.g. an NMS time limit hit)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def make_batch_specs_streaming(indices, metadata, batch, samplerate, cfg):
    """v1's ``make_batch_specs``, one window at a time.

    Same calls on the same inputs as
    ``chirpdetector.datahandling.dataset_parsing.make_batch_specs`` (power
    spectrogram per electrode, dB, sum over electrodes, tile, image), but
    each window's per-electrode spectrograms are freed before the next is
    computed, so a 12-window batch of a 15-electrode recording fits in a
    quarter of the GPU.
    """
    import torch
    from chirpdetector.datahandling.dataset_parsing import tile_batch_specs
    from chirpdetector.datahandling.signal_processing import (
        make_spectrogram_axes,
        spec_to_image,
    )
    from gridtools.utils.spectrograms import (
        compute_spectrogram,
        freqres_to_nfft,
        overlap_to_hoplen,
        to_decibel,
    )

    batch = np.swapaxes(batch, 1, 2)
    nfft = freqres_to_nfft(freq_res=cfg.spec.freq_res, samplingrate=samplerate)
    hop_length = overlap_to_hoplen(nfft=nfft, overlap=cfg.spec.overlap_frac)
    sums = []
    for signal in batch:
        spec = compute_spectrogram(
            data=signal, samplingrate=samplerate, nfft=nfft, hop_length=hop_length
        )[0]
        decibel = to_decibel(spec)
        del spec
        sums.append(torch.sum(decibel, dim=0))
        del decibel
    axes = [
        make_spectrogram_axes(
            start=idxs[0],
            stop=idxs[1],
            nfft=nfft,
            hop_length=hop_length,
            samplerate=samplerate,
        )
        for idxs in indices
    ]
    batch_specs = [(spec, *ax) for spec, ax in zip(sums, axes, strict=True)]
    batch_specs = [
        (meta, *spec) for meta, spec in zip(metadata, batch_specs, strict=True)
    ]
    sliced_specs = tile_batch_specs(batch_specs, cfg)
    metadata, specs, times, freqs = zip(*sliced_specs, strict=True)
    images = [spec_to_image(spec) for spec in specs]
    return metadata, images, times, freqs


def v1_assignable(specs, times, freqs, boxes: pd.DataFrame) -> np.ndarray:
    """Whether v1's emitter assigner would keep each box.

    ``MLPAssigner.assign`` of the frozen chirpdetector ends with
    ``batch_detections.dropna()``: a box for which it finds no peak in the
    power profile of its lower two thirds (no candidate EODf) is dropped
    from ``chirpdetector_bboxes.csv``. This repeats exactly that peak search
    (same crop, PIL resize to 100 x 100, prominence 10 % of the range); the
    assignment network itself never rejects (threshold 0 on its output).
    """
    from PIL import Image
    from scipy.signal import find_peaks

    out = []
    spec_index = boxes["spec"].to_numpy().astype(int)
    coords = boxes[["t1", "f1", "t2", "f2"]].to_numpy()
    for idx, box in zip(spec_index, coords, strict=True):
        upper_y_cutoff = 0.66 * (box[3] - box[1]) + box[1]
        try:
            window_spec = specs[idx][
                0, :, (times[idx] > box[0]) & (times[idx] < box[2])
            ]
            window_spec = window_spec[
                (freqs[idx] > box[1]) & (freqs[idx] < upper_y_cutoff)
            ]
            window_spec = np.array(
                Image.fromarray(window_spec.cpu().numpy()).resize((100, 100))
            )
        except Exception:  # an empty crop: v1 itself would have crashed
            out.append(False)
            continue
        target_prom = (np.max(window_spec) - np.min(window_spec)) * 0.1
        power = np.mean(window_spec, axis=1)
        out.append(len(find_peaks(power, prominence=target_prom)[0]) > 0)
    return np.asarray(out, dtype=bool)


def _columns(df: pd.DataFrame) -> pd.DataFrame:
    """v1's box table (t1, f1, t2, f2, score, batch, window) in the screen format."""
    return pd.DataFrame(
        {
            "t_start_s": df["t1"].astype(float),
            "t_end_s": df["t2"].astype(float),
            "f_low_hz": df["f1"].astype(float),
            "f_high_hz": df["f2"].astype(float),
            "score": df["score"].astype(float),
            "batch": df["batch"].astype(int),
            "window": df["window"].astype(int),
            "assignable": df["assignable"].astype(bool)
            if "assignable" in df.columns
            else True,
        }
    )


def _previous(
    d: pathlib.Path, name: str, parser_batches: list, rate: float
) -> dict | None:
    """An earlier run of this recording: its batches and tables, to extend."""
    info_path = d / f"{name}.json"
    if not (info_path.exists() and (d / f"{name}.csv").exists()):
        return None
    info = json.loads(info_path.read_text())
    if info["n_batches_total"] != len(parser_batches):
        msg = f"{info_path}: other batch layout ({info['n_batches_total']} batches)"
        raise RuntimeError(msg)
    batches = info.get("batches")
    if batches is None:  # written before batches were recorded: from the spans
        batches = [
            i
            for i, b in enumerate(parser_batches)
            if any(
                a - 1e-6 <= b[0][0] / rate and b[-1][1] / rate <= z + 1e-6
                for a, z in info["covered_s"]
            )
        ]
    return {
        "info": info,
        "batches": [int(i) for i in batches],
        "pre": pd.read_csv(d / f"{name}.prenms.csv"),
        "post": pd.read_csv(d / f"{name}.csv"),
    }


def run_recording(
    recording: pathlib.Path,
    dataset: str,
    out: pathlib.Path,
    cfg,
    detector,
    store_threshold: float,
    last_s: float | None,
    warnings: _Warnings,
) -> dict:
    import torch
    from chirpdetector.datahandling.bbox_tools import dataframe_nms
    from chirpdetector.datahandling.dataset_parsing import ArrayParser
    from chirpdetector.detection.detect_chirps import convert_detections
    from rich.console import Console
    from thunderlab.dataloader import DataLoader

    t0 = time.perf_counter()
    console = Console(file=io.StringIO())  # the frozen code logs a lot
    rec = DataLoader(str(recording / RAW_FILE))
    rate = float(rec.samplerate)
    n_samples = int(rec.shape[0])
    duration = n_samples / rate
    parser = ArrayParser(
        length=n_samples,
        samplingrate=rate,
        batchsize=cfg.spec.batch_size,
        windowsize=cfg.spec.time_window,
        overlap=cfg.spec.spec_overlap,
        console=console,
    )
    lo = -np.inf if last_s is None else duration - last_s
    d = out / dataset
    previous = _previous(d, recording.name, parser.batches, rate)
    done = set(previous["batches"]) if previous else set()
    pre = [] if previous is None else [previous["pre"]]
    post = [] if previous is None else [previous["post"]]
    max_boxes = 0 if previous is None else int(previous["info"]["max_boxes_per_image"])
    batches = sorted(done)
    n_batches = 0
    for i, batch_indices in enumerate(parser.batches):
        if batch_indices[-1][1] / rate <= lo or i in done:
            continue
        n_batches += 1
        batches.append(i)
        metadata = [
            {
                "recording": recording.name,
                "batch": i,
                "window": j,
                "indices": indices,
                "frange": [],
            }
            for j, indices in enumerate(batch_indices)
        ]
        raw = [np.array(rec[a:b, :]) for a, b in batch_indices]
        metadata, specs, times, freqs = make_batch_specs_streaming(
            batch_indices, metadata, raw, rate, cfg
        )
        before = len(warnings.messages)
        predictions = detector.predict(list(specs).copy())
        if len(warnings.messages) > before:
            msg = f"{recording.name} batch {i}: {warnings.messages[before:]}"
            raise RuntimeError(msg)
        counts = [len(p["boxes"]) for p in predictions]
        max_boxes = max(max_boxes, *counts)
        edges = np.cumsum([0, *counts])
        split_ids = [np.arange(a, b) for a, b in itertools.pairwise(edges)]
        df = convert_detections(predictions, split_ids, metadata, times, freqs, cfg)
        if len(df):
            df = df.reset_index(drop=True)
            keep = dataframe_nms(df, overlapthresh=NMS_IOU)
            pre.append(_columns(df).drop(columns="assignable"))
            kept = df.iloc[keep].copy()
            kept["assignable"] = v1_assignable(specs, times, freqs, kept)
            post.append(_columns(kept))
        del specs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rec.close()

    def table(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=columns)
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["t_start_s", "f_low_hz"], kind="stable")

    # covered intervals: merge abutting batch spans
    batches.sort()
    spans: list[list[float]] = []
    for i in batches:
        a, b = parser.batches[i][0][0] / rate, parser.batches[i][-1][1] / rate
        if spans and a <= spans[-1][1] + 1e-9:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    d.mkdir(parents=True, exist_ok=True)
    table(pre, COLUMNS[:-1]).to_csv(
        d / f"{recording.name}.prenms.csv", index=False, float_format="%.12g"
    )
    post_df = table(post, COLUMNS)
    post_df.to_csv(d / f"{recording.name}.csv.part", index=False, float_format="%.12g")
    info = {
        "recording": recording.name,
        "dataset": dataset,
        "path": str(recording),
        "samplerate_hz": rate,
        "n_samples": n_samples,
        "duration_s": duration,
        "requested_last_s": last_s,
        "covered_s": spans,
        "batches": batches,
        "n_batches_total": len(parser.batches),
        "n_batches_run": n_batches,
        "runtime_previous_s": None
        if previous is None
        else previous["info"].get("runtime_s"),
        "n_boxes_pre_nms": int(sum(len(f) for f in pre)),
        "n_boxes": len(post_df),
        "n_boxes_ge_0.5": int((post_df["score"] >= 0.5).sum()),
        "n_boxes_ge_0.5_assignable": int(
            ((post_df["score"] >= 0.5) & post_df["assignable"].astype(bool)).sum()
        ),
        "max_boxes_per_image": int(max_boxes),
        "capped": bool(max_boxes >= MAX_DET),
        "runtime_s": round(time.perf_counter() - t0, 1),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (d / f"{recording.name}.json").write_text(json.dumps(info, indent=1) + "\n")
    (d / f"{recording.name}.csv.part").rename(d / f"{recording.name}.csv")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=pathlib.Path, help="screen directory")
    ap.add_argument("recordings", nargs="+", type=pathlib.Path)
    ap.add_argument("--dataset", required=True, help="e.g. competition2019")
    ap.add_argument("--config", type=pathlib.Path, default=PRODUCTION_CONFIG)
    ap.add_argument("--weights", type=pathlib.Path, default=WEIGHTS)
    ap.add_argument("--store-threshold", type=float, default=0.05)
    ap.add_argument(
        "--last-s",
        type=float,
        default=None,
        help="only the batches overlapping the last S seconds (default: all)",
    )
    ap.add_argument(
        "--frozen-dir",
        type=pathlib.Path,
        default=None,
        help="where the frozen sources are exported (default OUT/.frozen)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if not 0 < args.store_threshold <= 0.25:
        ap.error("--store-threshold must lie in (0, 0.25]")
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = export_frozen(args.frozen_dir or args.out / ".frozen")

    import torch
    import ultralytics
    from chirpdetector.config import load_config
    from chirpdetector.detection.detection_models import YOLOv8
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    fraction = float(os.environ.get("CHIRP_MAX_VRAM_FRACTION", "0.25"))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
    digest = sha256(args.weights)
    if digest != WEIGHTS_SHA256:
        msg = f"{args.weights} is not the production model (sha256 {digest[:16]})"
        raise RuntimeError(msg)
    cfg = load_config(str(args.config))
    cfg.det.threshold = args.store_threshold
    cfg.hyper.modelpath = str(args.weights.parent)
    detector = YOLOv8(model=_LowConf(YOLO(str(args.weights)), args.store_threshold))
    warnings = _Warnings()
    LOGGER.addHandler(warnings)

    meta = {
        "name": "v1",
        "detector": "chirpdetector v1 (paper-chirp-competition-v1), YOLOv8l",
        "weights": str(args.weights),
        "weights_sha256": digest,
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "frozen": frozen,
        "store_threshold": args.store_threshold,
        "production_threshold": 0.5,
        "nms_iou": NMS_IOU,
        "tables": "post-NMS boxes (greedy NMS per batch, IoU 0.2) with score >= "
        "store_threshold; *.prenms.csv before NMS; assignable = False for boxes "
        "v1's assigner dropped (no peak, dropna)",
        "versions": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "script": str(pathlib.Path(__file__).resolve()),
    }
    meta_path = args.out / "screen.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ("weights_sha256", "store_threshold", "nms_iou", "config_sha256"):
            if old.get(key) != meta[key]:
                msg = f"{meta_path}: {key} differs ({old.get(key)} vs {meta[key]})"
                raise RuntimeError(msg)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")

    for recording in args.recordings:
        if args.overwrite:
            for suffix in (".csv", ".prenms.csv", ".json"):
                (args.out / args.dataset / f"{recording.name}{suffix}").unlink(
                    missing_ok=True
                )
        if not (recording / RAW_FILE).exists():
            log.warning("%s: no %s, skipped", recording, RAW_FILE)
            continue
        info = run_recording(
            recording,
            args.dataset,
            args.out,
            cfg,
            detector,
            args.store_threshold,
            args.last_s,
            warnings,
        )
        log.info(
            "%s: %d boxes (%d >= 0.5) in %d batches, %.0f s",
            recording.name,
            info["n_boxes"],
            info["n_boxes_ge_0.5"],
            info["n_batches_run"],
            info["runtime_s"],
        )


if __name__ == "__main__":
    main()
