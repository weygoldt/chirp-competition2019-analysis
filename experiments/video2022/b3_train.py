"""Train a YOLO pose model on an assembled fish dataset (``b3_dataset.py``), with early stopping.

Ultralytics YOLO26 pose, pretrained on COCO keypoints, fine-tuned on the
7-point fish skeleton. Augmentation suited to a top-down view of an arena:
any rotation (``degrees=180``), horizontal and vertical flips (``flip_idx``
swaps the left and right fin), mosaic, scale and brightness; no colour
augmentation beyond brightness (the videos are grey). Early stopping on the
validation fitness (``patience`` epochs without improvement).

    .venv/bin/python experiments/video2022/b3_train.py pose7_r1 --model yolo26x-pose.pt --imgsz 1280

Run it detached through ``scripts/guarded`` (see the README). Output:
``$CHIRP_DATA/video2022/runs/<run>/`` (Ultralytics run dir: weights/best.pt,
results.csv, plots) plus ``provenance.json`` (arguments, dataset summary, git
commit).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA  # noqa: E402

DATASETS = DATA / "video2022" / "datasets"
RUNS = DATA / "video2022" / "runs"
WEIGHTS = DATA / "video2022" / "weights"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--model", default="yolo26x-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=float, default=0.7, help="< 1: fraction of GPU memory (AutoBatch); >= 1: images")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from ultralytics import YOLO

    name = args.name or f"{args.dataset}_{Path(args.model).stem}_{args.imgsz}"
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    weights = WEIGHTS / args.model
    model = YOLO(str(weights) if weights.exists() else args.model)
    run = RUNS / name
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent).stdout.strip()
    run.mkdir(parents=True, exist_ok=True)
    (run / "provenance.json").write_text(json.dumps({
        "args": vars(args), "dataset": str(DATASETS / args.dataset), "git_commit": commit,
        "dataset_summary": (DATASETS / args.dataset / "summary.csv").read_text()}, indent=1))
    batch = int(args.batch) if args.batch >= 1 else args.batch
    model.train(data=str(DATASETS / args.dataset / "dataset.yaml"), imgsz=args.imgsz, epochs=args.epochs,
                patience=args.patience, batch=batch, workers=args.workers, device=0, seed=args.seed,
                project=str(RUNS), name=name, exist_ok=True, degrees=180.0, flipud=0.5, fliplr=0.5,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, mosaic=1.0, scale=0.5, translate=0.1, plots=True,
                cache=False, amp=True)


if __name__ == "__main__":
    import multiprocessing as mp

    # Python 3.14 starts DataLoader workers with "forkserver", which pickles the dataset
    # for every worker; "fork" (the Linux default before 3.14) does not (as paper_config.py)
    mp.set_start_method("fork", force=True)
    main()
