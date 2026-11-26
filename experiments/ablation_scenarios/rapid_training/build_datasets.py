"""Detector datasets of train_v4r in both front ends (front-end ablation round 2).

    scripts/guarded --name v4r-build --mem 24G --vram 0.3 -- python \
        experiments/ablation_scenarios/rapid_training/build_datasets.py A|B [--workers 8]

``$CHIRP_DATA/datasets/sim_v4r`` (arm A: the sim_v4 front end, 0.2048 / 0.0512 /
0.0128 s windows) and ``sim_v4r_longonly`` (arm B: the 0.2048 s window in all
three channels, as ``experiments/ablation/build_longonly.py``), from the scenes
of ``$CHIRP_DATA/scenes/train_v4r/{train,val,test}`` (``prepare_scenes.sh``),
everything else the `SceneDatasetConfig` defaults, as sim_v4. The front end and
tiling are checked against sim_v4 / sim_v4_longonly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/: paths.py
from paths import DATA, FAST  # noqa: E402

from chirpdetector.data import load_dataset  # noqa: E402
from chirpdetector.data.scenes import (  # noqa: E402
    SceneDatasetConfig,
    SplitRule,
    build_scene_dataset,
    find_scenes,
)
from chirpdetector.data.tiles import TileLayout  # noqa: E402
from chirpdetector.spectrogram import SpectrogramConfig  # noqa: E402

SCENES = DATA / "scenes/train_v4r"
SPLIT_DIRS = [f"{s}={SCENES / s}" for s in ("train", "val", "test")]
OUT = {"A": DATA / "datasets/sim_v4r", "B": DATA / "datasets/sim_v4r_longonly"}
REFERENCE = {"A": FAST / "datasets/sim_v4", "B": FAST / "datasets/sim_v4_longonly"}


def config(arm: str, workers: int, threads: int, device: str) -> SceneDatasetConfig:
    rule = SplitRule.from_roots(SPLIT_DIRS)
    if arm == "A":
        return SceneDatasetConfig(split=rule, workers=workers, threads_per_worker=threads, device=device)  # pyright: ignore[reportArgumentType]
    return SceneDatasetConfig(
        spectrogram=SpectrogramConfig(extra_windows_s=()),
        layout=TileLayout(tile_duration_s=3.2, overlap_s=1.0, channel_fill="reference"),
        split=rule,
        workers=workers,
        threads_per_worker=threads,
        device=device,  # pyright: ignore[reportArgumentType]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("A", "B"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    dirs = sorted(d for s in ("train", "val", "test") for d in find_scenes(SCENES / s))
    dataset = build_scene_dataset(
        dirs,
        OUT[args.arm],
        config(args.arm, args.workers, args.threads, args.device),
        overwrite=args.overwrite,
        provenance={
            "command": sys.argv,
            "split_dirs": SPLIT_DIRS,
            "ablation": "front-end ablation round 2: train_v4 + rapid-chirping scenes "
            "(experiments/ablation_scenarios/rapid_training); "
            + ("arm A (3 windows)" if args.arm == "A" else "arm B (long window x3)"),
        },
    )
    ref = load_dataset(REFERENCE[args.arm], check_files=False).manifest
    m = dataset.manifest
    for name in ("spectrogram", "tiling"):
        assert getattr(m, name) == getattr(ref, name), f"{name} differs from {REFERENCE[args.arm]}"
    counts = {s: len(m.image_ids(split=s)) for s in ("train", "val", "test")}
    print(f"wrote {len(m.images)} images of {len(dirs)} scenes to {OUT[args.arm]} ({counts}); "
          f"sha256 {m.content_sha256[:12]}")


if __name__ == "__main__":
    main()
