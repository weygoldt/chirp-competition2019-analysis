"""Build sim_v4_longonly: sim_v4 with the long window in all three channels.

Front-end ablation (plan, decision 6), arm B. Identical to the sim_v4 build
(`chirpdetector data build-scenes --split-dir ... --workers 8 --threads 2
--device cuda`, experiments/README.md step 5) except the front end:

- sim_v4:          extra_windows_s = (0.0512, 0.0128), channel_fill = null
                   (channels = 0.2048 / 0.0512 / 0.0128 s windows)
- sim_v4_longonly: extra_windows_s = (),               channel_fill = "reference"
                   (channels = 0.2048 s window, copied to channels 1 and 2:
                   the v1 convention of identical channels)

Everything else is the default of `SceneDatasetConfig`, as in the sim_v4
build: reference window, hop, band, aggregation, normalisation (measured on
channel 0 in both), click removal, tiling (3.2 s, 1.0 s overlap), edge
fraction, harmonic ignores, footprint boxes (computed on the reference
window, stored in the scenes). The CLI exposes no front-end options, hence
this script.

    scripts/guarded --name ablation-build --mem 24G --vram 0.4 -- uv run python \
        experiments/ablation/build_longonly.py --out $CHIRP_FAST/datasets/sim_v4_longonly
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

from chirpdetector.data.scenes import (  # noqa: E402
    SceneDatasetConfig,
    SplitRule,
    build_scene_dataset,
    find_scenes,
)
from chirpdetector.data.tiles import TileLayout  # noqa: E402
from chirpdetector.spectrogram import SpectrogramConfig  # noqa: E402

SPLIT_DIRS = [
    f"train={DATA / 'scenes/train_v4/train'}",
    f"val={DATA / 'scenes/train_v4/val'}",
    f"test={DATA / 'scenes/train_v4/test'}",
]


def config(workers: int, threads: int, device: str) -> SceneDatasetConfig:
    rule = SplitRule.from_roots(SPLIT_DIRS)
    return SceneDatasetConfig(
        spectrogram=SpectrogramConfig(extra_windows_s=()),
        layout=TileLayout(tile_duration_s=3.2, overlap_s=1.0, channel_fill="reference"),
        split=rule,
        workers=workers,
        threads_per_worker=threads,
        device=device,  # pyright: ignore[reportArgumentType]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="First N scenes (trial).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = config(args.workers, args.threads, args.device)
    roots = (cfg.split.roots or {}).values()
    dirs = sorted(d for root in roots for d in find_scenes(root))
    if args.limit is not None:
        dirs = dirs[: args.limit]
    dataset = build_scene_dataset(
        dirs,
        args.out,
        cfg,
        overwrite=args.overwrite,
        provenance={
            "command": sys.argv,
            "split_dirs": SPLIT_DIRS,
            "ablation": "front end: long window only, replicated to 3 channels "
            "(decision 6; arm B of experiments/ablation)",
        },
    )
    m = dataset.manifest
    counts = {s: len(m.image_ids(split=s)) for s in ("train", "val", "test")}
    print(
        f"wrote {len(m.images)} images of {len(dirs)} scenes to {args.out} "
        f"({counts}); content sha256 {m.content_sha256[:12]}"
    )


if __name__ == "__main__":
    main()
