"""Detector datasets of the breakdown benchmark, in arm A's and arm B's front end.

    scripts/guarded --name bench-build --mem 24G --vram 0.2 -- python \
        experiments/ablation_scenarios/bench_build.py A|B [--workers 6]

All scenes of $CHIRP_DATA/scenes/ablation_bench_v1 become the test split of
$CHIRP_DATA/datasets/ablation_bench_v1_{A,B}:

- A: the sim_v4 front end (0.2048 / 0.0512 / 0.0128 s windows in channels 0-2),
- B: the sim_v4_longonly front end (0.2048 s window in all three channels,
  experiments/ablation/build_longonly.py),

everything else the `SceneDatasetConfig` defaults, as in both training sets.
The front end and tiling written to the manifest are checked against the
training dataset of the arm (sim_v4 / sim_v4_longonly) before returning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
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

SCENES = DATA / "scenes/ablation_bench_v1"
REFERENCE = {"A": FAST / "datasets/sim_v4", "B": FAST / "datasets/sim_v4_longonly"}


def config(arm: str, workers: int, threads: int, device: str) -> SceneDatasetConfig:
    rule = SplitRule.from_roots([f"test={SCENES}"])
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
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = args.out or DATA / f"datasets/ablation_bench_v1_{args.arm}"
    dirs = find_scenes(SCENES)
    dataset = build_scene_dataset(
        dirs,
        out,
        config(args.arm, args.workers, args.threads, args.device),
        overwrite=args.overwrite,
        provenance={
            "command": sys.argv,
            "benchmark": "front-end breakdown benchmark v1 (experiments/ablation_scenarios)",
            "front_end": "arm A (3 windows)" if args.arm == "A" else "arm B (long window x3)",
        },
    )
    ref = load_dataset(REFERENCE[args.arm], check_files=False).manifest
    m = dataset.manifest
    for name in ("spectrogram", "tiling"):
        a, b = getattr(m, name), getattr(ref, name)
        assert a == b, f"{name} differs from {REFERENCE[args.arm]}: {a} vs {b}"
    print(f"wrote {len(m.images)} images of {len(dirs)} scenes to {out}; front end = {REFERENCE[args.arm].name}; sha256 {m.content_sha256[:12]}")


if __name__ == "__main__":
    main()
