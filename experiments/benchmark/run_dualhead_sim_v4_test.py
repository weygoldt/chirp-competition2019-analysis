"""Dual-head detections (score >= 0.05, post-filter on, tracks, decoder) on sim_v4 test.

One parquet per scene in OUT/<scene>.parquet (skipped if present), plus
OUT/provenance.json. Input of `chirpdetector label-benchmark build`.

    scripts/guarded --name hb-model --mem 16G --vram 0.3 -- \
      env PYTHONPATH=<chirpdetector>/src uv run python run_dualhead_sim_v4_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402


import json
import sys
import time
from pathlib import Path

from gridtools.io import load_tracks, open_recording

from chirpdetector.bundle import load_bundle
from chirpdetector.inference.spans import predict_span
from chirpdetector.params.inference import ParamsEstimator
from chirpdetector.prediction import load_detector

SCENES = DATA / "scenes/train_v4/test"
BUNDLE = DATA / "runs/dualhead_r18_sim_v4_r1/final/bundle"
PARAMS = DATA / "params/runs/v4-dualhead-sim-v4-r2/bundle"
OUT = DATA / "labels/benchmark/model_dualhead_sim_v4_test"
THRESHOLD = 0.05


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(BUNDLE)
    detector = load_detector(bundle, device="cuda")
    params = ParamsEstimator(PARAMS, device="cuda")
    scenes = sorted(d for d in SCENES.iterdir() if (d / "chirps.parquet").exists())
    provenance = {
        "bundle": str(BUNDLE),
        "params": str(PARAMS),
        "score_threshold": THRESHOLD,
        "postprocess": True,
        "tracks": "scene wavetracker files",
        "scenes": str(SCENES),
        "n_scenes": len(scenes),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=1))
    tic = time.time()
    for k, scene in enumerate(scenes):
        path = OUT / f"{scene.name}.parquet"
        if path.exists():
            continue
        rec = open_recording(scene)
        det = predict_span(
            bundle,
            rec,
            0.0,
            rec.duration_s,
            name=scene.name,
            score_threshold=THRESHOLD,
            detector=detector,
            device="cuda",
            params=params,
            tracks=load_tracks(scene),
            postprocess=True,
        )
        det.attrs = {}
        tmp = path.with_suffix(".tmp")
        det.to_parquet(tmp, index=False)
        tmp.replace(path)
        print(f"{k + 1}/{len(scenes)} {scene.name}: {len(det)} detections, "
              f"{time.time() - tic:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
