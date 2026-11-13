"""v2 detectors on benchmark scenes: dual-head (production post-processing,
re-attribution on) and YOLO26s sim_v4, every box >= 0.05 with drop reasons.

    run_models_v2.py MODEL SCENE_DIR...   (MODEL: dualhead | yolo26s)

One parquet per scene in models/<MODEL>/<scene>.parquet (skipped if
present): `predict_span` with the scene's tracks, post-processing on
(duplicates, harmonic copies; re-attribution automatic: on for the
emitter head), ``keep_dropped=True`` (dropped rows carry ``drop_reason``).
Run through scripts/guarded --vram 0.2.
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
from chirpdetector.prediction import load_detector

BUNDLES = {
    "dualhead": DATA / "runs/dualhead_r18_sim_v4_r1/final/bundle",
    "yolo26s": DATA / "runs/yolo26s_sim_v4_r1/final/bundle",
}
OUT = DATA / "labels/benchmark/models"
THRESHOLD = 0.05


def main() -> None:
    model, scenes = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(BUNDLES[model])
    detector = load_detector(bundle, device="cuda")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "bundle": str(BUNDLES[model]),
                "score_threshold": THRESHOLD,
                "postprocess": "defaults (re-attribution automatic)",
                "keep_dropped": True,
                "tracks": "scene wavetracker files",
            },
            indent=1,
        )
    )
    tic = time.time()
    todo = [s for s in scenes if not (out / f"{s.name}.parquet").exists()]
    for k, scene in enumerate(todo):
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
            tracks=load_tracks(scene),
            postprocess=True,
            keep_dropped=True,
        )
        provenance = det.attrs.get("provenance", {})
        det.attrs = {}
        path = out / f"{scene.name}.parquet"
        tmp = path.with_suffix(".tmp")
        det.to_parquet(tmp, index=False)
        tmp.replace(path)
        if k == 0:
            (out / "provenance_first_scene.json").write_text(json.dumps(provenance, indent=1, default=str))
        print(f"{model} {k + 1}/{len(todo)} {scene.name}: {len(det)} rows, {time.time() - tic:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
