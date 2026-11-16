"""Front-end ablation: experiments/benchmark/run_models_v2.py with the bundle
and the output directory as arguments (otherwise unchanged), so that arms A
and B run on the blind-benchmark scenes with the current code.

    run_benchmark_models.py NAME BUNDLE OUT_DIR SCENE_DIR...

One parquet per scene in OUT_DIR/<scene>.parquet (skipped if
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

THRESHOLD = 0.05


def main() -> None:
    model, bundle_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    scenes = [Path(p) for p in sys.argv[4:]]
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(bundle_path)
    detector = load_detector(bundle, device="cuda")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "weights_sha256": bundle.weights_sha256,
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
