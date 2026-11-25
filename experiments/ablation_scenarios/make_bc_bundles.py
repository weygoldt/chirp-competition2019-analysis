"""Bundles of the continued arm B (arm Bc) at fixed epochs, with the emitter unit repaired.

    uv run python experiments/ablation_scenarios/make_bc_bundles.py [--epochs 16 22]

Resuming a dual-head fit (``[model] resume``) leaves the EMA copy of the model
with the default ``emitter_unit = 1.0``: `fit` builds the model and its EMA
before `_load_state` restores ``emitter_unit`` (0.9152 = image / input height,
205 / 224), and sets it on the model only. The EMA is what is validated and
exported, so after the resume every validated metric that uses the emitter
(MAE, assignment, fitness -> best epoch) and every exported checkpoint decodes
the emitter offset 1 / 0.9152 = 1.093x too large. Training is unaffected (the
loss uses its own, correct unit).

Here: for each requested epoch, the snapshot ``epochs/epoch_NN.pt`` (EMA
weights, exported after that epoch by continue_longonly.sh) with only
``emitter_unit`` set to the value the run was trained with (the resume
state's, = the original arm B's and arm A's), in a copy of arm B's released
bundle (same front end, tiling, preprocessing) with the new weights hash, under
``RUN/bundles/epoch_NN``. Epochs: 16 (arm A's best epoch) and 22 (arm A's
last); the run's own best-epoch choice is not used (its fitness was computed
with the wrong unit).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

from chirpdetector.bundle import load_bundle  # noqa: E402
from chirpdetector.data.contract import sha256_file  # noqa: E402

RUN = DATA / "runs/dualhead_r18_sim_v4_longonly_r1c"
SOURCE_BUNDLE = DATA / "runs/dualhead_r18_sim_v4_longonly_r1/final/bundle"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, nargs="+", default=[16, 22])
    args = p.parse_args()
    state = torch.load(RUN / "resume_state_epoch10_patience100.pt", map_location="cpu", weights_only=False)
    unit = float(state["emitter_unit"])
    assert math.isclose(unit, 205 / 224), unit
    del state
    for epoch in args.epochs:
        snap = RUN / f"epochs/epoch_{epoch:02d}.pt"
        weights = torch.load(snap, map_location="cpu", weights_only=False)
        print(f"epoch {epoch}: snapshot emitter_unit {weights['emitter_unit']} -> {unit}")
        weights["emitter_unit"] = unit
        out = RUN / f"bundles/epoch_{epoch:02d}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        torch.save(weights, out / "weights.pt")
        record = json.loads((SOURCE_BUNDLE / "bundle.json").read_text())
        record["weights_sha256"] = sha256_file(out / "weights.pt")
        record["training"] = {
            **record["training"],
            "best_epoch": epoch,
            "epochs_run": epoch,
            "validation_metrics": None,
            "note": (
                f"arm Bc: dualhead_r18_sim_v4_longonly_r1 resumed after epoch 10, epoch {epoch} "
                f"snapshot ({snap}), emitter_unit repaired {unit} (resume left the EMA at 1.0); "
                "experiments/ablation_scenarios/make_bc_bundles.py"
            ),
        }
        (out / "bundle.json").write_text(json.dumps(record, indent=1) + "\n")
        bundle = load_bundle(out)
        print(f"  {out}: weights {bundle.weights_sha256[:12]}")


if __name__ == "__main__":
    main()
