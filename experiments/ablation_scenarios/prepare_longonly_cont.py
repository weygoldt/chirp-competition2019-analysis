"""Prepare arm B's continuation to arm A's training budget (22 epochs).

    uv run python experiments/ablation_scenarios/prepare_longonly_cont.py

Arm B (dualhead_r18_sim_v4_longonly_r1) was early-stopped after epoch 10
(best epoch 4, patience 6); arm A (the released model) ran 22 epochs (best 16).
On the validation set A's emitter error only fell below B's in epochs 11-16,
in the low-learning-rate part of the cosine schedule (epochs = 30) that B
never reached. To compare the front ends at an equal training budget, B is
resumed from its own ``final/dualhead/state.pt`` (epoch 10: weights, EMA,
optimizer, RNG states, the same cosine schedule) and continued through
epoch 22.

``[model] resume`` refuses any config change, and the restored stale counter
(6) would stop the fit after one epoch, so this script writes a COPY of the
state with ``optim.patience = 100`` (and nothing else changed) into the new
run dir, plus the matching config (the arm-B config with patience 100 and
``resume`` set). The original run directory is not touched.
``continue_longonly.sh`` then runs the fit, snapshots ``last.pt`` after
every epoch and stops the fit once epoch 22 is validated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA, REPO  # noqa: E402

SRC = DATA / "runs/dualhead_r18_sim_v4_longonly_r1"
RUN = DATA / "runs/dualhead_r18_sim_v4_longonly_r1c"
CONFIG_IN = REPO / "experiments/ablation/dualhead_r18_sim_v4_longonly.toml"
CONFIG_OUT = Path(__file__).with_name("dualhead_r18_sim_v4_longonly_cont.toml")
PATIENCE = 100


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    state_out = RUN / "resume_state_epoch10_patience100.pt"
    state = torch.load(SRC / "final/dualhead/state.pt", map_location="cpu", weights_only=False)
    print("source state: epoch", state["epoch"], "best", state["best_epoch"], "stale", state["stale"])
    print("optim:", state["config"]["optim"])
    assert state["epoch"] == 10 and state["best_epoch"] == 4
    state["config"]["optim"]["patience"] = PATIENCE
    torch.save(state, state_out)

    text = CONFIG_IN.read_text()
    assert text.count("patience = 6\n") == 1
    text = text.replace("patience = 6\n", f"patience = {PATIENCE}\n")
    assert text.count('type = "dualhead"\n') == 1
    text = text.replace('type = "dualhead"\n', f'type = "dualhead"\nresume = "{state_out}"\n')
    header = (
        "# Arm B continued to arm A's budget (experiments/ablation_scenarios/\n"
        "# prepare_longonly_cont.py): resumed after epoch 10 from a copy of\n"
        "# dualhead_r18_sim_v4_longonly_r1's state.pt with patience 100; stopped\n"
        "# after epoch 22 by continue_longonly.sh. Below: the arm-B config with\n"
        "# patience 6 -> 100 and [model] resume added.\n"
    )
    CONFIG_OUT.write_text(header + text)
    print("wrote", state_out, "and", CONFIG_OUT)


if __name__ == "__main__":
    main()
