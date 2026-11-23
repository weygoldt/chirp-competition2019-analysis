"""Does the chaser-deceleration cue (``offset_refine.py``) bring the model's chase offsets closer to the author's?

On the labelled clips (``chase_v1``, ``chase_v2``; winner / loser trajectories): for each
clip, the chase model is fitted on the other clips and decoded on it (as in
``c8_chase_model.py``); every labelled chase that a predicted chase overlaps (and that
does not run past the clip's end) gets the predicted offset and the predicted chaser.
The refinement's settings (``GRID``) are chosen on the *other* clips (smallest mean
|error|, errors capped at ``CAP_S`` so that a few far-off chases do not decide alone)
and applied to the held-out clip, so the refined errors are not fitted to the chases
they are measured on.

    .venv/bin/python experiments/video2022/c13_offset_refine.py

Output: ``$CHIRP_DATA/video2022/chases/model/offset_refine/`` (``errors.csv``,
``grid.csv``, ``errors.png``).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c8_chase_model as c8
import chase_model as cm
import offset_refine as orf

OUT = c8.OUT / "offset_refine"
CAP_S = 5.0
GRID = [orf.Refine(delta_s=d, w_chasee=wc, w_open=wo, before_s=b, after_s=a)
        for d, wc, wo, (b, a) in itertools.product((0.3, 0.6, 1.0), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0, 2.0),
                                                     ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (2.0, 4.0)))]


def loco_offsets(clips: list[dict], l2: float = 3.0) -> pd.DataFrame:
    """Per labelled chase (not running past the clip's end) the held-out model's matching chase."""
    rows = []
    dec = cm.Decoder()
    for k, c in enumerate(clips):
        X, y, cols = c8.matrix([x for j, x in enumerate(clips) if j != k])
        model = cm.Logistic(l2).fit(X, y)
        pred = []
        for d in ("A", "B"):
            p = model.predict(c["feats"][d][cols].to_numpy(float))
            pred += [(a, b, d) for a, b in c8.intervals(c["t"], cm.decode(p, dec))]
        for x in c["chases"]:
            if x["continues"]:
                continue
            ov = [(min(b, x["offset_s"]) - max(a, x["onset_s"]), a, b, d) for a, b, d in pred
                  if min(b, x["offset_s"]) > max(a, x["onset_s"])]
            if not ov:
                continue
            _, _, b, d = max(ov)
            rows.append({"clip": k, "clip_id": c["clip_id"], "label_off": x["offset_s"], "model_off": b,
                         "chaser": d, "chaser_labelled": x["chaser"], "contact": x.get("contact", False)})
    return pd.DataFrame(rows)


def refined_errors(clips: list[dict], d: pd.DataFrame, r: orf.Refine) -> np.ndarray:
    out = np.empty(len(d))
    cues: dict = {}
    for i, row in enumerate(d.itertuples()):
        s = clips[row.clip]["s"]
        key = (row.clip, row.chaser)
        if key not in cues:
            cues[key] = orf.cue(s, row.chaser, "B" if row.chaser == "A" else "A", r)
        out[i] = orf.refine(s["t"].to_numpy(), cues[key], row.model_off, r) - row.label_off
    return out


def score(err: np.ndarray) -> float:
    return float(np.mean(np.minimum(np.abs(err), CAP_S)))


def describe(err: np.ndarray) -> str:
    a = np.abs(err)
    return (f"|error| median {np.median(a):.2f} s, 75 % {np.percentile(a, 75):.2f}, 90 % {np.percentile(a, 90):.2f}, "
            f"within 0.5 s {np.mean(a <= 0.5):.0%}, within 1 s {np.mean(a <= 1):.0%}, bias {np.median(err):+.2f} s")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    clips = c8.load_clips(["chase_v1", "chase_v2"], "poses_r3", "identity")
    d = loco_offsets(clips)
    d["err_model"] = d["model_off"] - d["label_off"]
    all_err = np.array([refined_errors(clips, d, r) for r in GRID])  # (grid, chases)
    grid = pd.DataFrame([r.__dict__ | {"score": score(e), "median_abs": float(np.median(np.abs(e)))}
                         for r, e in zip(GRID, all_err, strict=True)]).sort_values("score")
    grid.to_csv(OUT / "grid.csv", index=False)
    # held-out choice: settings chosen on the other clips
    held = np.empty(len(d))
    chosen = []
    for k in d["clip"].unique():
        test = (d["clip"] == k).to_numpy()
        best = int(np.argmin([score(e[~test]) for e in all_err]))
        held[test] = all_err[best][test]
        chosen.append(best)
    d["err_refined"] = held
    d.to_csv(OUT / "errors.csv", index=False)
    print(f"{len(d)} labelled chase offsets matched by the held-out model")
    print("model offsets:          ", describe(d["err_model"].to_numpy()))
    print("refined (held out):     ", describe(held))
    best = GRID[int(np.bincount(chosen).argmax())]
    print("most often chosen:", best)
    print("best on all (in sample):", grid.iloc[0].to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    bins = np.arange(-5, 5.01, 0.25)
    axes[0].hist(np.clip(d["err_model"], -5, 5), bins=bins, color="0.6", label="model offsets")
    axes[0].hist(np.clip(held, -5, 5), bins=bins, histtype="step", color="#b30000", lw=2,
                 label="refined by the chaser's deceleration (held out)")
    axes[0].set_xlabel("offset - author's offset (s; clipped at +-5)")
    axes[0].set_ylabel("chases")
    axes[0].legend(fontsize=7, frameon=False)
    a0, a1 = np.abs(d["err_model"].to_numpy()), np.abs(held)
    axes[1].scatter(a0, a1, s=12, color="k")
    m = max(a0.max(), a1.max()) * 1.05
    axes[1].plot([0, m], [0, m], color="0.6", lw=0.8)
    axes[1].set_xscale("symlog", linthresh=0.5)
    axes[1].set_yscale("symlog", linthresh=0.5)
    axes[1].set_xlabel("|error| of the model offset (s)")
    axes[1].set_ylabel("|error| after refinement (s)")
    fig.suptitle(f"Chase offsets vs the author's ({len(d)} labelled chases, model and settings held out)", fontsize=10)
    fig.savefig(OUT / "errors.png", dpi=150)
    print(OUT / "errors.png")


if __name__ == "__main__":
    main()
