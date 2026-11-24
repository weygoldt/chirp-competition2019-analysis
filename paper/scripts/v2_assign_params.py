"""Simulator truth of the assignment true positives (Fig. 10: assignment by chirp size).

For every true positive of ``assign_baseline.csv`` (``v2_fig_extract.py
assign-baseline``: the post-processed evaluation of the simulated test
scenes, true positives at score >= 0.5, one row per tile-level match) this
writes the simulated chirp's true parameters, so that Fig. 10 can show the
emitter assignment by the chirp's true 10 % width and height.

The join
--------
1. ``label_row`` -> label. The evaluation's ground truth is
   ``chirpdetector.data.dataset_ground_truth`` of the test tiles of
   ``datasets/sim_v4`` (``GroundTruth`` sorts it by tile and box midpoint).
   Every row of the evaluation's ``items.csv`` that names a label is checked
   against it (tile, label kind, box midpoint, emitter); any mismatch stops
   the script.
2. label -> chirp: a tile belongs to one scene (the manifest's source), and
   the label's ``chirp_id`` is the scene's chirp.
3. chirp -> truth: ``chirpdetector.params.truth.scene_truth`` of the scene,
   the simulator's noise-free per-EOD-period frequency of the emitter
   (height: largest per-period excursion above the local baseline; FWHM and
   10 % width: widths at 50 % and 10 % of it; ``edge_clipped``: a width is a
   lower bound). This is the definition the chirp-parameter decoder is
   trained and evaluated on: where the decoder's evaluation
   (``params/runs/.../eval/predictions.parquet``, Fig. 10 decoder row) has the
   same chirp, its ``true_*`` columns must agree to 1e-9.

Output in ``$PAPER_OUT/data/processed/v2_figures/``: ``assign_params.csv``,
one row per true positive of ``assign_baseline.csv``, keyed like it by
``recording`` (tile), ``label_row`` and ``pred_row``; ``assign_params.json``,
the counts of the join.

Environment: the upgrade repository's root venv (chirpdetector, gridtools),
run from ``paper/``::

    /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python \\
        scripts/v2_assign_params.py

CPU only (no model is run): about two minutes for the 285 test scenes that
hold a true positive, about 1.3 GB of memory.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config

log = logging.getLogger("v2_assign_params")

CHIRP_DATA = paper_config.CHIRP_DATA
SIM_V4 = Path(os.environ.get("CHIRP_FAST", str(CHIRP_DATA))).expanduser() / "datasets" / "sim_v4"
PP_EVAL = CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_r1" / "eval_postprocess"
DECODER_EVAL = CHIRP_DATA / "params" / "runs" / "v4-dualhead-sim-v4-r2" / "bundle" / "eval"

KEYS = ["recording", "label_row", "pred_row"]  # a true positive of assign_baseline.csv
TRUTH = ["height_hz", "fwhm_s", "width10_s"]  # scene_truth columns used (s, Hz)
TOL = 1e-9  # agreement with the decoder evaluation's truth


def check_items(items: pd.DataFrame, labels: pd.DataFrame) -> int:
    """Check every item row that names a label against the label table.

    ``items``: the evaluation's ``items.csv`` (``recording``, ``label_row``,
    ``label``, ``label_t_center_s``, ``true_emitter_id``); ``labels``: the
    ground truth's label table in its row order (``recording``, ``label``,
    ``t_center_s``, ``emitter_id``). Raises ValueError on any mismatch;
    returns the number of rows checked.
    """
    it = items[items["label_row"].notna()]
    rows = it["label_row"].astype(int).to_numpy()
    if len(rows) and (rows.min() < 0 or rows.max() >= len(labels)):
        raise ValueError(f"label_row outside the {len(labels)} labels")
    lab = labels.iloc[rows]
    checks = {
        "tile": lab["recording"].astype(str).to_numpy() == it["recording"].astype(str).to_numpy(),
        "label kind": lab["label"].astype(str).to_numpy() == it["label"].astype(str).to_numpy(),
        "box midpoint": np.isclose(lab["t_center_s"].to_numpy(float),
                                   it["label_t_center_s"].to_numpy(float), rtol=0, atol=1e-6),
        "emitter": (pd.array(lab["emitter_id"], dtype="Int64").fillna(-1).to_numpy(int)
                    == pd.array(it["true_emitter_id"], dtype="Int64").fillna(-1).to_numpy(int)),
    }
    for what, ok in checks.items():
        if not ok.all():
            raise ValueError(f"{what} differs for {int((~ok).sum())} of {len(ok)} label rows "
                             "of items.csv: the ground truth is not the evaluation's")
    return len(it)


def tp_chirps(tp: pd.DataFrame, labels: pd.DataFrame, sources: Mapping[str, str]) -> pd.DataFrame:
    """Scene and chirp of each true positive.

    ``tp``: rows of ``assign_baseline.csv`` (``KEYS``, ``true_emitter_id``);
    ``labels``: the label table (``recording``, ``label``, ``emitter_id``,
    ``chirp_id``); ``sources``: tile -> scene. Returns ``KEYS``,
    ``source_id``, ``chirp_id``; raises if a true positive's label is not a
    chirp of its tile with its emitter.
    """
    lab = labels.iloc[tp["label_row"].astype(int).to_numpy()]
    bad = (
        (lab["recording"].astype(str).to_numpy() != tp["recording"].astype(str).to_numpy())
        | (lab["label"].astype(str).to_numpy() != "chirp")
        | (pd.array(lab["emitter_id"], dtype="Int64").fillna(-1).to_numpy(int)
           != pd.array(tp["true_emitter_id"], dtype="Int64").fillna(-1).to_numpy(int))
    )
    if bad.any():
        raise ValueError(f"{int(bad.sum())} true positives do not point to a chirp label "
                         "of their tile and emitter")
    missing = sorted(set(tp["recording"].astype(str)) - set(sources))
    if missing:
        raise ValueError(f"tiles without a scene: {missing[:5]}")
    out = tp[KEYS].copy()
    out["source_id"] = tp["recording"].astype(str).map(sources).to_numpy()
    out["chirp_id"] = lab["chirp_id"].astype(int).to_numpy()
    return out


def join_truth(chirps: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """The true positives' chirps with their truth.

    ``truth``: one row per (``source_id``, ``chirp_id``) with ``TRUTH`` and
    ``edge_clipped``. Every true positive must find its chirp (a missing
    chirp raises; NaN truth, e.g. a chirp that never rises above baseline,
    is kept and counted by the caller).
    """
    if truth.duplicated(["source_id", "chirp_id"]).any():
        raise ValueError("truth has duplicate (source_id, chirp_id)")
    out = chirps.merge(truth[["source_id", "chirp_id", *TRUTH, "edge_clipped"]],
                       on=["source_id", "chirp_id"], how="left", validate="many_to_one",
                       indicator=True)
    lost = out["_merge"] != "both"
    if lost.any():
        raise ValueError(f"{int(lost.sum())} true positives without a truth row")
    return out.drop(columns="_merge")


def check_decoder(truth: pd.DataFrame, decoder: pd.DataFrame) -> dict:
    """Compare with the decoder evaluation's truth (``true_<param>`` columns).

    Returns the number of shared chirps and the largest absolute difference
    per parameter; raises if any exceeds ``TOL``.
    """
    m = truth.merge(decoder[["source_id", "chirp_id", *(f"true_{p}" for p in TRUTH)]],
                    on=["source_id", "chirp_id"], how="inner", validate="one_to_one")
    out: dict = {"n_shared": int(len(m))}
    for p in TRUTH:
        a, b = m[p].to_numpy(float), m[f"true_{p}"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if (np.isfinite(a) != np.isfinite(b)).any():
            raise ValueError(f"{p}: finite here but NaN in the decoder evaluation (or vice versa)")
        diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        if diff > TOL:
            raise ValueError(f"{p}: differs from the decoder evaluation's truth by {diff:g}")
        out[f"max_abs_diff_{p}"] = diff
    return out


# ---------------------------------------------------------------------------
# root environment (chirpdetector, gridtools)
# ---------------------------------------------------------------------------
def ground_truth():
    """The evaluation's ground truth of the test tiles (its label table), tile -> scene,
    and scene -> scene directory."""
    from chirpdetector.data import load_dataset
    from chirpdetector.data.contract import dataset_ground_truth

    dataset = load_dataset(SIM_V4, check_files=False)
    ids = list(dataset.manifest.image_ids("test"))
    gt = dataset_ground_truth(dataset, ids)
    sources = {i: dataset.manifest.image(i).source for i in ids}
    scene_dirs = {s: Path(dataset.manifest.sources[s].extra["scene_dir"])
                  for s in set(sources.values())}
    return gt.labels, sources, scene_dirs


def scene_truths(scene_dirs: Mapping[str, Path]) -> pd.DataFrame:
    """``scene_truth`` of every chirp of the scenes (one row per chirp)."""
    from chirpdetector.params.truth import TRUTH_PAD_S, scene_truth

    frames = []
    for i, (source, directory) in enumerate(sorted(scene_dirs.items())):
        t = scene_truth(directory, pad_s=TRUTH_PAD_S)
        t.insert(0, "source_id", source)
        frames.append(t)
        if (i + 1) % 50 == 0:
            log.info("truth of %d / %d scenes", i + 1, len(scene_dirs))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out_dir = paper_config.processed_dir("v2_figures")
    tp = pd.read_csv(out_dir / "assign_baseline.csv")
    items = pd.read_csv(PP_EVAL / "sim_test_after" / "items.csv")
    labels, sources, scene_dirs = ground_truth()
    n_checked = check_items(items, labels)
    log.info("%d label rows of items.csv agree with the ground truth (%d labels)",
             n_checked, len(labels))
    chirps = tp_chirps(tp, labels, sources)
    wanted = {s: scene_dirs[s] for s in chirps["source_id"].unique()}
    truth = scene_truths(wanted)
    out = join_truth(chirps, truth)
    decoder = check_decoder(truth, pd.read_parquet(DECODER_EVAL / "predictions.parquet"))
    log.info("truth agrees with the decoder evaluation on %d shared chirps", decoder["n_shared"])
    finite = np.isfinite(out[TRUTH].to_numpy(float)).all(axis=1)
    meta = {
        "n_tp": int(len(tp)),
        "n_with_truth": int(finite.sum()),
        "n_truth_nan": int((~finite).sum()),
        "n_edge_clipped": int(out["edge_clipped"].astype(bool).sum()),
        "n_chirps": int(chirps[["source_id", "chirp_id"]].drop_duplicates().shape[0]),
        "n_scenes": len(wanted),
        "n_item_rows_checked": n_checked,
        "decoder_check": decoder,
    }
    out.to_csv(out_dir / "assign_params.csv", index=False)
    (out_dir / "assign_params.json").write_text(json.dumps(meta, indent=1) + "\n")
    log.info("wrote %s: %s", out_dir / "assign_params.csv", meta)


if __name__ == "__main__":
    main()
