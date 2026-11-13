"""Exploration: emitter errors, harmonic copies and duplicates on sim_v4 test.

Classifies every prediction of a model on the sim_v4 test tiles by its best
2-D IoU ground-truth box (chirp / harmonic / invisible / edge / none) and
writes one row per prediction with the live track frequencies at its centre.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/: paths.py
from paths import FAST  # noqa: E402


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gridtools.io import load_tracks

from chirpdetector.data import load_dataset

DS = FAST / "datasets/sim_v4"
pred_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])

ds = load_dataset(DS, check_files=False)
ids = list(ds.manifest.image_ids("test"))
boxes = ds.boxes[ds.boxes["image_id"].isin(set(ids))].copy()
boxes["cls"] = np.where(boxes["kind"] == "chirp", "chirp", boxes["reason"].astype(str))
pred = pd.read_parquet(pred_path).reset_index(drop=True)
pred["row"] = np.arange(len(pred))
cache, tracks = {}, {}
for i in ids:
    s = ds.manifest.image(i).source
    if s not in cache:
        cache[s] = load_tracks(ds.tracks_dir(s))
    tracks[i] = cache[s]

rows = []
bgroups = dict(tuple(boxes.groupby("image_id")))
for rec, g in pred.groupby("recording"):
    tr = tracks[rec]
    tc = 0.5 * (g.t_start_s.to_numpy() + g.t_end_s.to_numpy())
    tids = tr.ids
    F = np.stack([tr.track(int(k)).frequency_at(tc, 1.0) for k in tids], axis=1)
    b = bgroups.get(rec)
    for j, (idx, p) in enumerate(g.iterrows()):
        best, cls, chirp_id, em_id, em_hz = 0.0, "none", -1, -1, np.nan
        if b is not None:
            it = np.clip(np.minimum(p.t_end_s, b.t_end_s) - np.maximum(p.t_start_s, b.t_start_s), 0, None)
            iff = np.clip(np.minimum(p.f_high_hz, b.f_high_hz) - np.maximum(p.f_low_hz, b.f_low_hz), 0, None)
            inter = it * iff
            area = (p.t_end_s - p.t_start_s) * (p.f_high_hz - p.f_low_hz) + (b.t_end_s - b.t_start_s) * (b.f_high_hz - b.f_low_hz) - inter
            iou = (inter / area).to_numpy()
            k = int(np.argmax(iou))
            if iou[k] > 0:
                best = float(iou[k]); bb = b.iloc[k]
                cls, chirp_id, em_id, em_hz = bb.cls, int(bb.chirp_id), int(bb.emitter_id), float(bb.emitter_hz)
        rows.append({"row": p.row, "recording": rec, "t_c": tc[j], "iou": best, "cls": cls,
                     "chirp_id": chirp_id, "true_emitter_id": em_id, "true_emitter_hz": em_hz,
                     "live_hz": F[j][np.isfinite(F[j])].tolist(), "live_ids": tids[np.isfinite(F[j])].tolist()})
info = pd.DataFrame(rows)
out = pred.merge(info, on=["row", "recording"])
out.to_parquet(out_path)
print(out.cls.value_counts())
