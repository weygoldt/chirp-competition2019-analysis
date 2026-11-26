"""How many chirps of the real dense chirp train (Fig. S4C) a detector finds.

    python experiments/ablation_scenarios/rapid_training/real_train_counts.py BUNDLE --out OUT.json

The rules of Fig. S7A (``paper/scripts/plot_v2_frontend_ablation.py``), in the
root environment: the detector runs with ``predict_span`` over the window of
the failure-mode figure's snippet ``f3_dense`` (``v2_fig_extract.SNIPPETS``)
plus 2 s, on the production tile grid; detections at score >= 0.8 that
post-processing keeps, centred in the window, are assigned to the nearest
track within 20 Hz at their box centre; the fish's chirps are counted without
the detector (peaks of the 51-ms channel's power 40-90 Hz above its EODf,
prominence 6 dB, >= 40 ms apart; the snippet's own spectrogram) and matched
to its detections (box centre within 50 ms, one-to-one, closest first).

Before counting, the released detector is run once and must reproduce the
figure's numbers (109 chirps, 62 matched; ``REFERENCE``), so that this script
and the figure cannot drift apart. The snippet is read from the primary
paper run (``PAPER_SNIPPET``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/: paths.py
from paths import DATA, REC_2019, REPO  # noqa: E402

PAPER_SNIPPET = Path(os.environ.get(
    "CHIRP_PAPER_SNIPPET", REPO / "paper/output/rtdetr_gapfill_t080/data/processed/v2_figures/f3_dense.npz"))
RELEASED = DATA / "runs/dualhead_r18_sim_v4_r1/final/bundle"
REFERENCE = {"n_events": 109, "n_matched": 62}
OPERATING = 0.8
TOLERANCE_HZ = 20.0
MARGIN_S = 2.0
EVENTS = {"channel": 1, "band_above_hz": (40.0, 90.0), "prominence_db": 6.0,
          "min_interval_s": 0.04, "match_s": 0.05}


@cache
def snippet() -> dict:
    z = np.load(PAPER_SNIPPET)
    meta = json.loads(str(z["meta"]))
    off = z["track_offsets"]
    tracks = {int(t): (z["track_times_s"][a:b], z["track_freqs_hz"][a:b])
              for t, a, b in zip(z["track_ids"], off[:-1], off[1:], strict=True)}
    return {"meta": meta, "tracks": tracks, "times_s": z["times_s"], "freqs_hz": z["freqs_hz"],
            "power_db": z["power_db"]}


def chirp_events(snip: dict, t0: float, t1: float, tid: int) -> np.ndarray:
    lo, hi = EVENTS["band_above_hz"]
    t, f = snip["times_s"], snip["freqs_hz"]
    it = (t >= t0) & (t <= t1)
    tt = t[it]
    tr_t, tr_f = snip["tracks"][tid]
    fe = np.interp(tt, tr_t, tr_f)
    power = 10 ** (snip["power_db"][EVENTS["channel"]][:, it].astype(float) / 10)
    level = np.array([10 * np.log10(power[(f >= fe[k] + lo) & (f <= fe[k] + hi), k].mean())
                      for k in range(len(tt))])
    dist = max(1, int(round(EVENTS["min_interval_s"] / (tt[1] - tt[0]))))
    peaks, _ = find_peaks(level, prominence=EVENTS["prominence_db"], distance=dist)
    return tt[peaks]


def match_times(a: np.ndarray, b: np.ndarray, tol: float) -> set[int]:
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if abs(x - y) <= tol)
    ua, ub = set(), set()
    for _, i, j in pairs:
        if i not in ua and j not in ub:
            ua.add(i)
            ub.add(j)
    return ua


def assign(det: pd.DataFrame, snip: dict) -> np.ndarray:
    centre = 0.5 * (det.t_start_s.to_numpy() + det.t_end_s.to_numpy())
    best, dist = np.full(len(det), -1), np.full(len(det), np.inf)
    for tid, (tt, ff) in snip["tracks"].items():
        inside = (centre >= tt[0]) & (centre <= tt[-1])
        d = np.where(inside, np.abs(det.emitter_hz.to_numpy() - np.interp(centre, tt, ff)), np.inf)
        better = d < dist
        best[better], dist[better] = tid, d[better]
    best[dist > TOLERANCE_HZ] = -1
    return best


def detections(bundle_dir: Path, device: str) -> pd.DataFrame:
    from chirpdetector.bundle import load_bundle
    from chirpdetector.inference.spans import predict_span
    from gridtools.io import open_recording

    snip = snippet()
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    bundle = load_bundle(bundle_dir)
    stride = bundle.tiling.tile_duration_s - bundle.tiling.overlap_s
    a = np.floor((t0 - MARGIN_S) / stride) * stride
    recording = open_recording(REC_2019 / m["recording"])
    det = predict_span(bundle, recording, float(a), float(t1 + MARGIN_S), name=m["recording"],
                       score_threshold=0.05, device=device, postprocess=True, keep_dropped=True)
    det = det[(det.drop_reason.astype(str) == "") & (det.score >= OPERATING)].copy()
    c = 0.5 * (det.t_start_s + det.t_end_s)
    det = det[(c >= t0) & (c <= t1)].copy()
    det["track"] = assign(det, snip)
    return det


def counts(det: pd.DataFrame, emitter: int) -> dict:
    snip = snippet()
    t0, t1 = snip["meta"]["t_start_s"], snip["meta"]["t_end_s"]
    events = chirp_events(snip, t0, t1, emitter)
    mine = det[det.track == emitter]
    centres = np.sort(0.5 * (mine.t_start_s + mine.t_end_s).to_numpy())
    matched = match_times(events, centres, EVENTS["match_s"])
    per_box = [int(((events >= r.t_start_s) & (events <= r.t_end_s)).sum()) for r in mine.itertuples()]
    return {"n_events": len(events), "n_detections": len(mine), "n_matched": len(matched),
            "fraction_matched": len(matched) / len(events),
            "n_boxes_spanning_2_or_more": int(sum(p >= 2 for p in per_box)), "emitter_track": emitter}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ref = detections(RELEASED, args.device)
    emitter = int(ref.track.value_counts().drop(-1, errors="ignore").idxmax())
    ref_counts = counts(ref, emitter)
    for k, v in REFERENCE.items():
        if ref_counts[k] != v:
            msg = f"released detector: {k} = {ref_counts[k]}, Fig. S7 has {v}; rules drifted"
            raise RuntimeError(msg)
    out = counts(detections(args.bundle, args.device), emitter)
    out.update({"bundle": str(args.bundle), "released_reference": ref_counts})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
