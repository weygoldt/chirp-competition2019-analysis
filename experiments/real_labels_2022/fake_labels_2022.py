"""FAKE labels for a smoke test of score_2022.py (never real results).

Drives the labeller's session code (the same code the browser uses) over a
trial build and stores machine-made decisions under the labeller name
``FAKE`` in a scratch directory: a candidate is a chirp with probability
equal to its score (clicking the live track nearest the model's emitter); a
window gets a mark at every kept detection with score >= ``--mark-score`` in
its zone, minus a random ``--miss`` share, plus a mark at a random time in
``--extra`` of the windows (a chirp the detector missed). The scorer's
numbers from these labels mean nothing; the point is that every code path
runs. Refuses to write anywhere under ``$CHIRP_DATA/labels``.

    .venv/bin/python experiments/real_labels_2022/build_2022.py --dest /tmp/rv2_test
    .venv/bin/python experiments/real_labels_2022/fake_labels_2022.py --root /tmp/rv2_test/real_v2_2022 \
        --candidates /tmp/rv2_test/candidates/real_v2_2022.parquet --out /tmp/rv2_test/labels_fake
    .venv/bin/python experiments/real_labels_2022/score_2022.py --labels /tmp/rv2_test/labels_fake \
        --truth /tmp/rv2_test/real_v2_2022/truth/real_v2_2022_truth.parquet --out /tmp/rv2_test/report_fake
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design2022 as d

sys.path.insert(1, str(d.REAL_V1_CODE))
from common import RUN_2022, view_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="trial build's session root (truth/)")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mark-score", type=float, default=0.5)
    ap.add_argument("--miss", type=float, default=0.2)
    ap.add_argument("--extra", type=float, default=0.05)
    ap.add_argument("--skip", type=float, default=0.03, help="share of items left unlabelled")
    args = ap.parse_args()
    labels_root = (d.DATA / "labels").resolve()
    for p in (args.out, args.root, args.candidates):
        if p.resolve().is_relative_to(labels_root):
            raise SystemExit(f"refusing: {p} is under {labels_root} (fake labels go to a scratch copy)")
    from chirpdetector.labeling.candidates import load_or_build_candidate_queue
    from chirpdetector.labeling.queue import QueueConfig
    from chirpdetector.labeling.session import LabelSession
    from chirpdetector.labeling.store import LabelStore

    rng = np.random.default_rng(1)
    truth = pd.read_parquet(args.root / "truth" / f"{d.SESSION}_truth.parquet").set_index("candidate_id")
    queue = load_or_build_candidate_queue(args.candidates, QueueConfig(), args.out, shuffle=False)
    cache: dict[str, pd.DataFrame] = {}
    with LabelStore(args.out, "FAKE", "fake") as store:
        s = LabelSession(queue, store, config=view_config(), frontend="browser", blind=True,
                         background=False)
        for k, item in enumerate(queue.items):
            if rng.random() < args.skip:
                continue
            t = truth.loc[item.item_id.split("#", 1)[1]]
            tracks = s.tracks(k)
            if item.is_window:
                rec = t["recording"]
                if rec not in cache:
                    cache[rec] = pd.read_parquet(RUN_2022 / rec / "chirps_rtdetr.parquet",
                                                 columns=["t_center_s", "score", "kept", "emitter_hz"])
                det = cache[rec]
                det = det[det["kept"] & (det["score"] >= args.mark_score)
                          & (det["t_center_s"] >= item.t_start_s) & (det["t_center_s"] <= item.t_end_s)]
                times = [(r.t_center_s, r.emitter_hz) for r in det.itertuples() if rng.random() >= args.miss]
                if rng.random() < args.extra:
                    times.append((float(rng.uniform(item.t_start_s, item.t_end_s)), np.nan))
                for tc, f in times:
                    if not tracks:
                        break
                    f = f if np.isfinite(f) else tracks[0].frequency_at(tc)
                    tr = min(tracks, key=lambda x, tc=tc, f=f: abs(x.frequency_at(tc) - f))
                    s.click(k, tc, tr.frequency_at(tc), elapsed_s=1.0)
                s.done(k, elapsed_s=3.0)
                continue
            if rng.random() < float(t["score"]) and tracks:
                f = t["emitter_hz"] if np.isfinite(t["emitter_hz"]) else tracks[0].baseline_hz
                tr = min(tracks, key=lambda x: abs(x.frequency_at(item.t_center_s) - f))
                r = s.click(k, item.t_center_s, tr.frequency_at(item.t_center_s), elapsed_s=1.0)
                if not r.ok:
                    s.mark(k, "chirp_emitter_unclear", elapsed_s=1.0)
            else:
                s.mark(k, "not_chirp", elapsed_s=1.0)
        print(f"FAKE labels: {s.progress()}")
        s.shutdown()


if __name__ == "__main__":
    main()
