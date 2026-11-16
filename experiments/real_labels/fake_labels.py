"""FAKE labels for a smoke test of score_session.py (never real results).

Drives the labeller's session (the same code the browser uses) over a
session build and stores machine-made decisions in a separate output
directory: a candidate is called a chirp with probability equal to its
score (clicking the live track nearest the model's emitter), a window gets
a mark at every kept detection with score >= 0.5 in its zone. The numbers a
scorer computes from these labels mean nothing; the point is that every
code path runs.

    uv run python experiments/real_labels/fake_labels.py --root /tmp/real_v1_test/real_v1 \
        --candidates /tmp/real_v1_test/candidates/real_v1.parquet --out /tmp/real_v1_test/labels_fake
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUN_2019, RUN_2022, view_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if "real_v1_labels" in str(args.out):
        raise SystemExit("refusing to write fake labels into the session's label directory")
    from chirpdetector.labeling.candidates import load_or_build_candidate_queue
    from chirpdetector.labeling.queue import QueueConfig
    from chirpdetector.labeling.session import LabelSession
    from chirpdetector.labeling.store import LabelStore

    rng = np.random.default_rng(1)
    truth = pd.read_parquet(args.root / "truth" / "real_v1_truth.parquet").set_index("candidate_id")
    queue = load_or_build_candidate_queue(args.candidates, QueueConfig(), args.out, shuffle=False)
    runs = {"competition2019": RUN_2019, "competition2022": RUN_2022}
    cache: dict[str, pd.DataFrame] = {}
    with LabelStore(args.out, "FAKE", "fake") as store:
        s = LabelSession(queue, store, config=view_config(), frontend="browser", blind=True,
                         background=False)
        for k, item in enumerate(queue.items):
            cid = item.item_id.split("#", 1)[1]
            t = truth.loc[cid]
            tracks = s.tracks(k)
            if item.is_window:
                rec = t["recording"]
                if rec not in cache:
                    cache[rec] = pd.read_parquet(runs[t["dataset"]] / rec / "chirps_rtdetr.parquet",
                                                 columns=["t_center_s", "score", "kept", "emitter_hz"])
                d = cache[rec]
                d = d[d["kept"] & (d["score"] >= 0.5) & (d["t_center_s"] >= item.t_start_s)
                      & (d["t_center_s"] <= item.t_end_s)]
                for r in d.itertuples():
                    if not tracks:
                        break
                    tr = min(tracks, key=lambda x: abs(x.frequency_at(r.t_center_s) - (r.emitter_hz or 0)))
                    s.click(k, r.t_center_s, tr.frequency_at(r.t_center_s), elapsed_s=1.0)
                s.done(k, elapsed_s=5.0)
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
