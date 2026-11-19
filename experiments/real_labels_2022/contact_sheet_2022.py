"""Contact sheet of 12 items as the blind browser labeller shows them (real_v2_2022).

9 candidates and 3 recall windows drawn from the session's own sampling frame
(same pool, same window slots) but NOT from its queue or real_v1's, so that
the author sees no item before labelling it. Each item is rendered through
the labeller's server code (``LabelServer.view`` / ``LabelServer.image``: the
PNG panels and colours the page receives, with the session's settings) and
drawn with the page's overlays by real_v1's ``contact_sheet.draw_item``.

    .venv/bin/python experiments/real_labels_2022/contact_sheet_2022.py [--root DIR]

Output: ``<root>/review/contact_sheet/contact_sheet.png`` (+ the items' truth).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_2022 as b
import build_session as v1
import contact_sheet as v1_sheet
import design2022 as d
from common import VIEW, recording_dir, view_config

SEED = 919191
N_BOX, N_WIN = 9, 3


def pick_items(rng: np.random.Generator, root: Path) -> pd.DataFrame:
    truth = pd.read_parquet(root / "truth" / f"{d.SESSION}_truth.parquet")
    det = b.load_2022()
    v1_boxes, v1_slots, v1_earlier = b.real_v1_2022()
    regs, bg = v1.regions(), v1.background_segments()
    cand = b.candidate_pool(det, regs, bg, pd.concat([v1.earlier_boxes(), v1_earlier]), v1_boxes)
    pool = cand[cand["in_frame"] & cand["kept"]]
    queued = set(zip(truth["recording"], truth["chirp_id"], strict=True))
    pool = pool[[(r, c) not in queued for r, c in zip(pool["recording"], pool["chirp_id"], strict=True)]]
    cells = pool.groupby(["score_bin", "status", "group"]).size().index.tolist()
    rows = []
    for k in rng.choice(len(cells), size=N_BOX, replace=False):
        sb, st, g = cells[k]
        c = pool[(pool["score_bin"] == sb) & (pool["status"] == st) & (pool["group"] == g)]
        rows.append(c.iloc[rng.integers(len(c))])
    boxes = pd.DataFrame(rows)
    boxes["item_kind"] = "box"
    frame = pd.read_parquet(root / "truth" / "window_frame_slots.parquet")
    taken = set(zip(truth["recording"], truth["zone_t0_s"], strict=True)) | v1_slots
    frame = frame[[(r, t) not in taken for r, t in zip(frame["recording"], frame["zone_t0_s"], strict=True)]]
    wins = pd.concat([frame[frame["group"] == "pair3"].sample(2, random_state=rng),
                      frame[frame["group"] == "rest"].sample(1, random_state=rng)])
    wins["item_kind"] = "window"
    wins["t_start_s"], wins["t_end_s"] = wins["zone_t0_s"], wins["zone_t1_s"]
    wins["f_low_hz"] = wins["f_min_hz"]
    wins["f_high_hz"] = wins["f_max_hz"] + v1.WINDOW_HEADROOM_HZ
    items = pd.concat([boxes, wins], ignore_index=True)
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
    items["candidate_id"] = [f"sheet{k:02d}" for k in range(len(items))]
    items["tracks"] = [str(v1.labeller_tracks(r)[0]) for r in items["recording"]]
    items["recording_dir"] = [str(recording_dir("competition2022", r)) for r in items["recording"]]
    return items


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chirpdetector.labeling.candidates import load_or_build_candidate_queue
    from chirpdetector.labeling.queue import QueueConfig
    from chirpdetector.labeling.server import LabelServer
    from chirpdetector.labeling.session import LabelSession
    from chirpdetector.labeling.store import LabelStore

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=d.ROOT, help="session root (truth/)")
    args = ap.parse_args()
    out_dir = args.root / "review" / "contact_sheet"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = pick_items(np.random.default_rng(SEED), args.root)
    cands = pd.DataFrame({
        "recording": items["recording_dir"], "t_start_s": items["t_start_s"].clip(lower=0.0),
        "t_end_s": items["t_end_s"], "f_low_hz": items["f_low_hz"], "f_high_hz": items["f_high_hz"],
        "candidate_id": items["candidate_id"], "tracks": items["tracks"], "item_kind": items["item_kind"]})
    path = out_dir / "contact_sheet_items.parquet"
    cands.to_parquet(path, index=False)
    keep = ["candidate_id", "item_kind", "recording", "score_bin", "status", "group", "size", "score",
            "t_center_s", "zone_t0_s", "f_low_hz", "f_high_hz", "emitter_hz", "track_id", "n_live"]
    items[[c for c in keep if c in items]].to_csv(out_dir / "contact_sheet_items_truth.csv", index=False)
    with tempfile.TemporaryDirectory() as tmp:
        queue = load_or_build_candidate_queue(path, QueueConfig(), tmp, shuffle=False)
        with LabelStore(tmp, "contact-sheet", "sheet") as store:
            session = LabelSession(queue, store, config=view_config(), frontend="browser",
                                   blind=True, background=False)
            server = LabelServer(session, port=0)
            try:
                ncol, nrow = 4, 3
                fig = plt.figure(figsize=(4.6 * ncol, 3.4 * nrow))
                outer = fig.add_gridspec(nrow, ncol, hspace=0.35, wspace=0.18)
                for k in range(len(queue)):
                    v = server.view(k, VIEW["window_s"])
                    imgs = {n: server.image(k, VIEW["window_s"], n) for n in ("overview", "zoom")}
                    inner = outer[k // ncol, k % ncol].subgridspec(2, 1, height_ratios=[0.4, 0.6], hspace=0.12)
                    kind = "WINDOW (mark every chirp in the shaded zone)" if "zone" in v["item"] else "candidate"
                    v1_sheet.draw_item(fig.add_subplot(inner[0]), fig.add_subplot(inner[1]), v, imgs,
                                       f"#{k + 1} {kind}")
                fig.suptitle("real_v2_2022 (2022 recordings): the blind labeller's view, as in real_v1 "
                             "(5 s x 600 Hz, 204.8 ms, colours p5..max per panel, hint 0.3 s x 200 Hz); "
                             "items from the session's frame, not in its queue", fontsize=9, color="#dcdcdc")
                out = out_dir / "contact_sheet.png"
                fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#1b1b1b")
                plt.close(fig)
                print(f"wrote {out}")
            finally:
                server.httpd.server_close()
                session.shutdown()


if __name__ == "__main__":
    main()
