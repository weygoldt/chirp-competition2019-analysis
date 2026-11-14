"""Contact sheet of 24 items as the blind browser labeller shows them.

20 candidates and 4 recall windows drawn at random from the session's own
sampling frame (same strata, same windows' frame), but NOT from the queue,
so that the author does not see items before labelling them. Each item is
rendered through the labeller's server code (``LabelServer.view`` and
``LabelServer.image``: the exact PNG panels and colours the page receives,
with the session's settings: blind, 5 s x 600 Hz, hint box) and drawn with
the page's overlays (time marker, hint box, track ticks and numbers; the
shaded zone of a window). The page itself draws on a canvas; this is a
faithful replica of its geometry, not a screenshot.

    uv run python experiments/real_labels/contact_sheet.py

Output: ``$CHIRP_DATA/labels/real_v1/review/contact_sheet/contact_sheet.png``
(+ the candidates file of the 24 items).
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_session as bs  # noqa: E402
from common import ROOT, VIEW, dataset_of, recording_dir, view_config  # noqa: E402

SEED = 424242
N_BOX, N_WIN = 20, 4


def pick_items(rng: np.random.Generator, root: Path) -> pd.DataFrame:
    truth = pd.read_parquet(root / "truth" / "real_v1_truth.parquet")
    det = bs.load_detections()
    det["score_bin"] = pd.cut(det["score"], bs.SCORE_BINS, right=False,
                              labels=bs.SCORE_NAMES).astype(str)
    det["size"] = bs.size_of(det)
    regs, bg, earlier = bs.regions(), bs.background_segments(), bs.earlier_boxes()
    ok = bs.frame_mask(det, det["t_center_s"].to_numpy(), 0.0, regs, bg)
    ok[ok] &= ~bs.near_earlier(det[ok], earlier)
    pool = det[ok & det["kept"]]
    used = set(zip(truth["recording"], truth["chirp_id"], strict=True))
    pool = pool[[(r, c) not in used for r, c in zip(pool["recording"], pool["chirp_id"], strict=True)]]
    strata = pool.groupby(["dataset", "score_bin", "size"]).size().index.tolist()
    rows = []
    for k in range(N_BOX):
        ds, b, s = strata[rng.integers(len(strata))]
        g = pool[(pool["dataset"] == ds) & (pool["score_bin"] == b) & (pool["size"] == s)]
        rows.append(g.iloc[rng.integers(len(g))])
    boxes = pd.DataFrame(rows)
    boxes["item_kind"] = "box"
    frame = pd.read_parquet(root / "truth" / "window_frame_slots.parquet")
    taken = set(zip(truth["recording"], truth["zone_t0_s"], strict=True))
    frame = frame[[(r, t) not in taken for r, t in zip(frame["recording"], frame["zone_t0_s"], strict=True)]]
    wins = frame.iloc[rng.choice(len(frame), N_WIN, replace=False)].copy()
    wins["item_kind"] = "window"
    wins["t_start_s"], wins["t_end_s"] = wins["zone_t0_s"], wins["zone_t1_s"]
    wins["f_low_hz"] = wins["f_min_hz"]
    wins["f_high_hz"] = wins["f_max_hz"] + bs.WINDOW_HEADROOM_HZ
    items = pd.concat([boxes, wins], ignore_index=True)
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
    items["candidate_id"] = [f"sheet{k:02d}" for k in range(len(items))]
    items["tracks"] = [str(bs.labeller_tracks(r)[0]) for r in items["recording"]]
    items["recording_dir"] = [str(recording_dir(dataset_of(r), r)) for r in items["recording"]]
    return items


def draw_item(ax_o, ax_z, view: dict, images: dict, title: str) -> None:  # noqa: ANN001
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle

    it = view["item"]
    for name, ax in (("overview", ax_o), ("zoom", ax_z)):
        e = view["panels"][name]["extent"]
        ax.imshow(mpimg.imread(io.BytesIO(images[name]), format="png"), extent=e, aspect="auto",
                  interpolation="bilinear")
        ax.set_xlim(e[0], e[1])
        ax.set_ylim(e[2], e[3])
        h = e[3] - e[2]
        if "zone" in it:
            z0, z1 = it["zone"]
            ax.add_patch(Rectangle((z0, e[2]), z1 - z0, h, color="white", alpha=0.07, lw=0))
            for z in (z0, z1):
                ax.axvline(z, color=(1, 1, 1, 0.55), ls=(0, (6, 5)), lw=0.8)
        else:
            tick = max(0.06 * h, 0.0)
            ax.plot([it["t_center_s"]] * 2, [e[3], e[3] - tick], color="white", lw=1.6)
            ax.plot([it["t_center_s"]] * 2, [e[2], e[2] + tick], color="white", lw=1.6)
            if "hint_box" in it and name == "overview":
                t0, t1, f0, f1 = it["hint_box"]
                ax.add_patch(Rectangle((t0, f0), t1 - t0, f1 - f0, fill=False,
                                       ec=(1, 1, 1, 0.45), ls=(0, (6, 5)), lw=0.8))
        span = 0.1 * (e[1] - e[0])
        for tr in view["tracks"]:
            for a, b in ((e[0], e[0] + span), (e[1] - span, e[1])):
                ta = np.array([a, b])
                ax.plot(ta, np.interp(ta, tr["t"], tr["f"]), color="#d0d0d0", lw=0.7, ls=(0, (5, 4)))
            y = float(np.interp(e[0], tr["t"], tr["f"]))
            text = f"{tr['key']}" + (f"  {tr['baseline_hz']:.1f} Hz" if name == "zoom" else "")
            ax.text(e[0] + 0.01 * (e[1] - e[0]), y, text, fontsize=5, va="center", clip_on=True,
                    bbox={"fc": "#d0d0d0", "ec": "none", "alpha": 0.88, "pad": 0.6})
        ax.tick_params(labelsize=5, length=2, colors="#cfcfcf")
        if name == "overview":
            ax.set_xticklabels([])
    ax_o.set_title(title, fontsize=6, color="#dcdcdc")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chirpdetector.labeling.candidates import load_or_build_candidate_queue
    from chirpdetector.labeling.queue import QueueConfig
    from chirpdetector.labeling.server import LabelServer
    from chirpdetector.labeling.session import LabelSession
    from chirpdetector.labeling.store import LabelStore

    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="session root (truth/)")
    ap.add_argument("--partial", action="store_true", help="trial builds: complete recordings only")
    args = ap.parse_args()
    bs.PARTIAL = args.partial
    out_dir = args.root / "review" / "contact_sheet"
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = pick_items(rng, args.root)
    cands = pd.DataFrame({
        "recording": items["recording_dir"], "t_start_s": items["t_start_s"].clip(lower=0.0),
        "t_end_s": items["t_end_s"], "f_low_hz": items["f_low_hz"], "f_high_hz": items["f_high_hz"],
        "candidate_id": items["candidate_id"], "tracks": items["tracks"],
        "item_kind": items["item_kind"],
    })
    path = out_dir / "contact_sheet_items.parquet"
    cands.to_parquet(path, index=False)
    items.drop(columns=[c for c in ("recording_dir",) if c in items]).to_csv(
        out_dir / "contact_sheet_items_truth.csv", index=False)
    with tempfile.TemporaryDirectory() as tmp:
        queue = load_or_build_candidate_queue(path, QueueConfig(), tmp, shuffle=False)
        with LabelStore(tmp, "contact-sheet", "sheet") as store:
            session = LabelSession(queue, store, config=view_config(), frontend="browser",
                                   blind=True, background=False)
            server = LabelServer(session, port=0)
            try:
                ncol, nrow = 4, 6
                fig = plt.figure(figsize=(4.6 * ncol, 3.4 * nrow))
                outer = fig.add_gridspec(nrow, ncol, hspace=0.35, wspace=0.18)
                for k in range(len(queue)):
                    v = server.view(k, VIEW["window_s"])
                    imgs = {n: server.image(k, VIEW["window_s"], n) for n in ("overview", "zoom")}
                    inner = outer[k // ncol, k % ncol].subgridspec(2, 1, height_ratios=[0.4, 0.6], hspace=0.12)
                    kind = "WINDOW (mark every chirp in the shaded zone)" if "zone" in v["item"] else "candidate"
                    draw_item(fig.add_subplot(inner[0]), fig.add_subplot(inner[1]), v, imgs,
                              f"#{k + 1} {kind}")
                fig.suptitle("real_v1: the blind labeller's view (benchmark settings: 5 s x 600 Hz, "
                             "204.8 ms, colours p5..max per panel, hint 0.3 s x 200 Hz); "
                             "items from the session's frame, not in its queue", fontsize=9,
                             color="#dcdcdc")
                out = out_dir / "contact_sheet.png"
                fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#1b1b1b")
                plt.close(fig)
                print(f"wrote {out}")
            finally:
                server.httpd.server_close()
                session.shutdown()


if __name__ == "__main__":
    main()
