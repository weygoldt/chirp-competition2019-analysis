"""Contact sheets of the candidate examples of ``find_examples_sw.py`` (for the author's choice).

Step ``windows`` writes ``candidate_windows.txt`` (one ``NAME RECORDING T0 T1``
per line) next to the candidate tables; extract them in the root environment::

    xargs -L1 echo --window < .../candidate_windows.txt | tr '\\n' ' ' | \\
        xargs $CHIRP_ROOT_PY scripts/v2_fig_extract.py snippets

(or pass the ``--window`` options directly). Step ``render`` draws the first
``N`` candidates of each kind as the figure panels do (``v2_fig_common``):
harmonic copies and duplicates as in Fig. 9B (kept box solid with its emitter,
removed box dashed), missed labelled chirps as in Fig. S4A (hand labels as
triangles, the stored box below the operating threshold dotted), and writes
``candidates_<kind>.png`` to the figure directory.

    cd paper && PAPER_DETECTIONS=rtdetr_sw_gapfill_t080 uv run python scripts/sheet_examples_sw.py windows
    cd paper && PAPER_DETECTIONS=rtdetr_sw_gapfill_t080 uv run python scripts/sheet_examples_sw.py render
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_config
import v2_fig_common as C
from plot_v2_failure_modes import hand_labels, match_labels
from style import set_light_style

N = 6
PAD_S = 1.0  # extra context around the short windows (the extraction pads to a tile anyway)


def tables() -> dict[str, pd.DataFrame]:
    return {k: pd.read_csv(C.SNIPPETS / f"candidates_{k}.csv").head(N)
            for k in ("harmonics", "duplicates", "missed")}


def windows() -> None:
    lines = []
    for kind, t in tables().items():
        for k, r in enumerate(t.itertuples()):
            pad = 0.0 if kind == "missed" else PAD_S
            lines.append(f"cand_{kind}_{k} {r.recording} {r.t0 - pad:.2f} {r.t1 + pad:.2f}")
    path = C.SNIPPETS / "candidate_windows.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {path}")


def pair_panel(ax, name, r, a_id, b_id, title):
    snip = C.load_snippet(name)
    t0, t1 = float(r.t0), float(r.t1)
    rows = C.window(snip, t0 - PAD_S, t1 + PAD_S).set_index("chirp_id")
    fish, _ = C.fish_styles(snip, t0 - 2, t1 + 2)
    C.show_power(ax, snip, t0, t1, 580, 1520)
    C.draw_tracks(ax, snip, fish, t0, t1)
    C.fish_box(ax, rows.loc[a_id], t0, fish)
    C.dropped_box(ax, rows.loc[b_id], t0)
    ax.set_title(title, fontsize="small")
    C.spec_yaxis(ax, ticks=[600, 800, 1000, 1200, 1400])


def missed_panel(ax, name, r):
    snip = C.load_snippet(name)
    t0, t1 = float(r.t0), float(r.t1)
    rows = C.window(snip, t0, t1)
    lab = hand_labels(r.recording)
    lab = lab[(lab.t_end_s > t0) & (lab.t_start_s < t1)]
    f0 = float(min(lab.f_low_hz.min(), rows[rows.operating].f_low_hz.min())) - 30
    f1 = float(max(lab.f_high_hz.max(), rows[rows.operating].f_high_hz.max())) + 30
    fish, _ = C.fish_styles(snip, t0, t1)
    C.show_power(ax, snip, t0, t1, f0, f1)
    C.draw_tracks(ax, snip, fish, t0, t1, bridge_s=C.ASSIGN_MAX_GAP_S)
    drawn = C.draw_detections(ax, rows, fish, t0)
    kept = drawn[drawn.drawn_as.isin(["assigned", "unassigned"])]
    lab_c, unmatched, extra = match_labels(lab, kept)
    C.label_marks(ax, lab_c - t0)
    for tc in unmatched:
        below = rows[~rows.operating & ((rows.t_center_s - tc).abs() <= 0.1)]
        if len(below):
            C.draw_missed(ax, below.sort_values("score").iloc[-1], t0)
    ax.set_title(f"{r.recording} {t0:.1f}-{t1:.1f} s: {len(lab)} labels, missed at "
                 f"{', '.join(f'{u:.2f}' for u in unmatched)} (below: {r.below_score:.2f}), "
                 f"{len(extra)} detections without label", fontsize="small")
    C.spec_yaxis(ax)


def render() -> None:
    set_light_style()
    C.rc_tweaks()
    t = tables()
    for kind in ("harmonics", "duplicates"):
        cand = t[kind]
        fig, axs = plt.subplots(1, len(cand), figsize=(3.0 * len(cand), 3.6), squeeze=False)
        for k, (ax, r) in enumerate(zip(axs[0], cand.itertuples(), strict=True)):
            if kind == "harmonics":
                title = (f"{k}: {r.recording}\nfund {r.fundamental} ({r.fundamental_score:.2f}), "
                         f"harm {r.harmonic} ({r.harmonic_score:.2f})")
                pair_panel(ax, f"cand_{kind}_{k}", r, r.fundamental, r.harmonic, title)
            else:
                title = (f"{k}: {r.recording}\nkept {r.kept} ({r.kept_score:.2f}), "
                         f"dropped {r.dropped} ({r.dropped_score:.2f})")
                pair_panel(ax, f"cand_{kind}_{k}", r, r.kept, r.dropped, title)
        fig.tight_layout()
        path = paper_config.figure(f"candidates_{kind}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(path)
    cand = t["missed"]
    fig, axs = plt.subplots(len(cand), 1, figsize=(12, 2.6 * len(cand)), squeeze=False)
    for k, (ax, r) in enumerate(zip(axs[:, 0], cand.itertuples(), strict=True)):
        missed_panel(ax, f"cand_missed_{k}", r)
    fig.tight_layout()
    path = paper_config.figure("candidates_missed.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    {"windows": windows, "render": render}[sys.argv[1]]()
    _ = np  # numpy for interactive use
