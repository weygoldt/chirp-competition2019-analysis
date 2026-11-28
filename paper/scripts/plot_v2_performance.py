"""Figure: performance of chirp detection pipeline v2 (manuscript Fig. 10).

Replaces ``plot_pipeline_performances_manuscript_supplement_supplement.pdf``.
Framing (author, 2026-11-16): on real recordings a human expert is not
ground truth, so the real-data panels show AGREEMENT WITH THE EXPERT (used to
choose the operating threshold); only the simulator gives absolute
performance (model and human expert vs simulator truth). One row per
pipeline stage (author, 2026-11-16; assignment by chirp size and the 10 %
width added 2026-11-24): detection (A-D), assignment (E-H), chirp
parameters (I-K). Only the production detector (RT-DETR dual-head; the
pipeline's, ``paper_config.PIPELINE``) is shown;
YOLO26s has no emitter head and is no like-for-like comparison. The
operating threshold is the run's (``meta.json`` of PAPER_DETECTIONS' run;
0.8 for rtdetr_gapfill_t080); every operating point follows it.

A   Simulated held-out test scenes (unseen fish on unseen background
    recordings; 9,401 visible chirps), after post-processing: precision
    against recall over the score threshold,
    ``<detector run>/eval_postprocess/sim_test_after`` (``C.PP_EVAL``);
    operating point from the curve's row at the operating threshold.
B   Blind benchmark on 1,010 simulated items (536 scored true chirps):
    recall of the human expert and of RT-DETR dual-head (score >= operating
    threshold, kept), by chirp SNR (``C.BENCHMARK/items.csv``;
    unweighted, Wilson 95 % intervals).
C   Real recordings (2019/2020 competitions), agreement with one expert
    (blind labelling session real_v1), drawn like A as one detector curve
    over the score threshold: share of the expert's window chirps the
    detector found (x; ``recall.csv``) against share of detections the
    expert judged chirps (y; ``precision_at_threshold.csv``);
    design-weighted, 95 % bootstrap intervals at 0.5, 0.9 and the operating
    threshold. Shares the y-axis with D.
D   Detections the expert judged chirps, for detections assigned to the
    trial's winner vs loser, by score bin (0.5-0.7, 0.7-0.9, >= 0.9;
    ``precision_by_status.csv``).
E   Emitter assignment of the simulated true positives at the operating
    threshold (true positives of the evaluation at 0.5 with score >= the
    threshold and a known ΔF), by the emitter's ΔF to the nearest other live
    track (``emitter_df_bin``; ``v2_fig_extract.py assign-baseline``). The
    box-position baseline of that file is still computed and written to the
    stats (``assignment.*.edge_*``) but no longer drawn (author, 2026-11-24:
    the manuscript never assigns by box position).
F   The benchmark of B: emitter correct among each one's own detections
    (expert: items called chirp; detector: kept, score >= operating
    threshold), by the emitter's ΔF to the nearest other live fish
    (``df_nearest_hz``); unweighted, Wilson 95 % intervals.
G,H Detection and assignment by the chirp's TRUE 10 % width (G; octave bins
    10, 20, 40, 80, 160 ms, outer bins open) and TRUE height (H; octave bins
    50, 100, 200, 400 Hz): the simulator's per-period truth of each chirp
    (``v2_assign_params.py``). Two series (author's request 2026-11-25:
    detection by chirp parameter without another row): recall, the share of
    the visible chirps of A (its 9,401 chirp labels) detected at the
    operating threshold (``detect_params.csv``: the evaluation's matching
    re-run with the boxes of score >= the threshold, A's operating point;
    squares), and the share of E's true positives assigned correctly
    (``assign_params.csv``; circles, as E). Numbers: chirp labels per bin
    (recall's n). G and H share a y-axis ("Fraction", 0.6-1; author,
    2026-11-25: E's 0.4-1 left the flat series squeezed at the top).
I-K Chirp-parameter decoder for boxes at the operating threshold on the
    simulated test chirps of unseen fish (fish-disjoint template splits):
    calibrated estimate vs per-period truth of height (I), FWHM (J) and 10 %
    width (K) (``params/.../eval/predictions.parquet`` of the bundle
    recalibrated without the octave-error template of the validation split).
    Every chirp is a point coloured by its local density (author, 2026-11-27,
    variant 2 of five drafts: no pixels, raw data visible): a Gaussian KDE of
    the points in log10 space (Scott's bandwidth), the colour its rank among
    the points' densities (densest on top; the scale starts above the
    palest mint so that sparse points stay visible); a very thin light-grey
    dashed contour encloses the densest 95 % of the chirps (the KDE level
    that 95 % of the points exceed; one key for I-K in I's upper left:
    author, 2026-11-27). Limits fit the data after the octave
    fix: I 20-800 Hz, J 1.5-600 ms, K 3-700 ms (``n_beyond_axes``: 0). The
    test split's octave-error template
    (``v2_fig_common.OCTAVE_TEMPLATES``: a frequency trace at twice the
    baseline EODf, true "height" ~770 Hz) is left out (author, 2026-11-27);
    its count is ``decoder.n_octave_excluded``. The former
    real long-chirp comparison (``real_agreement.csv``) is dropped: 429 of
    its 533 chirps come from fish whose templates were in the decoder's
    training split.

Figure standard (author, 2026-11-24): 17.4 cm wide, placed at 100 % of the
text width (printed at x0.92); 8-pt labels, "small" (6.7 pt) legends,
annotations and n; the palette of Figs. 5-7 only: detector = the indigo of
``style.other_color`` (filled markers), human expert = grey open markers,
winner/loser = ``style.winner_color``/``loser_color``, black references,
mako densities. Layout (author, 2026-11-25): explicit cm in
``v2_performance_layout.py`` (rows of equal span, equal panel widths per
row, wider gaps between groups and rows, the simulated and real panels of
the detection row under two group headings); the script prints every
axes' extent in cm.

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_performance.py

Runtime: ~15 s cold, ~4 s warm. Prints every number it plots and writes
them with write_stats ("v2_performance"; bins and thresholds as word keys,
e.g. ``agreement.t0_8``, ``benchmark.dualhead.by_snr.below_m20_db``,
``assignment.by_width10.head_correct.10_to_20_ms``,
``sim_recall.by_height.detected.0_to_50_hz``, ``sim_recall.by_width10.p_range``).
"""

from __future__ import annotations

import json
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

import v2_fig_common as C
import v2_performance_layout as L
from manuscript_stats import write_stats
from style import loser_color, other_color, set_light_style, winner_color

NAME = "plot_v2_performance"
# figure size [cm]: full text width (17.4 cm, placed at 100 %, printed at x0.92
# like the other full-width figures); at most 15.05 cm tall
W, H = L.W, L.H
THR = C.OPERATING
# the expert's session on the real recordings, scored for the pipeline's
# detector (labels/real_v1/report for v2, report_sw for the single-window one)
REAL_V1 = C.REAL_SESSION
# the production detector's name in the blind benchmark's report; its series
# and stats keys stay "dualhead" (the production detector of either pipeline)
BM = C.BENCHMARK_MODEL
REAL_DATASET = "2019"  # the 2019/2020 tube competitions (the paper's data)
MARKED = (0.5, 0.9)  # thresholds marked in C besides the operating one
# bins as labelled in the input tables, and as keys of the stats
STATUS_BINS = ["0.5-0.7", "0.7-0.9", "0.9-1"]
STATUS_KEYS = ["score_0_5_to_0_7", "score_0_7_to_0_9", "score_0_9_to_1"]
STATUS_CENTRES = [0.6, 0.8, 0.95]  # centres of the score bins in D
DF_BINS = ["0-10", "10-25", "25-50", "50-100", "100+"]
DF_KEYS = ["df_0_to_10_hz", "df_10_to_25_hz", "df_25_to_50_hz", "df_50_to_100_hz", "df_100_hz_up"]
DF_POS = [5.0, 15.8, 35.4, 70.7, 141.0]  # geometric bin centres; open bins at the edge
SNR_BINS = ["<-20 dB", "-20 to -10 dB", "-10 to 0 dB", "0-10 dB", ">=10 dB"]
SNR_KEYS = ["below_m20_db", "m20_to_m10_db", "m10_to_0_db", "0_to_10_db", "10_db_up"]
SNR_POS = [-25.0, -15.0, -5.0, 5.0, 15.0]
# true chirp size (G, H): octave bins, outer bins open; points at the
# geometric centres (open bins half an octave beyond their edge)
WIDTH_EDGES_MS = [10.0, 20.0, 40.0, 80.0, 160.0]
WIDTH_KEYS = ["0_to_10_ms", "10_to_20_ms", "20_to_40_ms", "40_to_80_ms", "80_to_160_ms",
              "160_ms_up"]
HEIGHT_EDGES_HZ = [50.0, 100.0, 200.0, 400.0]
HEIGHT_KEYS = ["0_to_50_hz", "50_to_100_hz", "100_to_200_hz", "200_to_400_hz", "400_hz_up"]
PARAMS_KEYS = ["recording", "label_row", "pred_row"]  # a true positive (assign_*.csv)
DETECT_KEYS = ["recording", "label_row"]  # a chirp label (detect_params.csv)

# palette (Figs. 5-7): the detector in the indigo of the other-event colour,
# filled; the human expert grey and open; references black
DETECTOR = other_color
EXPERT = "#808080"
SERIES = {
    "dualhead": {"color": DETECTOR, "mfc": DETECTOR, "mec": "k", "mew": 0.5},
    "human": {"color": EXPERT, "mfc": "white", "mec": EXPERT, "mew": 0.8},
}
LABELS = {"dualhead": "RT-DETR dual-head", "human": "human expert"}
# G, H: two detector series, told apart by marker and line style
SIZE_SERIES = {
    "detected": {**SERIES["dualhead"], "marker": "s", "ls": (0, (2.5, 1.5)),
                 "label": "detected"},
    "head_correct": {**SERIES["dualhead"], "marker": "o", "ls": "-",
                     "label": "correctly assigned"},
}
LW = 1.0  # data lines
LW_CURVE = 1.3  # threshold curves (A, C)
EB_LW = 0.8  # error bars
REF_LW = 0.6  # reference lines
OP_MS = 4.0  # operating-point marker
PT_MS = 3.3  # proportion markers
SPINE_OFFSET = 5


def fmt_thr(t: float) -> str:
    return f"{t:g}"


def thr_key(t: float) -> str:
    """A threshold as a stats key: 0.8 -> t0_8."""
    return "t" + fmt_thr(t).replace(".", "_")


def by_bin(keys, k, n, p, lo, hi) -> dict:
    """Per bin: successes, items, proportion and its 95 % interval."""
    return {key: {"k": a, "n": b, "p": c, "ci_95": [d, e]}
            for key, a, b, c, d, e in zip(keys, k, n, p, lo, hi, strict=True)}


def octave_positions(edges: list[float]) -> list[float]:
    """Geometric centres of the bins of ``edges`` plus two open outer bins."""
    e = np.asarray(edges, float)
    step = np.sqrt(e[1] / e[0])
    return [e[0] / step, *np.sqrt(e[:-1] * e[1:]), e[-1] * step]


def despine(ax, *, trim=False):
    sns.despine(ax=ax, offset=SPINE_OFFSET, trim=trim)


HEADING_RULE = {"color": "#b4b4b4", "lw": 0.6}  # the rule of a group heading
HEADING_PAD = 0.15  # gap between the heading text and its rule [cm]


def group_heading(fig, text: str, x0: float, x_text: float, x1: float, y: float) -> None:
    """A group heading on a thin rule from x0 to x1 (cm), at height y (cm); the
    text starts at x_text."""
    tr = C.fig_cm(fig)
    t = fig.text(x_text, y, text, transform=tr, ha="left", va="center", fontsize="medium")
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(tr.inverted())
    for a, b in ((x0, bb.x0 - HEADING_PAD), (bb.x1 + HEADING_PAD, x1)):
        fig.add_artist(Line2D([a, b], [y, y], transform=tr, solid_capstyle="butt",
                              **HEADING_RULE))


def extent_table(fig, axes: dict) -> str:
    """Every axes' extent [cm] as drawn (after aspect adjustment), as a table."""
    w, h = fig.get_size_inches() / C.CM
    boxes = {}
    for name, a in axes.items():
        a.apply_aspect()
        p = a.get_position()
        boxes[name] = L.Box(p.x0 * w, p.y0 * h, p.width * w, p.height * h)
    return L.extent_table(boxes)


# ---------------------------------------------------------------------------
# real recordings: agreement with the expert
# ---------------------------------------------------------------------------
def agreement_panel(ax, report):
    """Agreement with the expert as one detector curve over the score
    threshold, drawn like A's precision-recall curve: share of the expert's
    chirps the detector found (x) against share of detections the expert
    judged chirps (y). Thresholds 0.5, 0.9 and the operating one marked
    with 95 % intervals (both directions)."""
    prec = pd.read_csv(REAL_V1 / "precision_at_threshold.csv")
    rec = pd.read_csv(REAL_V1 / "recall.csv")
    prec = prec[prec.dataset.astype(str) == REAL_DATASET].set_index("threshold")
    rec = rec[rec.dataset.astype(str) == REAL_DATASET].set_index("threshold")
    thr = np.array(sorted(set(prec.index) & set(rec.index)))
    if not np.any(np.isclose(thr, THR)):
        msg = f"operating threshold {THR} not among the session's thresholds {thr}"
        raise ValueError(msg)
    s = SERIES["dualhead"]
    y, ylo, yhi = (prec.loc[thr, k].to_numpy() for k in ("precision", "ci_low", "ci_high"))
    x, xlo, xhi = (rec.loc[thr, k].to_numpy() for k in ("recall", "ci_low", "ci_high"))
    marked = np.array([any(np.isclose(t, m) for m in (*MARKED, THR)) for t in thr])
    ax.plot(x, y, color=s["color"], lw=LW_CURVE, zorder=2, solid_joinstyle="round")
    ax.plot(x[~marked], y[~marked], "o", ms=2.2, mfc=s["color"], mec=s["color"], mew=0.5,
            zorder=3, clip_on=False)
    ax.errorbar(x[marked], y[marked], xerr=[x[marked] - xlo[marked], xhi[marked] - x[marked]],
                yerr=[y[marked] - ylo[marked], yhi[marked] - y[marked]], fmt="o", ms=PT_MS,
                mfc=s["mfc"], mec=s["mec"], mew=s["mew"], ecolor=s["color"], elinewidth=EB_LW,
                capsize=0, zorder=4, clip_on=False)
    k_op = int(np.flatnonzero(np.isclose(thr, THR))[0])
    C.annotate(ax, rf"score $\geq$ {fmt_thr(THR)}", xy=(x[k_op], y[k_op]),
               xytext=(0.5, 0.42), color="k", rad=0.2, shrink_b=4)
    # the other marked thresholds: their value beside the point, clear of the
    # curve, the error bars and the arrow
    for t, dxy in ((0.5, (-1, -13)), (0.9, (-6, -8))):
        k = int(np.flatnonzero(np.isclose(thr, t))[0])
        ax.annotate(fmt_thr(t), xy=(x[k], y[k]), xytext=dxy, textcoords="offset points",
                    ha="right", va="center", fontsize="small", color=C.MUTED)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Expert's chirps found")
    ax.set_ylabel("Detections\njudged chirps")  # two lines: the short axis
    despine(ax, trim=True)
    report["agreement"] = {
        "thresholds": thr.tolist(), "marked": [*MARKED, THR],
        "windows": int(rec.loc[thr[0], "windows"]), "marks": int(rec.loc[thr[0], "marks"]),
        **{thr_key(t): {"detections_judged_chirps": {"value": y[i], "ci_95": [ylo[i], yhi[i]]},
                        "expert_chirps_found": {"value": x[i], "ci_95": [xlo[i], xhi[i]]},
                        "n_candidates": int(prec.loc[t, "n"])}
           for i, t in enumerate(thr)},
    }


def status_panel(ax, report):
    s = pd.read_csv(REAL_V1 / "precision_by_status.csv")
    s = s[(s.dataset.astype(str) == REAL_DATASET) & s.scores.isin(STATUS_BINS)]
    out = {}
    for status, col, dx in (("winner", winner_color, -0.012), ("loser", loser_color, 0.012)):
        g = s[s.status == status].set_index("scores").reindex(STATUS_BINS)
        xx = np.asarray(STATUS_CENTRES) + dx
        p = g.precision.to_numpy()
        ax.plot(xx, p, color=col, lw=LW, zorder=3, clip_on=False)
        ax.errorbar(xx, p, yerr=[p - g.ci_low.to_numpy(), g.ci_high.to_numpy() - p], fmt="o",
                    ms=PT_MS, mfc=col, mec="k", mew=0.5, ecolor=col, elinewidth=EB_LW,
                    capsize=0, zorder=4, clip_on=False, label=status)
        # "precision" of the session's table = share of detections the expert judged chirps
        out[status] = {key: {"n": r.n, "n_hat": r.N_hat, "judged_chirps": r.precision,
                             "ci_95": [r.ci_low, r.ci_high]}
                       for key, r in zip(STATUS_KEYS, g.itertuples(), strict=True)}
    # score bins as points at their centres between the bin edges (ticks)
    ax.set_xticks([0.5, 0.7, 0.9, 1.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Detector score")
    despine(ax, trim=True)
    # the empty upper left (low-score bins sit near zero)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04), frameon=False,
              handletextpad=0.3, borderaxespad=0.1, handlelength=1.2)
    report["by_status"] = out


# ---------------------------------------------------------------------------
# simulated: detection
# ---------------------------------------------------------------------------
def pr_panel(ax, models, base, report):
    report["sim"] = {}
    for m in models:
        curve = pd.read_csv(base(m) / "pr_curve.csv")
        curve = curve[curve["level"] == "all"].sort_values("threshold")
        # the operating point: the curve's row at the threshold (the
        # evaluation re-matches at every threshold)
        op = curve[curve.threshold >= THR].iloc[0]
        report["sim"][m] = {k: float(op[k]) for k in ("threshold", "tp", "fp", "fn",
                                                     "precision", "recall")}
        s = SERIES[m]
        ax.plot(curve["recall"], curve["precision"], color=s["color"], lw=LW_CURVE, zorder=3,
                solid_joinstyle="round")
        ax.plot(op["recall"], op["precision"], "o", ms=OP_MS, mfc=s["mfc"], mec=s["mec"],
                mew=s["mew"], zorder=5, clip_on=False)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.8, 1.0)
    ax.set_ylim(0.85, 1.0)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xticklabels(["0.8", "", "0.9", "", "1.0"])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    despine(ax, trim=True)


def proportion_points(ax, x, k, n, series, *, dx=0.0, logx=False, zorder=4):
    """Proportions k/n at x joined by a line, with Wilson 95 % intervals.

    ``series``: colour, marker face/edge colour and edge width; optionally
    ``marker`` (default circle) and line style ``ls`` (default solid).
    """
    p = k / n
    lo, hi = C.wilson(k, n)
    xx = np.asarray(x) * (10**dx if logx else 1) + (0 if logx else dx)
    ax.plot(xx, p, color=series["color"], lw=LW, ls=series.get("ls", "-"), zorder=zorder - 1,
            clip_on=False)
    ax.errorbar(xx, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                fmt=series.get("marker", "o"), ms=PT_MS, mfc=series["mfc"], mec=series["mec"],
                mew=series["mew"], ecolor=series["color"], elinewidth=EB_LW, capsize=0,
                zorder=zorder, clip_on=False)
    return p, lo, hi


N_GAP_PT = 2.0  # least gap between two neighbouring n labels
N_LIFT_PT = 7.5  # lift of every second n label when neighbours would touch


def n_row(ax, xs, ns, y, fmt="{:,}"):
    """Items per bin along the panel's foot (small, muted).

    Call after the panel's limits and scales are final: where two
    neighbouring numbers would touch, every second one is lifted by one line
    (``N_LIFT_PT``), so that no number overlaps another.
    """
    texts = [ax.annotate(fmt.format(int(n)), xy=(x, y), xytext=(0, 0),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize="small", color=C.MUTED, annotation_clip=False)
             for x, n in zip(xs, ns, strict=True)]
    renderer = ax.figure.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    gap = N_GAP_PT * ax.figure.dpi / 72
    if any(a.x1 + gap > b.x0 for a, b in pairwise(boxes)):
        for t in texts[1::2]:
            t.xyann = (0, N_LIFT_PT)
    return texts


def benchmark_panel(ax, report):
    b = pd.read_csv(C.BENCHMARK / "items.csv", low_memory=False)
    t = b[(b["truth"] == "chirp") & b["scored"].astype(bool)].copy()
    # the model at the operating threshold: kept and score >= threshold
    # (``<model>_chirp`` is kept and score >= 0.5; the production detector's
    # name in the report is C.BENCHMARK_MODEL)
    t["model_chirp"] = t[f"{BM}_chirp"].astype(bool) & (t[f"{BM}_score"] >= THR)
    t["model_emitter_ok"] = t[f"{BM}_emitter_ok"].astype(bool) & t["model_chirp"]
    out = {}
    for name, key, dx in (("human_chirp", "human", -0.9), ("model_chirp", "dualhead", 0.9)):
        g = t.groupby("snr_bin")[name].agg(["sum", "size"]).reindex(SNR_BINS)
        p, lo, hi = proportion_points(ax, SNR_POS, g["sum"].to_numpy(float),
                                      g["size"].to_numpy(float), SERIES[key], dx=dx,
                                      zorder=5 if key == "dualhead" else 4)
        out[key] = {"recall": float(t[name].mean()),
                    "by_snr": by_bin(SNR_KEYS, g["sum"], g["size"], p, lo, hi)}
    out["human"]["emitter_ok"] = float(t["human_emitter_ok"].astype(bool).mean())
    out["dualhead"]["emitter_ok"] = float(t["model_emitter_ok"].astype(bool).mean())
    n = t.groupby("snr_bin").size().reindex(SNR_BINS)
    ax.set_xlim(-30, 20)
    ax.set_xticks([-20, -10, 0, 10])
    ax.set_xticklabels(["$-$20", "", "0", "10"])  # a label at -10 would touch -20
    ax.set_xlabel("Chirp SNR [dB]")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    despine(ax)
    n_row(ax, SNR_POS, n, 0.505, fmt="{}")
    out["n_true_chirps"] = int(len(t))
    out["n_items"] = int(len(b))
    out["threshold"] = THR
    report["benchmark"] = out


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------
ASSIGN_YLIM = (0.4, 1.0)
N_Y = 0.405  # the n row at the foot of the assignment panels
SIZE_YLIM = (0.6, 1.0)  # G, H: detection and assignment by chirp size
SIZE_N_Y = 0.605


def assign_yaxis(ax, ylabel: str | None = "Correctly assigned", ylim=ASSIGN_YLIM,
                 ticks=(0.4, 0.6, 0.8, 1.0)):
    """The y-axis of the assignment row (E-H; one range): labelled, or
    without tick labels where a panel shares its left neighbour's axis."""
    ax.set_ylim(*ylim)
    ax.set_yticks(list(ticks))
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(labelleft=False)


def log_xaxis(ax, lims, ticks):
    """Log x-axis with plain-number ticks only at ``ticks``."""
    ax.set_xscale("log")
    ax.set_xlim(*lims)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def df_axis(ax, *, ylabel=True):
    """ΔF axis of E and F (log x, open outer bins) and the row's y-axis."""
    log_xaxis(ax, (3.3, 210), [10, 25, 50, 100])
    assign_yaxis(ax, "Correctly assigned" if ylabel else None)
    despine(ax)


def df_bins(df_hz: pd.Series) -> pd.Series:
    """The evaluation's ΔF bins (left-closed edges 0, 10, 25, 50, 100 Hz)."""
    return pd.cut(df_hz, [0, 10, 25, 50, 100, np.inf], labels=DF_BINS, right=False)


def size_bins(values: pd.Series, edges: list[float], keys: list[str]) -> pd.Series:
    """Left-closed bins of ``edges`` with open outer bins, labelled by ``keys``."""
    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum())} chirps without a true value")
    return pd.cut(values, [-np.inf, *edges, np.inf], labels=keys, right=False)


def benchmark_emitter_panel(ax, report):
    """Benchmark: emitter correct among each one's own detections, expert vs
    detector, by the emitter's ΔF to the nearest other live fish."""
    b = pd.read_csv(C.BENCHMARK / "items.csv", low_memory=False)
    t = b[(b["truth"] == "chirp") & b["scored"].astype(bool)].copy()
    n_unknown = int(t["df_nearest_hz"].isna().sum())
    t["df_bin"] = df_bins(t["df_nearest_hz"])
    t = t[t["df_bin"].notna()]
    called = {
        "human": t["human_chirp"].astype(bool),
        "dualhead": t[f"{BM}_chirp"].astype(bool) & (t[f"{BM}_score"] >= THR),
    }
    ok = {"human": "human_emitter_ok", "dualhead": f"{BM}_emitter_ok"}
    out = {}
    for key, dx in (("human", -0.025), ("dualhead", 0.025)):
        d = t[called[key]]
        if (d[ok[key]].astype(bool) & ~called[key][d.index]).any():
            raise ValueError(f"{ok[key]} set for items {key} did not call a chirp")
        g = d.groupby("df_bin", observed=False)[ok[key]].agg(["sum", "size"]).reindex(DF_BINS)
        p, lo, hi = proportion_points(ax, DF_POS, g["sum"].to_numpy(float),
                                      g["size"].to_numpy(float), SERIES[key],
                                      dx=dx, logx=True, zorder=5 if key == "dualhead" else 4)
        # n of a bin: the chirps of that bin this one called chirps
        out[key] = {"overall": float(d[ok[key]].astype(bool).mean()), "n": int(len(d)),
                    "by_df": by_bin(DF_KEYS, g["sum"], g["size"], p, lo, hi)}
    n = t.groupby("df_bin", observed=False).size().reindex(DF_BINS)
    df_axis(ax, ylabel=False)
    n_row(ax, DF_POS, n, N_Y, fmt="{}")
    out["n_true_chirps_per_bin"] = dict(zip(DF_KEYS, n.astype(int).tolist(), strict=True))
    # the lowest share correct of expert and detector from 25 Hz up (cited in the caption)
    out["min_p_from_25_hz"] = min(out[key]["by_df"][b]["p"]
                                  for key in ("human", "dualhead") for b in DF_KEYS[2:])
    out["n_df_unknown"] = n_unknown
    out["threshold"] = THR
    report["benchmark_emitter"] = out


def assignment_set(report) -> pd.DataFrame:
    """The simulated true positives of E (score >= threshold, known ΔF), with
    their chirp's truth; writes the counts of E to ``report["assignment"]``.

    The box-position baselines (``edge_*``) are computed for the stats only.
    """
    a = pd.read_csv(C.SNIPPETS / "assign_baseline.csv")
    if "emitter_df_bin" not in a.columns:
        msg = "assign_baseline.csv predates emitter_df_bin: rerun v2_fig_extract.py assign-baseline"
        raise ValueError(msg)
    meta = json.loads((C.SNIPPETS / "assign_baseline.json").read_text())
    params_path = C.SNIPPETS / "assign_params.csv"
    if not params_path.exists():
        raise FileNotFoundError(f"{params_path} missing: run scripts/v2_assign_params.py (root env)")
    params = pd.read_csv(params_path)
    params_meta = json.loads((C.SNIPPETS / "assign_params.json").read_text())
    n_tp_05 = int(len(a))
    a = a.merge(params[[*PARAMS_KEYS, "height_hz", "width10_s"]], on=PARAMS_KEYS, how="left",
                validate="one_to_one", indicator=True)
    if (a["_merge"] != "both").any():
        msg = (f"{int((a['_merge'] != 'both').sum())} true positives of assign_baseline.csv "
               "are not in assign_params.csv: rerun scripts/v2_assign_params.py")
        raise ValueError(msg)
    a = a[a["score"] >= THR]  # true positives at the operating threshold
    allp = {k: float(a[k].mean()) for k in ("head_correct", "edge_correct",
                                            "edge_offset_correct")}
    n_all = int(len(a))
    # by the emitter's ΔF to the nearest other live track (not the evaluation's
    # closest pair of any two tracks)
    a = a[a["emitter_df_bin"].isin(DF_BINS)].copy()
    report["assignment"] = {
        "threshold": THR,
        "all_tp": allp,
        "n_all_tp": n_all,
        "n_tp_evaluation": n_tp_05,  # true positives of the evaluation (at 0.5)
        "emitter_df_known": {k: float(a[k].mean()) for k in allp},
        "n_emitter_df_known": int(len(a)),
        "offset_hz": meta["offset_hz"],
        "params_join": {k: params_meta[k] for k in ("n_tp", "n_with_truth", "n_truth_nan",
                                                    "n_edge_clipped", "n_chirps", "n_scenes")},
    }
    return a


def assignment_panel(ax, a, report):
    """Simulated: emitter correct by the emitter's ΔF (the dual-head's emitter)."""
    by_df = {}
    for name, draw in (("head_correct", True), ("edge_offset_correct", False)):
        g = a.groupby("emitter_df_bin")[name].agg(["sum", "size"]).reindex(DF_BINS)
        k, n = g["sum"].to_numpy(float), g["size"].to_numpy(float)
        if draw:
            p, lo, hi = proportion_points(ax, DF_POS, k, n, SERIES["dualhead"], logx=True)
        else:  # the box-position baseline: stats only (not drawn)
            p, (lo, hi) = k / n, C.wilson(k, n)
        by_df[name] = by_bin(DF_KEYS, g["sum"], g["size"], p, lo, hi)
    n = a.groupby("emitter_df_bin").size().reindex(DF_BINS)
    df_axis(ax)
    n_row(ax, DF_POS, n, N_Y)
    report["assignment"]["by_df"] = by_df


def detection_set(report) -> pd.DataFrame:
    """The visible simulated test chirps of A (one row per chirp label) with
    their truth and whether they are detected at the operating threshold;
    writes the counts to ``report["sim_recall"]``.

    Their number and the detected ones must be A's operating point (the
    evaluation's matching re-run at the threshold, ``v2_assign_params.py``).
    """
    path = C.SNIPPETS / "detect_params.csv"
    if not path.exists():
        msg = (f"{path} missing: run scripts/v2_assign_params.py --threshold {fmt_thr(THR)} "
               "(root env)")
        raise FileNotFoundError(msg)
    meta = json.loads((C.SNIPPETS / "detect_params.json").read_text())
    if not np.isclose(meta["threshold"], THR):
        msg = (f"detect_params.csv is at threshold {meta['threshold']}, the run's is {THR}: "
               f"rerun scripts/v2_assign_params.py --threshold {fmt_thr(THR)}")
        raise ValueError(msg)
    d = pd.read_csv(path)
    if d.duplicated(DETECT_KEYS).any():
        raise ValueError("detect_params.csv: a chirp label in more than one row")
    op = report["sim"]["dualhead"]
    n, k = int(len(d)), int(d["detected"].sum())
    if (n, k) != (int(op["tp"] + op["fn"]), int(op["tp"])):
        msg = (f"detect_params.csv: {k} of {n} chirp labels detected, A's operating point "
               f"{int(op['tp'])} of {int(op['tp'] + op['fn'])}")
        raise ValueError(msg)
    report["sim_recall"] = {
        "threshold": THR, "n_labels": n, "n_detected": k, "recall": k / n,
        "join": {key: meta[key] for key in ("n_with_truth", "n_truth_nan", "n_edge_clipped",
                                            "n_chirps", "n_scenes",
                                            "n_tp_reference_score_ge_threshold")},
    }
    return d


def size_panel(ax, a, d, column, scale, edges, keys, lims, name, report):
    """Simulated, by the chirp's true size (octave bins): recall among the
    visible chirps ``d`` (squares) and emitter correct among the true
    positives ``a`` (circles); n: chirp labels per bin (recall's n)."""
    pos = octave_positions(edges)
    drawn = {}
    for key, table, dx in (("detected", d, -0.035), ("head_correct", a, 0.035)):
        b = size_bins(table[column] * scale, edges, keys)
        g = table.groupby(b, observed=False)[key].agg(["sum", "size"]).reindex(keys)
        if (g["size"] == 0).any():
            raise ValueError(f"{name}, {key}: empty bins {list(g.index[g['size'] == 0])}")
        p, lo, hi = proportion_points(ax, pos, g["sum"].to_numpy(float),
                                      g["size"].to_numpy(float), SIZE_SERIES[key], dx=dx,
                                      logx=True, zorder=5 if key == "head_correct" else 4)
        drawn[key] = (g, p, lo, hi)
    log_xaxis(ax, lims, edges)
    despine(ax)
    n_row(ax, pos, drawn["detected"][0]["size"].to_numpy(), SIZE_N_Y)

    g, p, lo, hi = drawn["detected"]
    report["sim_recall"][name] = {
        "detected": by_bin(keys, g["sum"], g["size"], p, lo, hi),
        "p_range": {"min": float(np.min(p)), "max": float(np.max(p))},
        "n": int(g["size"].sum()), "edges": list(edges),
    }
    g, p, lo, hi = drawn["head_correct"]
    out = {"head_correct": by_bin(keys, g["sum"], g["size"], p, lo, hi),
           "p_range": {"min": float(np.min(p)), "max": float(np.max(p))},
           "n": int(g["size"].sum()), "edges": list(edges)}
    # the same by Δf stratum: does size matter where assignment is hard?
    b = size_bins(a[column] * scale, edges, keys)
    for stratum, mask in (("df_below_10_hz", a["emitter_df_hz"] < 10),
                          ("df_10_hz_up", a["emitter_df_hz"] >= 10)):
        gs = a[mask].groupby(b[mask], observed=False)["head_correct"].agg(["sum", "size"])
        gs = gs.reindex(keys)
        ks, ns = gs["sum"].to_numpy(float), gs["size"].to_numpy(float)
        ls, hs = C.wilson(ks, ns)
        out[stratum] = by_bin(keys, gs["sum"], gs["size"], ks / ns, ls, hs)
        out[f"{stratum}_p_range"] = {"min": float(np.min(ks / ns)), "max": float(np.max(ks / ns))}
    report["assignment"][name] = out


def series_handles(series: dict) -> list:
    """Key entries (line, marker) of the given series."""
    return [Line2D([], [], color=s["color"], lw=LW, ls=s.get("ls", "-"),
                   marker=s.get("marker", "o"), ms=PT_MS, mfc=s["mfc"], mec=s["mec"],
                   mew=s["mew"], label=s["label"])
            for s in series.values()]


# ---------------------------------------------------------------------------
# chirp parameters
# ---------------------------------------------------------------------------
DECODER_DOT = 0.6        # marker area of the decoder panels' points [pt^2]
DECODER_VMIN = -0.35     # colour scale of the density rank (0..1) starts above the palest mint
DECODER_REGION = 0.95    # the contour encloses this share of the chirps (densest first)
DECODER_GRID = 220       # KDE grid per axis for the contour
REGION_LW = 0.35
REGION_COLOR = "0.6"
REGION_LS = (0, (2.5, 1.5))


def decoder_panel(ax, sim, param, scale, unit, lims, ticks, report):
    """Estimate vs truth: points coloured by local density, the 95 % region dashed."""
    from scipy.stats import gaussian_kde

    true = sim[f"true_{param}"].to_numpy() * scale
    est = sim[f"cal_{param}"].to_numpy() * scale
    points = np.vstack([np.log10(true), np.log10(est)])
    kde = gaussian_kde(points)  # Scott's bandwidth, in log10 units
    density = kde(points)
    order = np.argsort(density, kind="stable")  # densest drawn last (on top)
    rank = np.empty(len(order))
    rank[order] = np.arange(len(order)) / max(len(order) - 1, 1)
    ax.scatter(true[order], est[order], c=rank[order], cmap=C.DENSITY_CMAP,
               vmin=DECODER_VMIN, vmax=1, s=DECODER_DOT, lw=0, rasterized=True, zorder=1)
    g = np.linspace(np.log10(lims[0]), np.log10(lims[1]), DECODER_GRID)
    gx, gy = np.meshgrid(g, g)
    grid = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    level = float(np.quantile(density, 1 - DECODER_REGION))
    ax.contour(10 ** gx, 10 ** gy, grid, levels=[level], colors=REGION_COLOR,
               linewidths=REGION_LW, linestyles=[REGION_LS], zorder=2)
    ax.plot(lims, lims, color="k", lw=REF_LW, ls=(0, (3, 2)), zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_aspect("equal")
    despine(ax)
    err = np.abs(est - true)
    beyond = int(((true < lims[0]) | (true > lims[1]) | (est < lims[0]) | (est > lims[1])).sum())
    if beyond:
        print(f"decoder panel {param}: {beyond} points beyond the axes {lims}")
    out = {  # errors in `unit`
        "threshold": THR,
        "n_sim": int(len(true)),
        "n_beyond_axes": beyond,
        "region_share": DECODER_REGION,
        "median_abs_error": float(np.median(err)),
        "median_rel_error": float(np.median(err / true)),
        "median_signed_error": float(np.median(est - true)),
    }
    # keyed by the parameter in the unit of the errors: height_hz, fwhm_ms, width10_ms
    report.setdefault("decoder", {})[f"{param.rsplit('_', 1)[0]}_{unit.lower()}"] = out
    return out


def decoder_legend(ax):
    """The key of I-K (they share it): the 95 % region and the identity, in I's empty upper left."""
    share = rf"{round(100 * DECODER_REGION)}\,\%"
    handles = [
        Line2D([], [], color=REGION_COLOR, lw=REGION_LW, ls=REGION_LS,
               label=f"{share} of chirps"),
        Line2D([], [], color="k", lw=REF_LW, ls=(0, (3, 2)), label="Identity"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.04), frameon=False,
              handlelength=2.2, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.3)


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    print(f"detections {C.DETECTIONS}, operating threshold {THR}, real data {REAL_DATASET}")
    report: dict = {"operating_threshold": THR}
    fig = plt.figure(figsize=(W * C.CM, H * C.CM))

    # layout [cm] (v2_performance_layout): one row per pipeline stage:
    # detection (A-D: simulated A, B | real C, D sharing the agreement axis,
    # under two group headings), assignment (E-H: E, F by ΔF sharing the
    # y-axis and one ΔF label | G, H by true chirp size sharing the y-axis),
    # chirp parameters (I-K); the key above
    boxes = L.layout()
    ax = {name: C.axes_cm(fig, b.x, b.y, b.w, b.h) for name, b in boxes.items()}
    ax_a, ax_b, ax_c, ax_d = (ax[k] for k in "ABCD")
    ax_e, ax_f, ax_g, ax_h = (ax[k] for k in "EFGH")
    ax_i, ax_j, ax_k = (ax[k] for k in "IJK")

    # --- detection -------------------------------------------------------------
    pr_panel(ax_a, ["dualhead"], lambda m: C.PP_EVAL[m] / "sim_test_after", report)
    op = report["sim"]["dualhead"]
    C.annotate(ax_a, rf"score $\geq$ {fmt_thr(THR)}", xy=(op["recall"], op["precision"]),
               xytext=(0.87, 0.9), color="k", rad=0.2, shrink_b=4)
    benchmark_panel(ax_b, report)
    agreement_panel(ax_c, report)
    status_panel(ax_d, report)
    ax_d.tick_params(labelleft=False)

    # --- assignment ------------------------------------------------------------
    a = assignment_set(report)
    d = detection_set(report)
    assignment_panel(ax_e, a, report)
    benchmark_emitter_panel(ax_f, report)
    # one ΔF label centred under the pair E, F
    ax_e.set_xlabel(r"$\Delta f$ to the nearest other fish [Hz]")
    pe, pf = ax_e.get_position(), ax_f.get_position()
    ax_e.xaxis.label.set_x(((pe.x0 + pf.x1) / 2 - pe.x0) / pe.width)
    size_panel(ax_g, a, d, "width10_s", 1e3, WIDTH_EDGES_MS, WIDTH_KEYS, (5.0, 320.0),
               "by_width10", report)
    size_panel(ax_h, a, d, "height_hz", 1.0, HEIGHT_EDGES_HZ, HEIGHT_KEYS, (25.0, 800.0),
               "by_height", report)
    assign_yaxis(ax_g, "Fraction", ylim=SIZE_YLIM, ticks=(0.6, 0.8, 1.0))
    assign_yaxis(ax_h, None, ylim=SIZE_YLIM, ticks=(0.6, 0.8, 1.0))
    ax_g.set_xlabel(r"True 10\,\% width [ms]")
    ax_h.set_xlabel("True height [Hz]")
    # the key of G and H in G's empty middle (between the n row and the data)
    ax_g.legend(handles=series_handles(SIZE_SERIES), loc="center left",
                bbox_to_anchor=(0.0, 0.52), frameon=False, handlelength=2.2,
                handletextpad=0.5, borderaxespad=0.1, labelspacing=0.3)

    # --- chirp parameters --------------------------------------------------------
    sim = pd.read_parquet(C.DECODER_EVAL / "predictions.parquet")
    sim = sim[sim["score"] >= THR]
    octave = sim["template_id"].isin(C.OCTAVE_TEMPLATES)
    sim = sim[~octave]
    decoder_panel(ax_i, sim, "height_hz", 1.0, "Hz", (20, 800),
                  [30, 100, 300], report)
    decoder_panel(ax_j, sim, "fwhm_s", 1e3, "ms", (1.5, 600),
                  [3, 10, 30, 100, 300], report)
    decoder_panel(ax_k, sim, "width10_s", 1e3, "ms", (3, 700),
                  [3, 10, 30, 100, 300], report)
    report["decoder"]["n_octave_excluded"] = int(octave.sum())
    ax_i.set_xlabel("True height [Hz]")
    ax_i.set_ylabel("Estimated height [Hz]")
    ax_j.set_xlabel("True FWHM [ms]")
    ax_j.set_ylabel("Estimated FWHM [ms]")
    ax_k.set_xlabel(r"True 10\,\% width [ms]")
    ax_k.set_ylabel("Estimated\n" r"10\,\% width [ms]")  # two lines: longer than the square
    decoder_legend(ax_i)  # one key for I-K (author, 2026-11-27)

    # --- key, group headings, letters ---------------------------------------------
    fig.legend(handles=series_handles({m: {**SERIES[m], "label": LABELS[m]}
                                       for m in ("dualhead", "human")}),
               loc="center", ncol=2, frameon=False, fontsize="small", handlelength=1.6,
               handletextpad=0.5, columnspacing=1.8,
               bbox_to_anchor=(L.W / 2 / W, L.KEY_Y / H))
    letters = L.letters()
    # headings over the detection and assignment rows (E-H are all simulated;
    # none over I-K: author, 2026-11-25)
    for text, first, last in (("Simulated", "A", "B"), ("Real recordings", "C", "D"),
                              ("Simulated", "E", "H")):
        x0 = boxes[first].x - letters[first] - L.LETTER_HALF
        group_heading(fig, text, x0, boxes[first].x, boxes[last].right,
                      L.heading_y(boxes[first].top))
    C.letters_cm(fig, [(ax[k], k, letters[k], L.LETTER_DY) for k in "ABCDEFGHIJK"])

    print(extent_table(fig, ax))
    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))
    write_stats("v2_performance", report)


if __name__ == "__main__":
    main()
