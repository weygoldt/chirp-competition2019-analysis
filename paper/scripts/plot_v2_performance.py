"""Figure: performance of chirp detection pipeline v2.

Replaces ``plot_pipeline_performances_manuscript_supplement_supplement.pdf``.
Framing (author, 2026-11-16): on real recordings a human expert is not
ground truth, so the real-data panels show AGREEMENT WITH THE EXPERT (used to
choose the operating threshold); only the simulator gives absolute
performance (model and human expert vs simulator truth). One row per
pipeline stage (author, 2026-11-16): detection (A-D), assignment (E, F),
chirp parameters (G, H). Only the production detector (RT-DETR dual-head)
is shown; YOLO26s has no emitter head and is no like-for-like comparison.
The operating threshold is the run's (``meta.json`` of PAPER_DETECTIONS'
run; 0.8 for rtdetr_gapfill_t080); every operating point follows it.

A   Simulated held-out test scenes (unseen fish on unseen background
    recordings; 9,401 visible chirps), after post-processing: precision
    against recall over the score threshold,
    ``runs/dualhead_r18_sim_v4_r1/eval_postprocess/sim_test_after``;
    operating point from the curve's row at the operating threshold.
B   Blind benchmark on 1,010 simulated items (536 scored true chirps):
    recall of the human expert and of RT-DETR dual-head (score >= operating
    threshold, kept), by chirp SNR (``labels/benchmark/report_v2/items.csv``;
    unweighted, Wilson 95 % intervals).
C   Real recordings (2019/2020 competitions), agreement with one expert
    (blind labelling session real_v1), drawn like A as one detector curve
    over the score threshold: share of the expert's window chirps the
    detector found (x; ``recall.csv``) against share of detections the
    expert judged chirps (y; ``precision_at_threshold.csv``);
    design-weighted, 95 % bootstrap intervals at 0.5, 0.9 and the operating
    threshold. One colour: both measures describe the detector (two blue
    lines over the threshold read as detector vs expert). Shares the y-axis
    with D.
D   Detections the expert judged chirps, for detections assigned to the
    trial's winner vs loser, by score bin (0.5-0.7, 0.7-0.9, >= 0.9;
    ``precision_by_status.csv``).
E   Emitter assignment of the simulated true positives at the operating
    threshold (true positives of the evaluation at 0.5 with score >= the
    threshold), by the emitter's ΔF to the nearest other live track
    (``emitter_df_bin``): predicted emitter EODf vs the box position only
    (``v2_fig_extract.py assign-baseline``). Shares the axes with F.
F   The benchmark of B: emitter correct among each one's own detections
    (expert: items called chirp; detector: kept, score >= operating
    threshold), by the emitter's ΔF to the nearest other live fish
    (``df_nearest_hz``); unweighted, Wilson 95 % intervals.
G,H Chirp-parameter decoder for boxes at the operating threshold on the
    simulated test chirps of unseen fish (fish-disjoint template splits):
    calibrated estimate vs per-period truth
    (``params/.../eval/predictions.parquet``; density). The former real
    long-chirp comparison (``real_agreement.csv``) is dropped: 429 of its
    533 chirps come from fish whose templates were in the decoder's
    training split.

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_performance.py

Runtime: ~15 s cold, ~3 s warm. Prints every number it plots.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

import v2_fig_common as C
from style import loser_color, set_light_style, winner_color

NAME = "plot_v2_performance"
W, H = 17.4, 15.8  # figure size [cm]
THR = C.OPERATING
REAL_V1 = C.CHIRP_DATA / "labels" / "real_v1" / "report"
REAL_DATASET = "2019"  # the 2019/2020 tube competitions (the paper's data)
MARKED = (0.5, 0.9)  # thresholds marked in A besides the operating one
STATUS_BINS = ["0.5-0.7", "0.7-0.9", "0.9-1"]
STATUS_CENTRES = [0.6, 0.8, 0.95]  # centres of the score bins in D
DF_BINS = ["0-10", "10-25", "25-50", "50-100", "100+"]
DF_POS = [5.0, 15.8, 35.4, 70.7, 141.0]  # geometric bin centres; open bins at the edge
SNR_BINS = ["<-20 dB", "-20 to -10 dB", "-10 to 0 dB", "0-10 dB", ">=10 dB"]
SNR_POS = [-25.0, -15.0, -5.0, 5.0, 15.0]
OP_MS = 4.2  # operating-point marker
PT_MS = 3.6  # proportion markers


def fmt_thr(t: float) -> str:
    return f"{t:g}"


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
    prec = prec[prec.dataset == REAL_DATASET].set_index("threshold")
    rec = rec[rec.dataset == REAL_DATASET].set_index("threshold")
    thr = np.array(sorted(set(prec.index) & set(rec.index)))
    if not np.any(np.isclose(thr, THR)):
        msg = f"operating threshold {THR} not among the session's thresholds {thr}"
        raise ValueError(msg)
    col = C.MODEL_COLORS["dualhead"]
    y, ylo, yhi = (prec.loc[thr, k].to_numpy() for k in ("precision", "ci_low", "ci_high"))
    x, xlo, xhi = (rec.loc[thr, k].to_numpy() for k in ("recall", "ci_low", "ci_high"))
    marked = np.array([any(np.isclose(t, m) for m in (*MARKED, THR)) for t in thr])
    ax.plot(x, y, color=col, lw=1.0, zorder=2, solid_joinstyle="round")
    ax.plot(x[~marked], y[~marked], "o", ms=2.2, mfc=col, mec=col, mew=0.6, zorder=3,
            clip_on=False)
    ax.errorbar(x[marked], y[marked], xerr=[x[marked] - xlo[marked], xhi[marked] - x[marked]],
                yerr=[y[marked] - ylo[marked], yhi[marked] - y[marked]], fmt="o", ms=PT_MS,
                mfc=col, mec="k", mew=0.6, ecolor=col, elinewidth=0.8, capsize=0, zorder=4,
                clip_on=False)
    k_op = int(np.flatnonzero(np.isclose(thr, THR))[0])
    C.annotate(ax, rf"score $\geq$ {fmt_thr(THR)}", xy=(x[k_op], y[k_op]),
               xytext=(0.55, 0.45), color="k", rad=0.2, shrink_b=4)
    # the other marked thresholds: their value next to the point
    for t, dxy in ((0.5, (-4, -9)), (0.9, (-5, 5))):
        k = int(np.flatnonzero(np.isclose(thr, t))[0])
        ax.annotate(fmt_thr(t), xy=(x[k], y[k]), xytext=dxy, textcoords="offset points",
                    ha="right", va="center", fontsize="small", color=C.MUTED)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Expert's chirps found")
    ax.set_ylabel("Detections judged chirps")
    sns.despine(ax=ax, offset=4, trim=False)
    as_dict = lambda v, lo, hi: {fmt_thr(t): {"value": float(a), "ci": [float(b), float(c)]}  # noqa: E731
                                 for t, a, b, c in zip(thr, v, lo, hi, strict=True)}
    report["agreement"] = {
        "dataset": REAL_DATASET, "thresholds": thr.tolist(), "marked": [*MARKED, THR],
        "detections_judged_chirps": as_dict(y, ylo, yhi),
        "expert_chirps_found": as_dict(x, xlo, xhi),
        "n_candidates": {fmt_thr(t): int(prec.loc[t, "n"]) for t in thr},
        "windows": int(rec.loc[thr[0], "windows"]), "marks": int(rec.loc[thr[0], "marks"]),
    }


def status_panel(ax, report):
    s = pd.read_csv(REAL_V1 / "precision_by_status.csv")
    s = s[(s.dataset == REAL_DATASET) & s.scores.isin(STATUS_BINS)]
    out = {}
    for status, col, dx in (("winner", winner_color, -0.012), ("loser", loser_color, 0.012)):
        g = s[s.status == status].set_index("scores").reindex(STATUS_BINS)
        xx = np.asarray(STATUS_CENTRES) + dx
        p = g.precision.to_numpy()
        ax.plot(xx, p, color=col, lw=0.8, zorder=3, clip_on=False)
        ax.errorbar(xx, p, yerr=[p - g.ci_low.to_numpy(), g.ci_high.to_numpy() - p], fmt="o",
                    ms=PT_MS, mfc=col, mec="k", mew=0.6, ecolor=col, elinewidth=0.9,
                    capsize=0, zorder=4, clip_on=False, label=status)
        out[status] = g[["n", "N_hat", "precision", "ci_low", "ci_high"]].round(4).to_dict(orient="index")
    # score bins as points at their centres between the bin edges (ticks)
    ax.set_xticks([0.5, 0.7, 0.9, 1.0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Detections judged chirps")
    ax.set_xlabel("Detector score")
    sns.despine(ax=ax, offset=4, trim=False)
    # the empty upper left (low-score bins sit near zero)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04), frameon=False,
              handletextpad=0.4, borderaxespad=0.1)
    report["by_status"] = out


# ---------------------------------------------------------------------------
# simulated: detection
# ---------------------------------------------------------------------------
def pr_panel(ax, models, base, report):
    for m in models:
        curve = pd.read_csv(base(m) / "pr_curve.csv")
        curve = curve[curve["level"] == "all"].sort_values("threshold")
        # the operating point: the curve's row at the threshold (the
        # evaluation re-matches at every threshold)
        op = curve[curve.threshold >= THR].iloc[0]
        report[f"sim/{m}"] = {k: float(op[k]) for k in ("threshold", "tp", "fp", "fn",
                                                       "precision", "recall")}
        col = C.MODEL_COLORS[m]
        ax.plot(curve["recall"], curve["precision"], color=col,
                lw=1.5 if m == "dualhead" else 1.0, zorder=3 if m == "dualhead" else 2,
                solid_joinstyle="round")
        ax.plot(op["recall"], op["precision"], "o", ms=OP_MS, mfc=col, mec="k",
                mew=0.6, zorder=5, clip_on=False)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.8, 1.004)
    ax.set_ylim(0.85, 1.006)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xticklabels(["0.8", "", "0.9", "", "1.0"])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    sns.despine(ax=ax, offset=4, trim=True)


def proportion_points(ax, x, k, n, color, *, filled=True, dx=0.0, logx=False, zorder=4):
    p = k / n
    lo, hi = C.wilson(k, n)
    xx = np.asarray(x) * (10**dx if logx else 1) + (0 if logx else dx)
    ax.plot(xx, p, color=color, lw=0.8, zorder=zorder - 1, clip_on=False)
    ax.errorbar(xx, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                fmt="o", ms=PT_MS, mfc=color if filled else "white",
                mec="k" if filled else color, mew=0.6 if filled else 0.9, ecolor=color,
                elinewidth=0.9, capsize=0, zorder=zorder, clip_on=False)
    return p, lo, hi


def n_row(ax, xs, ns, y, fmt="{:,}"):
    for x, n in zip(xs, ns, strict=True):
        ax.text(x, y, fmt.format(int(n)), ha="center", va="bottom", fontsize="small",
                color=C.MUTED)


def benchmark_panel(ax, report):
    b = pd.read_csv(C.BENCHMARK / "items.csv", low_memory=False)
    t = b[(b["truth"] == "chirp") & b["scored"].astype(bool)].copy()
    # the model at the operating threshold: kept and score >= threshold
    # (``dualhead_chirp`` is kept and score >= 0.5)
    t["model_chirp"] = t["dualhead_chirp"].astype(bool) & (t["dualhead_score"] >= THR)
    t["model_emitter_ok"] = t["dualhead_emitter_ok"].astype(bool) & t["model_chirp"]
    out = {}
    for name, key, dx in (("human_chirp", "human", -0.9), ("model_chirp", "dualhead", 0.9)):
        g = t.groupby("snr_bin")[name].agg(["sum", "size"]).reindex(SNR_BINS)
        p, lo, hi = proportion_points(ax, SNR_POS, g["sum"].to_numpy(float),
                                      g["size"].to_numpy(float), C.MODEL_COLORS[key], dx=dx)
        out[name] = {"by_snr": dict(zip(SNR_BINS, np.round(p, 4).tolist(), strict=True)),
                     "lo": np.round(lo, 4).tolist(), "hi": np.round(hi, 4).tolist(),
                     "n": g["size"].astype(int).tolist(), "overall": float(t[name].mean())}
    for name in ("human_emitter_ok", "model_emitter_ok"):
        out[name] = float(t[name].astype(bool).mean())
    n = t.groupby("snr_bin").size().reindex(SNR_BINS)
    n_row(ax, SNR_POS, n, 0.505, fmt="{}")
    ax.set_xlim(-30, 20)
    ax.set_xticks([-20, -10, 0, 10])
    ax.set_xlabel("Chirp SNR [dB]")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    sns.despine(ax=ax, offset=4, trim=False)
    out["n_true_chirps"] = int(len(t))
    out["n_items"] = int(len(b))
    out["threshold"] = THR
    report["benchmark"] = out


# ---------------------------------------------------------------------------
# simulated: assignment
# ---------------------------------------------------------------------------
def df_axis(ax, *, ylabel=True):
    """ΔF axis of E and F (log x, open outer bins) and their shared y-axis."""
    ax.set_xscale("log")
    ax.set_xlim(3.3, 210)
    ax.xaxis.set_major_locator(FixedLocator([10, 25, 50, 100]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    if ylabel:
        ax.set_ylabel("Correctly assigned")
    else:
        ax.tick_params(labelleft=False)
    sns.despine(ax=ax, offset=4, trim=False)


def df_bins(df_hz: pd.Series) -> pd.Series:
    """The evaluation's ΔF bins (left-closed edges 0, 10, 25, 50, 100 Hz)."""
    return pd.cut(df_hz, [0, 10, 25, 50, 100, np.inf], labels=DF_BINS, right=False)


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
        "dualhead": t["dualhead_chirp"].astype(bool) & (t["dualhead_score"] >= THR),
    }
    ok = {"human": "human_emitter_ok", "dualhead": "dualhead_emitter_ok"}
    out = {}
    for key, dx in (("human", -0.025), ("dualhead", 0.025)):
        d = t[called[key]]
        if (d[ok[key]].astype(bool) & ~called[key][d.index]).any():
            raise ValueError(f"{ok[key]} set for items {key} did not call a chirp")
        g = d.groupby("df_bin", observed=False)[ok[key]].agg(["sum", "size"]).reindex(DF_BINS)
        p, lo, hi = proportion_points(ax, DF_POS, g["sum"].to_numpy(float),
                                      g["size"].to_numpy(float), C.MODEL_COLORS[key],
                                      dx=dx, logx=True)
        out[key] = {"by_df": dict(zip(DF_BINS, np.round(p, 4).tolist(), strict=True)),
                    "lo": np.round(lo, 4).tolist(), "hi": np.round(hi, 4).tolist(),
                    "n_called": g["size"].astype(int).tolist(),
                    "overall": float(d[ok[key]].astype(bool).mean()), "n": int(len(d))}
    n = t.groupby("df_bin", observed=False).size().reindex(DF_BINS)
    n_row(ax, DF_POS, n, 0.305, fmt="{}")
    df_axis(ax, ylabel=False)
    out["n_true_chirps_per_bin"] = n.astype(int).tolist()
    out["n_df_unknown"] = n_unknown
    out["threshold"] = THR
    report["benchmark_emitter"] = out


def assignment_panel(ax, report):
    a = pd.read_csv(C.SNIPPETS / "assign_baseline.csv")
    if "emitter_df_bin" not in a.columns:
        msg = "assign_baseline.csv predates emitter_df_bin: rerun v2_fig_extract.py assign-baseline"
        raise ValueError(msg)
    meta = json.loads((C.SNIPPETS / "assign_baseline.json").read_text())
    n_tp_05 = int(len(a))
    a = a[a["score"] >= THR]  # true positives at the operating threshold
    allp = {k: float(a[k].mean()) for k in ("head_correct", "edge_correct",
                                            "edge_offset_correct")}
    n_all = int(len(a))
    # by the emitter's ΔF to the nearest other live track (not the evaluation's
    # closest pair of any two tracks)
    a = a[a["emitter_df_bin"].isin(DF_BINS)]
    twofish = {k: float(a[k].mean()) for k in allp}
    rows = []
    for name, col, dx, filled in (
        ("head_correct", C.MODEL_COLORS["dualhead"], -0.025, True),
        ("edge_offset_correct", C.MODEL_COLORS["edge"], 0.025, False),
    ):
        g = a.groupby("emitter_df_bin")[name].agg(["sum", "size"]).reindex(DF_BINS)
        p, lo, hi = proportion_points(ax, DF_POS, g["sum"].to_numpy(float),
                                      g["size"].to_numpy(float), col, dx=dx, logx=True,
                                      filled=filled)
        rows.append(pd.DataFrame({"method": name, "df_bin": DF_BINS, "k": g["sum"],
                                  "n": g["size"], "p": p, "lo": lo, "hi": hi}))
    n = a.groupby("emitter_df_bin").size().reindex(DF_BINS)
    n_row(ax, DF_POS, n, 0.305)
    df_axis(ax)
    # the dual-head series is in the figure key; only the baseline needs a label
    handles = [Line2D([], [], color=C.MODEL_COLORS["edge"], lw=0.8, marker="o", ms=PT_MS,
                      mfc="white", mec=C.MODEL_COLORS["edge"], mew=0.9,
                      label="box position only")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, 0.1),
              frameon=False, borderaxespad=0.1, handletextpad=0.4, handlelength=1.4)
    report["assignment"] = {
        "threshold": THR,
        "table": pd.concat(rows).round(4).to_dict(orient="records"),
        "all_tp": allp,
        "n_all_tp": n_all,
        "n_tp_at_0.5": n_tp_05,
        "emitter_df_known": twofish,
        "n_emitter_df_known": int(len(a)),
        "offset_hz": meta["offset_hz"],
    }


# ---------------------------------------------------------------------------
# chirp parameters
# ---------------------------------------------------------------------------
def decoder_panel(ax, sim, param, scale, unit, lims, ticks, report):
    true = sim[f"true_{param}"].to_numpy() * scale
    est = sim[f"cal_{param}"].to_numpy() * scale
    edges = np.geomspace(*lims, 56)
    h, _, _ = np.histogram2d(true, est, bins=[edges, edges])
    h = np.where(h > 0, h, np.nan)
    ax.pcolormesh(edges, edges, h.T, cmap=C.DENSITY_CMAP,
                  norm=LogNorm(vmin=1, vmax=np.nanmax(h)), rasterized=True, zorder=1,
                  linewidth=0)
    ax.plot(lims, lims, color="k", lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_aspect("equal")
    sns.despine(ax=ax, offset=4, trim=False)
    err = np.abs(est - true)
    out = {
        "threshold": THR,
        "n_sim": int(len(true)),
        "median_abs_error": float(np.median(err)),
        "median_rel_error": float(np.median(err / true)),
        "median_signed_error": float(np.median(est - true)),
        "unit": unit,
    }
    report[f"decoder/{param}"] = out
    return out


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {"operating_threshold": THR, "detections": C.DETECTIONS}
    fig = plt.figure(figsize=(W * C.CM, H * C.CM))

    # layout [cm]: one row per pipeline stage: detection (A-D: simulated A, B;
    # real C, D sharing the agreement axis), assignment (E, F sharing both
    # axes), chirp parameters (G, H); the model key above the top row
    ph, sq = 3.3, 3.4
    y1, y2, y3 = 11.2, 6.35, 1.25
    ax_a = C.axes_cm(fig, 1.45, y1, 2.9, ph)
    ax_b = C.axes_cm(fig, 5.55, y1, 2.9, ph)
    ax_c = C.axes_cm(fig, 9.75, y1, 3.7, ph)
    ax_d = C.axes_cm(fig, 14.05, y1, 2.9, ph)
    ax_e = C.axes_cm(fig, 1.45, y2, 7.1, ph)
    ax_f = C.axes_cm(fig, 9.0, y2, 7.1, ph)
    ax_g = C.axes_cm(fig, 1.75, y3, sq, sq)
    ax_h = C.axes_cm(fig, 6.95, y3, sq, sq)

    # --- detection -------------------------------------------------------------
    pr_panel(ax_a, ["dualhead"], lambda m: C.PP_EVAL[m] / "sim_test_after", report)
    ax_a.set_title("Simulated", loc="left", pad=6)
    op = report["sim/dualhead"]
    C.annotate(ax_a, rf"score $\geq$ {fmt_thr(THR)}", xy=(op["recall"], op["precision"]),
               xytext=(0.865, 0.905), color="k", rad=0.2, shrink_b=4)
    benchmark_panel(ax_b, report)
    agreement_panel(ax_c, report)
    ax_c.set_title("Real recordings", loc="left", pad=6)
    status_panel(ax_d, report)
    ax_d.set_ylabel("")
    ax_d.tick_params(labelleft=False)

    # --- assignment ------------------------------------------------------------
    assignment_panel(ax_e, report)
    benchmark_emitter_panel(ax_f, report)
    # one ΔF label centred under the pair E, F
    ax_e.set_xlabel(r"$\Delta f$ to the nearest other fish [Hz]")
    pe, pf = ax_e.get_position(), ax_f.get_position()
    ax_e.xaxis.label.set_x(((pe.x0 + pf.x1) / 2 - pe.x0) / pe.width)

    # --- chirp parameters --------------------------------------------------------
    sim = pd.read_parquet(C.DECODER_EVAL / "predictions.parquet")
    sim = sim[sim["score"] >= THR]
    decoder_panel(ax_g, sim, "height_hz", 1.0, "Hz", (15, 1000),
                  [20, 50, 100, 200, 500], report)
    decoder_panel(ax_h, sim, "fwhm_s", 1e3, "ms", (2, 500),
                  [3, 10, 30, 100, 300], report)
    ax_g.set_xlabel("True height [Hz]")
    ax_g.set_ylabel("Estimated height [Hz]")
    ax_h.set_xlabel("True FWHM [ms]")
    ax_h.set_ylabel("Estimated FWHM [ms]")

    # --- key ---------------------------------------------------------------------
    handles = [Line2D([], [], color=C.MODEL_COLORS[m], lw=1.5 if m == "dualhead" else 1.0,
                      marker="o", ms=OP_MS - 0.6, mfc=C.MODEL_COLORS[m], mec="k", mew=0.6,
                      label=C.MODEL_LABELS[m])
               for m in ("dualhead", "human")]
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize="small",
               handlelength=1.6, handletextpad=0.5, columnspacing=1.8,
               bbox_to_anchor=(0.5, (y1 + ph + 0.95) / H))
    C.letters_cm(fig, [(ax_a, "A", 1.15, 0.3), (ax_b, "B", 1.15, 0.3),
                       (ax_c, "C", 1.15, 0.3), (ax_d, "D", 0.3, 0.3),
                       (ax_e, "E", 1.15, 0.3), (ax_f, "F", 0.3, 0.3),
                       (ax_g, "G", 1.45, 0.3), (ax_h, "H", 1.45, 0.3)])

    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
