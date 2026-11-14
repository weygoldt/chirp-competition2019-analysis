"""Figure: performance of chirp detection pipeline v2.

Replaces ``plot_pipeline_performances_manuscript_supplement_supplement.pdf``
(cross-validation curves of the v1 YOLOv8 variants). Panels grouped by
pipeline stage through spacing (detection: top row; assignment: bottom
left; chirp parameters: bottom right), one colour per model throughout;
every plotted number is recomputed here from saved per-item or
per-threshold data:

A   Real recordings, hand labels (365 chirps in 6 labelled regions of 4
    held-out recordings): precision against recall over the score threshold
    for the RT-DETR dual-head detector (production) and YOLO26s (boxes only,
    reference). Curves: ``pr_curve.csv`` (level "all") of
    ``eval/real_labels_365/<model>``; operating points (score >= 0.5):
    counted from that evaluation's ``items.csv``.
B   Simulated held-out test scenes (unseen fish on unseen background
    recordings; 9,401 visible chirps), after post-processing: the same for
    RT-DETR dual-head and YOLO26s (``runs/<model>/eval_postprocess/
    sim_test_after``).
C   Blind benchmark on 1,010 simulated items (536 scored true chirps):
    recall of one human expert and of the RT-DETR dual-head detector
    (>= 0.5), by chirp SNR (``labels/benchmark/report_v2/items.csv``;
    unweighted, Wilson 95 % intervals).
D   Emitter assignment on the simulated true positives at score >= 0.5
    (after post-processing), by ΔF between the emitter and the closest
    other fish: nearest live track to the predicted emitter EODf against
    the box position only (nearest live track to the box's lower edge +
    41 Hz, the median offset of the training chirps). Per-item data
    recomputed by ``v2_fig_extract.py assign-baseline``.
E,F Chirp-parameter decoder on 7,678 simulated test chirps of unseen fish
    (calibrated estimate against the noise-free per-EOD-period truth;
    ``params/.../eval/predictions.parquet``; density) and on 513 real long
    chirps of the competition recordings (estimate against an independent
    frequency-trace reference; ``real_agreement.csv``; black).

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
from style import set_light_style

NAME = "plot_v2_performance"
W, H = 17.4, 12.4  # figure size [cm]
DF_BINS = ["0-10", "10-25", "25-50", "50-100", "100+"]
DF_POS = [5.0, 15.8, 35.4, 70.7, 141.0]  # geometric bin centres; open bins at the edge
SNR_BINS = ["<-20 dB", "-20 to -10 dB", "-10 to 0 dB", "0-10 dB", ">=10 dB"]
SNR_POS = [-25.0, -15.0, -5.0, 5.0, 15.0]
OP_MS = 4.2  # operating-point marker
PT_MS = 3.6  # proportion markers


def counts(items: pd.DataFrame) -> dict:
    o = items["outcome"].value_counts()
    tp, fp, fn = int(o.get("tp", 0)), int(o.get("fp", 0)), int(o.get("fn", 0))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp),
            "recall": tp / (tp + fn)}


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def pr_panel(ax, models, base, kind, report, ylabel=True):
    for m in models:
        d = base(m)
        curve = pd.read_csv(d / "pr_curve.csv")
        curve = curve[curve["level"] == "all"].sort_values("threshold")
        op = counts(pd.read_csv(d / "items.csv"))
        report[f"{kind}/{m}"] = op
        col = C.MODEL_COLORS[m]
        ax.plot(curve["recall"], curve["precision"], color=col,
                lw=1.5 if m == "dualhead" else 1.0, zorder=3 if m == "dualhead" else 2,
                solid_joinstyle="round")
        ax.plot(op["recall"], op["precision"], "o", ms=OP_MS, mfc=col, mec="k",
                mew=0.6, zorder=5, clip_on=False)
    ax.set_xlabel("Recall")
    ax.set_xlim(0.8, 1.004)
    ax.set_ylim(0.85, 1.006)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xticklabels(["0.8", "", "0.9", "", "1.0"])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    if ylabel:
        ax.set_ylabel("Precision")
    else:
        ax.set_yticklabels([])
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
    t = b[(b["truth"] == "chirp") & b["scored"].astype(bool)]
    out = {}
    for name, key, dx in (("human_chirp", "human", -0.9), ("dualhead_chirp", "dualhead", 0.9)):
        g = t.groupby("snr_bin")[name].agg(["sum", "size"]).reindex(SNR_BINS)
        p, lo, hi = proportion_points(ax, SNR_POS, g["sum"].to_numpy(), g["size"].to_numpy(),
                                      C.MODEL_COLORS[key], dx=dx)
        out[name] = {"by_snr": dict(zip(SNR_BINS, np.round(p, 4).tolist(), strict=True)),
                     "lo": np.round(lo, 4).tolist(), "hi": np.round(hi, 4).tolist(),
                     "n": g["size"].astype(int).tolist(), "overall": float(t[name].mean())}
    for name in ("human_emitter_ok", "dualhead_emitter_ok"):
        out[name] = float(t[name].mean())
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
    report["benchmark"] = out


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------
def assignment_panel(ax, report):
    a = pd.read_csv(C.SNIPPETS / "assign_baseline.csv")
    meta = json.loads((C.SNIPPETS / "assign_baseline.json").read_text())
    allp = {k: float(a[k].mean()) for k in ("head_correct", "edge_correct",
                                            "edge_offset_correct")}
    n_all = int(len(a))
    a = a[a["df_bin"].isin(DF_BINS)]
    twofish = {k: float(a[k].mean()) for k in allp}
    rows = []
    for name, col, dx, filled in (
        ("head_correct", C.MODEL_COLORS["dualhead"], -0.025, True),
        ("edge_offset_correct", C.MODEL_COLORS["edge"], 0.025, False),
    ):
        g = a.groupby("df_bin")[name].agg(["sum", "size"]).reindex(DF_BINS)
        p, lo, hi = proportion_points(ax, DF_POS, g["sum"].to_numpy(), g["size"].to_numpy(),
                                      col, dx=dx, logx=True, filled=filled)
        rows.append(pd.DataFrame({"method": name, "df_bin": DF_BINS, "k": g["sum"],
                                  "n": g["size"], "p": p, "lo": lo, "hi": hi}))
    n = a.groupby("df_bin").size().reindex(DF_BINS)
    n_row(ax, DF_POS, n, 0.305)
    ax.set_xscale("log")
    ax.set_xlim(3.3, 210)
    ax.xaxis.set_major_locator(FixedLocator([10, 25, 50, 100]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\Delta f$ to the other fish [Hz]")
    ax.set_ylabel("Correctly assigned")
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    sns.despine(ax=ax, offset=4, trim=False)
    handles = [
        Line2D([], [], color=C.MODEL_COLORS["dualhead"], lw=0.8, marker="o", ms=PT_MS,
               mfc=C.MODEL_COLORS["dualhead"], mec="k", mew=0.6,
               label=r"predicted emitter EOD$f$"),
        Line2D([], [], color=C.MODEL_COLORS["edge"], lw=0.8, marker="o", ms=PT_MS,
               mfc="white", mec=C.MODEL_COLORS["edge"], mew=0.9, label="box position only"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, 0.13),
              frameon=False, borderaxespad=0.1, handletextpad=0.4)
    report["assignment"] = {
        "table": pd.concat(rows).round(4).to_dict(orient="records"),
        "all_tp": allp,
        "n_all_tp": n_all,
        "two_or_more_fish": twofish,
        "n_two_or_more_fish": int(len(a)),
        "offset_hz": meta["offset_hz"],
    }


# ---------------------------------------------------------------------------
# chirp parameters
# ---------------------------------------------------------------------------
def decoder_panel(ax, sim, real, param, scale, unit, lims, ticks, report):
    true = sim[f"true_{param}"].to_numpy() * scale
    est = sim[f"cal_{param}"].to_numpy() * scale
    r_ref = real[f"ref_{param}"].to_numpy() * scale
    r_est = real[param].to_numpy() * scale
    edges = np.geomspace(*lims, 56)
    h, _, _ = np.histogram2d(true, est, bins=[edges, edges])
    h = np.where(h > 0, h, np.nan)
    ax.pcolormesh(edges, edges, h.T, cmap=C.DENSITY_CMAP,
                  norm=LogNorm(vmin=1, vmax=np.nanmax(h)), rasterized=True, zorder=1,
                  linewidth=0)
    ax.plot(r_ref, r_est, "o", ms=1.6, mfc="k", mec="none", alpha=0.6, zorder=3,
            rasterized=True)
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
    rr = np.abs(r_est - r_ref)
    out = {
        "n_sim": int(len(true)),
        "median_abs_error": float(np.median(err)),
        "median_rel_error": float(np.median(err / true)),
        "n_real": int(len(r_ref)),
        "real_median_abs_difference": float(np.median(rr)),
        "real_median_rel_difference": float(np.median(rr / r_ref)),
        "real_median_difference": float(np.median(r_est - r_ref)),
        "unit": unit,
    }
    report[f"decoder/{param}"] = out
    return out


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {}
    fig = plt.figure(figsize=(W * C.CM, H * C.CM))

    # layout [cm]: detection on top (A, B share the precision axis, C next to
    # them); below, assignment (D) set apart from the chirp parameters (E, F)
    pw, ph = 4.3, 3.4
    y1, y2 = 7.15, 1.55
    ax_a = C.axes_cm(fig, 1.45, y1, pw, ph)
    ax_b = C.axes_cm(fig, 6.45, y1, pw, ph)
    ax_c = C.axes_cm(fig, 12.6, y1, pw, ph)
    ax_d = C.axes_cm(fig, 1.45, y2, pw, ph)
    sq = 3.4
    ax_e = C.axes_cm(fig, 8.75, y2, sq, sq)
    ax_f = C.axes_cm(fig, 13.75, y2, sq, sq)

    # --- detection ---------------------------------------------------------
    real_dir = {"dualhead": "dualhead_r18_sim_v4_r1", "yolo26s": "yolo26s_sim_v4_r1"}
    pr_panel(ax_a, ["yolo26s", "dualhead"], lambda m: C.REAL_EVAL / real_dir[m],
             "real", report)
    pr_panel(ax_b, ["yolo26s", "dualhead"], lambda m: C.PP_EVAL[m] / "sim_test_after",
             "sim", report, ylabel=False)
    ax_a.set_title("Real recordings", loc="left", pad=6)
    ax_b.set_title("Simulated", loc="left", pad=6)
    op = report["real/dualhead"]
    C.annotate(ax_a, r"score $\geq$ 0.5", xy=(op["recall"], op["precision"]),
               xytext=(0.87, 0.905), color="k", rad=0.2, shrink_b=4)
    benchmark_panel(ax_c, report)

    # --- assignment and chirp parameters ------------------------------------
    assignment_panel(ax_d, report)
    sim = pd.read_parquet(C.DECODER_EVAL / "predictions.parquet")
    real = pd.read_csv(C.DECODER_EVAL / "real_agreement.csv")
    real = real[real["detected"].astype(bool)]
    decoder_panel(ax_e, sim, real, "height_hz", 1.0, "Hz", (15, 1000),
                  [20, 50, 100, 200, 500], report)
    decoder_panel(ax_f, sim, real, "fwhm_s", 1e3, "ms", (2, 500),
                  [3, 10, 30, 100, 300], report)
    ax_e.set_xlabel("Reference height [Hz]")
    ax_e.set_ylabel("Estimated height [Hz]")
    ax_f.set_xlabel("Reference FWHM [ms]")
    ax_f.set_ylabel("Estimated FWHM [ms]")
    # key of E and F, below them (no empty corner inside: outliers everywhere)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C.DENSITY_CMAP(0.6), ec="none",
                      label="simulated (vs truth)"),
        Line2D([], [], ls="none", marker="o", ms=2.4, mfc="k", mec="none",
               label="real long chirps (vs trace)"),
    ]
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False,
               fontsize="small", handlelength=1.0, handletextpad=0.4, columnspacing=1.6,
               bbox_to_anchor=((8.75 + 13.75 + sq) / 2 / W, 0.22 / H))

    # --- model key, letters -------------------------------------------------
    handles = [Line2D([], [], color=C.MODEL_COLORS[m], lw=1.5 if m == "dualhead" else 1.0,
                      marker="o", ms=OP_MS - 0.6, mfc=C.MODEL_COLORS[m], mec="k", mew=0.6,
                      label=C.MODEL_LABELS[m])
               for m in ("dualhead", "yolo26s", "human")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, fontsize="small", handlelength=1.6, columnspacing=1.8,
               handletextpad=0.5)
    C.letters_cm(fig, [(ax_a, "A", 1.15, 0.3), (ax_b, "B", 0.35, 0.3),
                       (ax_c, "C", 1.15, 0.3), (ax_d, "D", 1.15, 0.3),
                       (ax_e, "E", 1.4, 0.3), (ax_f, "F", 1.4, 0.3)])

    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
