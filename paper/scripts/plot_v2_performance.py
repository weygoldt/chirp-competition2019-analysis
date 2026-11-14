"""Figure: performance of chirp detection pipeline v2.

Replaces ``plot_pipeline_performances_manuscript_supplement_supplement.pdf``
(cross-validation curves of the v1 YOLOv8 variants). Grouped by pipeline
stage (the stages of the pipeline figure), one colour per model throughout;
every plotted number is recomputed here from saved per-item or
per-threshold data:

Detection
A   Real recordings, hand labels (365 chirps in 6 labelled regions of 4
    held-out recordings): precision against recall over the score threshold
    for the dual-head detector (production), YOLO26s (box-only reference)
    and v1. Curves: ``pr_curve.csv`` (level "all") of
    ``eval/real_labels_365/<model>``; operating points (score >= 0.5):
    counted from that evaluation's ``items.csv``.
B   Simulated held-out test scenes (unseen fish on unseen background
    recordings; 9,401 visible chirps), after post-processing: the same for
    the dual-head and YOLO26s (``runs/<model>/eval_postprocess/
    sim_test_after``).
C   Blind benchmark on 1,010 simulated items (536 scored true chirps):
    recall of one expert labeller and of the dual-head detector (>= 0.5),
    by chirp SNR (``labels/benchmark/report_v2/items.csv``; unweighted,
    Wilson 95 % intervals).
Assignment
D   Emitter assignment on the simulated true positives at score >= 0.5
    (after post-processing), by ΔF between the emitter and the closest
    other fish: nearest live track to the predicted emitter EODf (dual-head)
    against the box-only rule (nearest live track to the box's lower edge +
    41 Hz, the median offset of the training chirps). Per-item data
    recomputed by ``v2_fig_extract.py assign-baseline``.
Chirp parameters
E,F Chirp-parameter decoder on 7,678 simulated test chirps of unseen fish
    (calibrated estimate against the noise-free per-EOD-period truth;
    ``params/.../eval/predictions.parquet``; density of points) and on 513
    real long chirps of the competition recordings (estimate against an
    independent frequency-trace reference; ``real_agreement.csv``; black).

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_performance.py

Runtime: ~15 s. Prints every number it plots.
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
from style import letter_subplots, set_light_style

NAME = "plot_v2_performance"
H = 12.75  # figure height [cm]
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


def letters(fig, axes_and_offsets):
    """letter_subplots with offsets given in cm from each axes' upper left corner."""
    W, Hh = fig.get_size_inches() / C.CM
    axs, xo, yo, lets = [], [], [], []
    for ax, let, dx, dy in axes_and_offsets:
        bb = ax.get_position()
        axs.append(ax)
        lets.append(let)
        xo.append(-dx / (bb.width * W))
        yo.append(1 + dy / (bb.height * Hh))
    letter_subplots(axs, letters=lets, xoffset=xo, yoffset=yo)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def pr_panel(ax, models, base, kind, report):
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
                mew=0.6, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.65, 1.0)
    ax.set_ylim(0.85, 1.0)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    sns.despine(ax=ax, offset=3, trim=True)


def proportion_points(ax, x, k, n, color, *, filled=True, dx=0.0, logx=False, zorder=4):
    p = k / n
    lo, hi = C.wilson(k, n)
    xx = np.asarray(x) * (10**dx if logx else 1) + (0 if logx else dx)
    ax.plot(xx, p, color=color, lw=0.8, zorder=zorder - 1)
    ax.errorbar(xx, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                fmt="o", ms=PT_MS, mfc=color if filled else "white", mec=color if not filled else "k",
                mew=0.6 if filled else 0.9, ecolor=color, elinewidth=0.9, capsize=0,
                zorder=zorder)
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
    sns.despine(ax=ax, offset=3, trim=False)
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
    sns.despine(ax=ax, offset=3, trim=False)
    handles = [
        Line2D([], [], color=C.MODEL_COLORS["dualhead"], lw=0.8, marker="o", ms=PT_MS,
               mfc=C.MODEL_COLORS["dualhead"], mec="k", mew=0.6,
               label=r"predicted emitter EOD$f$"),
        Line2D([], [], color=C.MODEL_COLORS["edge"], lw=0.8, marker="o", ms=PT_MS,
               mfc="white", mec=C.MODEL_COLORS["edge"], mew=0.9,
               label=f"box edge + {meta['offset_hz']:.0f} Hz"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, 0.12),
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
DENSITY_CMAP = sns.light_palette(C.MODEL_COLORS["dualhead"], as_cmap=True)


def decoder_panel(ax, sim, real, param, scale, unit, lims, ticks, report):
    true = sim[f"true_{param}"].to_numpy() * scale
    est = sim[f"cal_{param}"].to_numpy() * scale
    r_ref = real[f"ref_{param}"].to_numpy() * scale
    r_est = real[param].to_numpy() * scale
    edges = np.geomspace(*lims, 56)
    h, _, _ = np.histogram2d(true, est, bins=[edges, edges])
    h = np.where(h > 0, h, np.nan)
    ax.pcolormesh(edges, edges, h.T, cmap=DENSITY_CMAP, norm=LogNorm(vmin=1, vmax=np.nanmax(h)),
                  rasterized=True, zorder=1, linewidth=0)
    ax.plot(r_ref, r_est, "o", ms=1.6, mfc="k", mec="none", alpha=0.55, zorder=3,
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
    sns.despine(ax=ax, offset=3, trim=False)
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
    ax.text(0.98, 0.03,
            "median deviation\n"
            rf"simulated vs truth {100 * out['median_rel_error']:.1f}\,\%" "\n"
            rf"real vs trace {100 * out['real_median_rel_difference']:.1f}\,\%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize="small",
            linespacing=1.15)
    return out


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {}
    fig = plt.figure(figsize=(C.FIG_WIDTH, H * C.CM))

    # layout [cm]
    xs = [1.45, 7.25, 13.05]
    pw, ph = 3.95, 3.45
    y1, y2 = 7.2, 1.15

    ax_a = C.axes_cm(fig, xs[0], y1, pw, ph)
    ax_b = C.axes_cm(fig, xs[1], y1, pw, ph)
    ax_c = C.axes_cm(fig, xs[2], y1, pw, ph)
    ax_d = C.axes_cm(fig, xs[0], y2, pw, ph)
    ax_e = C.axes_cm(fig, xs[1] + 0.25, y2, ph, ph)
    ax_f = C.axes_cm(fig, xs[2] + 0.25, y2, ph, ph)

    # --- detection ---------------------------------------------------------
    real_dir = {"dualhead": "dualhead_r18_sim_v4_r1", "yolo26s": "yolo26s_sim_v4_r1",
                "v1": "v1"}
    pr_panel(ax_a, ["v1", "yolo26s", "dualhead"], lambda m: C.REAL_EVAL / real_dir[m],
             "real", report)
    pr_panel(ax_b, ["yolo26s", "dualhead"], lambda m: C.PP_EVAL[m] / "sim_test_after",
             "sim", report)
    n_real = report["real/dualhead"]["tp"] + report["real/dualhead"]["fn"]
    n_sim = report["sim/dualhead"]["tp"] + report["sim/dualhead"]["fn"]
    ax_a.set_title(f"Real, {n_real:,} hand labels", loc="left")
    ax_b.set_title(f"Simulated, {n_sim:,} chirps", loc="left")
    op = report["real/dualhead"]
    ax_a.annotate(r"score $\geq$ 0.5", xy=(op["recall"] - 0.005, op["precision"] - 0.002),
                  xytext=(0.86, 0.905), fontsize="small", ha="center", va="top",
                  arrowprops=dict(arrowstyle="-|>", color="k", lw=0.6, mutation_scale=5,
                                  shrinkA=1, shrinkB=2.5,
                                  connectionstyle="arc3,rad=0.25"))
    benchmark_panel(ax_c, report)
    ax_c.set_title("Expert vs detector (blind)", loc="left")

    # --- assignment and parameters ------------------------------------------
    assignment_panel(ax_d, report)
    ax_d.set_title(f"Simulated, {report['assignment']['n_two_or_more_fish']:,} chirps",
                   loc="left")
    sim = pd.read_parquet(C.DECODER_EVAL / "predictions.parquet")
    real = pd.read_csv(C.DECODER_EVAL / "real_agreement.csv")
    real = real[real["detected"].astype(bool)]
    decoder_panel(ax_e, sim, real, "height_hz", 1.0, "Hz", (15, 1000),
                  [20, 50, 100, 200, 500], report)
    decoder_panel(ax_f, sim, real, "fwhm_s", 1e3, "ms", (2, 500),
                  [3, 10, 30, 100, 300], report)
    ax_e.set_xlabel("Reference height [Hz]")
    ax_e.set_ylabel("Decoded height [Hz]")
    ax_f.set_xlabel("Reference FWHM [ms]")
    ax_f.set_ylabel("Decoded FWHM [ms]")
    ax_e.set_title("Height", loc="left")
    ax_f.set_title("FWHM", loc="left")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=DENSITY_CMAP(0.7), ec="none",
                      label="simulated"),
        Line2D([], [], ls="none", marker="o", ms=2.2, mfc="k", mec="none",
               label="real long chirps"),
    ]
    ax_e.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.02),
                frameon=False, borderaxespad=0.1, handlelength=1.0, handletextpad=0.4)

    # --- model key, stage brackets, letters ---------------------------------
    handles = [Line2D([], [], color=C.MODEL_COLORS[m], lw=1.5 if m == "dualhead" else 1.0,
                      marker="o", ms=OP_MS - 0.6, mfc=C.MODEL_COLORS[m], mec="k", mew=0.6,
                      label=C.MODEL_LABELS[m])
               for m in ("dualhead", "yolo26s", "v1", "human")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, fontsize="small", handlelength=1.6, columnspacing=1.6,
               handletextpad=0.5)
    W = 17.4
    C.bracket(fig, xs[0] - 1.1, xs[2] + pw, y1 + ph + 0.72, "Detection")
    C.bracket(fig, xs[0] - 1.1, xs[0] + pw, y2 + ph + 0.72, "Assignment")
    C.bracket(fig, xs[1] - 0.9, xs[2] + 0.25 + ph, y2 + ph + 0.72, "Chirp parameters")
    letters(fig, [(ax_a, "A", 1.15, 0.3), (ax_b, "B", 1.15, 0.3), (ax_c, "C", 1.15, 0.3),
                  (ax_d, "D", 1.15, 0.3), (ax_e, "E", 1.4, 0.3), (ax_f, "F", 1.4, 0.3)])

    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
