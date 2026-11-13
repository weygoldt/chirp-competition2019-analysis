"""Figure: performance of chirp detection pipeline v2.

Replaces ``plot_pipeline_performances_manuscript_supplement_supplement.pdf``
(cross-validation curves of the v1 YOLOv8 variants). Every plotted number is
recomputed here from saved per-item or per-threshold data:

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
C   Emitter assignment on the simulated true positives at score >= 0.5
    (after post-processing), by ΔF between the emitter and the closest
    other fish: nearest live track to the predicted emitter EODf (dual-head)
    against the box-only baseline (nearest live track to the box's lower
    edge + 41 Hz, the median offset of the training chirps). Per-item data
    recomputed by ``v2_fig_extract.py assign-baseline``.
D   Blind benchmark on 1,010 simulated items (536 scored true chirps):
    recall of one expert labeller and of the dual-head detector (>= 0.5),
    by chirp SNR (``labels/benchmark/report_v2/items.csv``; unweighted).
E,F Chirp-parameter decoder on 7,678 simulated test chirps of unseen fish
    (calibrated estimate against the noise-free per-EOD-period truth;
    ``params/.../eval/predictions.parquet``) and on 513 real long chirps of
    the competition recordings (estimate against an independent
    frequency-trace reference; ``real_agreement.csv``).

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_performance.py

Runtime: ~15 s. Prints every number it plots.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v2_fig_common as C
from style import letter_subplots, set_light_style

NAME = "plot_v2_performance"
LABELS = {"dualhead": "v2 dual-head", "yolo26s": "v2 YOLO26s", "v1": "v1 (2024)"}
DF_BINS = ["0-10", "10-25", "25-50", "50-100", "100+"]
DF_TICKS = ["$<$10", "10--25", "25--50", "50--100", r"$\geq$100"]
SNR_BINS = ["<-20 dB", "-20 to -10 dB", "-10 to 0 dB", "0-10 dB", ">=10 dB"]
SNR_TICKS = ["$<-$20", "$-$20 to $-$10", "$-$10 to 0", "0 to 10", r"$\geq$10"]


def counts(items: pd.DataFrame) -> dict:
    o = items["outcome"].value_counts()
    tp, fp, fn = int(o.get("tp", 0)), int(o.get("fp", 0)), int(o.get("fn", 0))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp),
            "recall": tp / (tp + fn)}


def pr_panel(ax, models, base, kind, report):
    for m in models:
        d = base(m)
        curve = pd.read_csv(d / "pr_curve.csv")
        curve = curve[curve["level"] == "all"].sort_values("threshold")
        op = counts(pd.read_csv(d / "items.csv"))
        report[f"{kind}/{m}"] = op
        col = C.MODEL_COLORS[m]
        ax.plot(curve["recall"], curve["precision"], color=col, lw=1.1,
                label=LABELS[m], zorder=3 if m == "dualhead" else 2)
        ax.plot(op["recall"], op["precision"], "o", ms=4, mfc=col, mec="k",
                mew=0.5, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")


def assignment_panel(ax, report):
    a = pd.read_csv(C.SNIPPETS / "assign_baseline.csv")
    meta = json.loads((C.SNIPPETS / "assign_baseline.json").read_text())
    allp = {k: float(a[k].mean()) for k in ("head_correct", "edge_correct",
                                            "edge_offset_correct")}
    n_all = int(len(a))
    a = a[a["df_bin"].isin(DF_BINS)]
    twofish = {k: float(a[k].mean()) for k in allp}
    x = np.arange(len(DF_BINS))
    rows = []
    for name, col, off, lab in (
        ("head_correct", C.MODEL_COLORS["dualhead"], -0.12, "emitter EOD$f$"),
        ("edge_offset_correct", C.MODEL_COLORS["edge"], 0.12, "box edge"),
    ):
        g = a.groupby("df_bin")[name].agg(["sum", "size"]).reindex(DF_BINS)
        p = g["sum"] / g["size"]
        lo, hi = C.wilson(g["sum"].to_numpy(), g["size"].to_numpy())
        ax.errorbar(x + off, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)], fmt="o", ms=3.2, color=col,
                    mec="k", mew=0.4, elinewidth=0.8, capsize=0, label=lab)
        rows.append(pd.DataFrame({"method": name, "df_bin": DF_BINS, "k": g["sum"],
                                  "n": g["size"], "p": p, "lo": lo, "hi": hi}))
    n = a.groupby("df_bin").size().reindex(DF_BINS)
    for xi, ni in zip(x, n, strict=True):
        ax.text(xi, 0.215, f"{ni:,}", ha="center", va="bottom", fontsize="x-small",
                color="#555555")
    ax.set_xticks(x, DF_TICKS, rotation=35, ha="right")
    ax.set_xlabel(r"$\Delta f$ to the other fish [Hz]")
    ax.set_ylabel("Correct emitter")
    ax.set_ylim(0.2, 1.03)
    ax.set_xlim(-0.5, len(DF_BINS) - 0.5)
    report["assignment"] = {
        "table": pd.concat(rows).round(4).to_dict(orient="records"),
        "all_tp": allp,
        "n_all_tp": n_all,
        "two_or_more_fish": twofish,
        "n_two_or_more_fish": int(len(a)),
        "offset_hz": meta["offset_hz"],
    }


def benchmark_panel(ax, report):
    b = pd.read_csv(C.BENCHMARK / "items.csv", low_memory=False)
    t = b[(b["truth"] == "chirp") & b["scored"].astype(bool)]
    x = np.arange(len(SNR_BINS))
    out = {}
    for name, col, off, lab in (
        ("human_chirp", C.MODEL_COLORS["human"], -0.12, "expert"),
        ("dualhead_chirp", C.MODEL_COLORS["dualhead"], 0.12, "v2 dual-head"),
    ):
        g = t.groupby("snr_bin")[name].agg(["sum", "size"]).reindex(SNR_BINS)
        p = g["sum"] / g["size"]
        lo, hi = C.wilson(g["sum"].to_numpy(), g["size"].to_numpy())
        ax.errorbar(x + off, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)], fmt="o", ms=3.2, color=col,
                    mec="k", mew=0.4, elinewidth=0.8, capsize=0, label=lab)
        out[name] = {"by_snr": dict(zip(SNR_BINS, np.round(p.to_numpy(), 4).tolist(),
                                        strict=True)),
                     "n": g["size"].astype(int).tolist(),
                     "overall": float(t[name].mean())}
    for name in ("human_emitter_ok", "dualhead_emitter_ok"):
        out[name] = float(t[name].mean())
    n = t.groupby("snr_bin").size().reindex(SNR_BINS)
    for xi, ni in zip(x, n, strict=True):
        ax.text(xi, 0.515, f"{ni}", ha="center", va="bottom", fontsize="x-small",
                color="#555555")
    ax.set_xticks(x, SNR_TICKS, rotation=30, ha="right")
    ax.set_xlabel("Chirp SNR [dB]")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.5, 1.03)
    ax.set_xlim(-0.5, len(SNR_BINS) - 0.5)
    out["n_true_chirps"] = int(len(t))
    out["n_items"] = int(len(b))
    report["benchmark"] = out


def decoder_panel(ax, sim, real, param, scale, unit, lims, report):
    true = sim[f"true_{param}"].to_numpy() * scale
    est = sim[f"cal_{param}"].to_numpy() * scale
    ax.plot(true, est, ".", ms=1.2, mew=0, color=C.MODEL_COLORS["dualhead"],
            alpha=0.25, rasterized=True, label="simulated")
    r_ref = real[f"ref_{param}"].to_numpy() * scale
    r_est = real[param].to_numpy() * scale
    ax.plot(r_ref, r_est, ".", ms=1.6, mew=0, color=C.MODEL_COLORS["human"],
            alpha=0.6, rasterized=True, label="real (trace ref.)")
    ax.plot(lims, lims, color="k", lw=0.6, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    err = np.abs(est - true)
    rerr = err / true
    rr = np.abs(r_est - r_ref)
    report[f"decoder/{param}"] = {
        "n_sim": int(len(true)),
        "median_abs_error": float(np.median(err)),
        "median_rel_error": float(np.median(rerr)),
        "n_real": int(len(r_ref)),
        "real_median_abs_difference": float(np.median(rr)),
        "real_median_rel_difference": float(np.median(rr / r_ref)),
        "unit": unit,
    }


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    report: dict = {}
    fig = plt.figure(figsize=(C.FIG_WIDTH, 12.0 * C.CM))
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.935, bottom=0.14,
                          wspace=0.5, hspace=0.78)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_d, ax_e, ax_f = (fig.add_subplot(gs[1, i]) for i in range(3))

    pr_panel(ax_a, ["v1", "yolo26s", "dualhead"],
             lambda m: C.REAL_EVAL / {"dualhead": "dualhead_r18_sim_v4_r1",
                                      "yolo26s": "yolo26s_sim_v4_r1", "v1": "v1"}[m],
             "real", report)
    pr_panel(ax_b, ["yolo26s", "dualhead"],
             lambda m: C.PP_EVAL[m] / "sim_test_after", "sim", report)
    for ax in (ax_a, ax_b):
        ax.set_xlim(0.65, 1.005)
        ax.set_ylim(0.85, 1.005)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    ax_a.set_title(r"Real, hand-labelled", loc="left", fontsize="small")
    ax_b.set_title(r"Simulated test scenes", loc="left", fontsize="small")
    ax_a.legend(loc="lower left", fontsize="x-small", frameon=False, handlelength=1.2,
                borderaxespad=0.2)

    assignment_panel(ax_c, report)
    ax_c.set_title("Simulated test scenes", loc="left", fontsize="small")
    ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize="x-small",
                frameon=False, handletextpad=0.2, borderaxespad=0.2)

    benchmark_panel(ax_d, report)
    ax_d.set_title("Blind benchmark (simulated)", loc="left", fontsize="small")
    ax_d.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize="x-small",
                frameon=False, handletextpad=0.2, borderaxespad=0.2)

    sim = pd.read_parquet(C.DECODER_EVAL / "predictions.parquet")
    real = pd.read_csv(C.DECODER_EVAL / "real_agreement.csv")
    real = real[real["detected"].astype(bool)]
    decoder_panel(ax_e, sim, real, "height_hz", 1.0, "Hz", (15, 1000), report)
    decoder_panel(ax_f, sim, real, "fwhm_s", 1e3, "ms", (1.5, 500), report)
    ax_e.set_xlabel("True height [Hz]")
    ax_e.set_ylabel("Decoded height [Hz]")
    ax_f.set_xlabel("True FWHM [ms]")
    ax_f.set_ylabel("Decoded FWHM [ms]")
    ax_e.legend(loc="upper left", fontsize="x-small", frameon=False, markerscale=4,
                handletextpad=0.1, borderaxespad=0.2)
    ax_e.set_title("Decoder", loc="left", fontsize="small")

    letter_subplots([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], xoffset=-0.32, yoffset=1.12)
    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
