"""Figure (supplement): the matched comparison of Fig. 8 B as a rate, with every chirp, and its checks.

A   the winner's re-attack rate after the loser's first chirp after the offset (teal) and at matched moments
    without a chirp (gray), as in Fig. 8 B (``stats_reattack_matched.py``): the first chase onset after the
    trigger, smoothed with a Gaussian kernel (SD ``settings.kernel_sigma_s``, reflected at 0), per interval
    still at risk, per minute; bands 2.5-97.5 % of resamples of the competitions; dotted: the matched rate x
    0.5 (what a chirp that halved the rate would give). Before the trigger the rate is 0 by construction (no
    re-attack can precede it), so the axis starts at the trigger.
B, C  as A and Fig. 8 B with every loser chirp between the offset and the next onset as a trigger.
D   matched rate ratios in the 30 s after the trigger (after a chirp / matched): the primary hazard model's HR
    (posterior median, 95 % HDI); the data (first chirps, all chirps; bootstrap intervals); the selection trap
    (open circles: matched intervals required to stay silent until their re-attack); surrogates (median and
    2.5-97.5 % of the replicates): loser chirp trains shuffled among the intervals of a competition (no effect
    possible) and re-attacks simulated with a planted hazard ratio (grey tick) of 1, 0.5 and 2.

    paper/run.sh plot_receiver_matched_checks.py   # chain reattack, after stats_reattack_matched.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

import plot_receiver_before_after as fig8
import reattack_matched as rm
from offset_window import offset_window
from paper_config import FIGURES

NAME = "plot_receiver_matched_checks"
CHIRP_C, CTRL_C, BAND_GREY, BAND_ALPHA, MUTED = fig8.CHIRP_C, fig8.CTRL_C, fig8.BAND_GREY, fig8.BAND_ALPHA, fig8.MUTED
ANNOT_FS = fig8.ANNOT_FS
TRAP_DY = 0.3  # D: the trap's mark sits this far below its row
RATIO_XLIM = (0.3, 3.3)

# Layout [cm]: A-C in one row (the rate and survival panels of the matched design), D and its key below.
W_CM, H_CM = 17.4, 11.9
ROW1_BOTTOM, ROW1_H = 6.9, 3.65
ROW2_BOTTOM, ROW2_H = 1.15, 4.0
COL_X, COL_W = (1.25, 7.0, 12.75), 4.45
D_X, D_W = 3.85, 5.3  # D's row labels to its left
KEY_X = 9.9
LETTER_DY = 0.35


def rate_top(curves: dict, designs=("first_chirp", "all_chirps")) -> float:
    """The rate axes' common top: the highest band edge rounded up to 0.5 /min, at least 3."""
    hi = max(float(np.nanmax(curves[f"{d}_hazard_{g}_hi"])) for d in designs for g in rm.GROUPS)
    return max(3.0, float(np.ceil(2 * hi) / 2))


def rate_panel(ax, curves: dict, design: str, top: float, appease: bool = True) -> None:
    """Kernel-smoothed re-attack rate after the trigger (teal) and matched moments (gray), per minute."""
    t = curves["grid"]
    for g, line_c, band_c in ((rm.CONTROL, CTRL_C, BAND_GREY), (rm.TRIGGER, CHIRP_C, CHIRP_C)):
        k = f"{design}_hazard_{g}"
        ax.fill_between(t, curves[k + "_lo"], curves[k + "_hi"], color=band_c, alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(t, curves[k], color=line_c, lw=1.0, zorder=3 if g == rm.TRIGGER else 2)
    if appease:
        ax.plot(t, fig8.APPEASE_HR * curves[f"{design}_hazard_{rm.CONTROL}"], color=CHIRP_C, lw=0.8,
                ls=fig8.APPEASE_LS, zorder=2.5)
    ax.axvline(0, c="k", lw=0.5, ls="--")
    ax.set_xlim(0, float(t[-1]))
    ax.set_xticks(np.arange(0, float(t[-1]) + 0.1, 10))
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 0.01, 1))
    ax.set_ylabel("Re-attacks/min")


def forest_rows(summary: dict, hr: dict) -> list[dict]:
    """D's rows: the model's HR; the data's matched ratios (first chirps with the trap below, all chirps);
    the surrogates (median and 2.5-97.5 % of the replicates, the design and the trap)."""
    def mark(d, **kw):
        return {"value": d["ratio"], "ci": d["ratio_ci"], **kw}

    trap = {"filled": False, "color": MUTED, "dy": TRAP_DY}
    rows = [{"label": "hazard model (HR)", "marks": [{"value": hr["median"], "ci": hr["hdi"], "color": CHIRP_C}]},
            {"label": "data: first chirps", "marks": [mark(summary["first_chirp"]["w30"]),
                                                      mark(summary["trap"]["w30"], **trap)]},
            {"label": "data: all chirps", "marks": [mark(summary["all_chirps"]["w30"])], "gap_after": True}]
    names = {"shuffled": "shuffled chirps", "sim_hr_1": "simulated HR 1", "sim_hr_0p5": "simulated HR 0.5",
             "sim_hr_2": "simulated HR 2"}
    for key, label in names.items():
        d = summary["surrogates"][key]
        rows.append({"label": label, "planted": d["planted"], "marks": [
            {"value": d["design_w30"]["median"], "ci": d["design_w30"]["q"]},
            {"value": d["trap_w30"]["median"], "ci": d["trap_w30"]["q"], **trap}]})
    for y, r in enumerate(rows):
        r["y"] = y + (0.5 if y >= 3 else 0.0)
    return rows


def forest_panel(ax, rows: list[dict]) -> None:
    """D: per row a dot (estimate or surrogate median) and a whisker (interval); the trap as an open circle
    just below; the planted ratio of a surrogate as a grey tick."""
    for r in rows:
        y = r["y"]
        if r.get("planted") is not None:
            ax.plot([r["planted"]] * 2, [y - 0.4, y + 0.4], color=MUTED, lw=1.2, zorder=1, solid_capstyle="butt")
        for m in r["marks"]:
            yy, c, filled = y + m.get("dy", 0.0), m.get("color", "k"), m.get("filled", True)
            ax.plot(m["ci"], [yy, yy], color=c, lw=0.9, zorder=2, solid_capstyle="butt")
            ax.plot([m["value"]], [yy], ls="", marker="o", ms=3.2 if filled else 2.9, mfc=c if filled else "white",
                    mec="white" if filled else c, mew=0.5 if filled else 0.7, zorder=3)
        if r.get("gap_after"):
            ax.axhline(y + 0.75, color="0.85", lw=0.5, zorder=0)
    ax.axvline(1, color="k", lw=0.5, ls=":", zorder=0.5)
    ax.set_xscale("log")
    ax.set_xlim(*RATIO_XLIM)
    ax.set_xticks([0.5, 1, 2])
    ax.set_xticklabels(["0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_ylim(max(r["y"] for r in rows) + 0.7, -0.7)
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=ANNOT_FS)
    ax.set_xlabel("Re-attack rate ratio, 0--30 s after the chirp")


def forest_key(fig, summary: dict) -> None:
    """D's symbols and what each surrogate is."""
    W, H = fig.get_size_inches() * 2.54
    keys = [Line2D([], [], ls="", marker="o", ms=3.2, mfc="k", mec="white", mew=0.5),
            Line2D([], [], ls="", marker="o", ms=2.9, mfc="white", mec=MUTED, mew=0.7),
            Line2D([], [], ls="", marker="|", ms=6, mew=1.2, color=MUTED)]
    fig.legend(keys, ["matched design: at the matched moment the loser\n"
                      "is silent and the winner has not attacked again;\n"
                      "stops counting when its own loser chirps",
                      "trap: matched intervals must stay silent until\n"
                      "the re-attack (keeps intervals that ended quickly)",
                      "planted hazard ratio of a surrogate"],
               loc="upper left", bbox_to_anchor=(KEY_X / W, (ROW2_BOTTOM + ROW2_H + 0.2) / H), frameon=False,
               fontsize=ANNOT_FS, borderaxespad=0, borderpad=0, handletextpad=0.5, labelspacing=0.6,
               handlelength=1.0)
    n = summary["surrogates"]["shuffled"]["n_replicates"]
    fig.text(KEY_X / W, (ROW2_BOTTOM + 0.05) / H,
             f"Surrogates, {n} each (median, 2.5--97.5\\,\\%):\n"
             "shuffled: each interval gets the loser chirps of\n"
             "another interval of the same competition, at their\n"
             "times since the offset (no effect possible);\n"
             "simulated: re-attacks drawn anew, the chirp\n"
             "multiplying their rate by a known hazard ratio.",
             ha="left", va="bottom", fontsize=ANNOT_FS, color=MUTED, linespacing=1.25)


def draw(curves: dict, summary: dict, hr: dict):
    fig8.set_light_style()
    fig8._rc_tweaks()
    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
    ax_a, ax_b, ax_c = (fig8._axes_cm(fig, x, ROW1_BOTTOM, COL_W, ROW1_H) for x in COL_X)
    ax_d = fig8._axes_cm(fig, D_X, ROW2_BOTTOM, D_W, ROW2_H)
    top = rate_top(curves)
    rate_panel(ax_a, curves, "first_chirp", top)
    ax_a.set_xlabel("Time since first chirp [s]")
    rate_panel(ax_b, curves, "all_chirps", top)
    ax_b.set_xlabel("Time since any chirp [s]")
    fig8.survival_panel(ax_c, curves, "all_chirps")
    ax_c.set_xlabel("Time since any chirp [s]")
    W, H = fig.get_size_inches() * 2.54
    fig.legend(fig8.matched_handles(), ["after the loser's chirp (A: first chirp; B, C: any chirp)",
                                        "matched moments, no chirp yet",
                                        f"if a chirp halved the re-attack rate (HR {fig8.APPEASE_HR:g})"],
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.08 / H), frameon=False, fontsize=ANNOT_FS,
               borderaxespad=0, borderpad=0, handlelength=1.5, handletextpad=0.5, columnspacing=1.5)
    forest_panel(ax_d, forest_rows(summary, hr))
    forest_key(fig, summary)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        sns.despine(ax=ax, offset=4, trim=False)
    ax_d.spines["left"].set_visible(False)
    fig8._letters_cm(fig, [(ax_a, "A", 0.95, LETTER_DY), (ax_b, "B", 0.95, LETTER_DY),
                           (ax_c, "C", 0.95, LETTER_DY), (ax_d, "D", D_X - 0.3, LETTER_DY)])
    return fig, {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}


def main() -> None:
    window = offset_window()
    curves, summary = fig8.load_matched()
    if summary["window_s"] != list(window):
        raise ValueError(f"reattack_matched was computed with the window {summary['window_s']}, not {window}")
    fig, axes = draw(curves, summary, fig8.model_hr())
    print(fig8.axes_extents(fig, list(axes.items())))
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{NAME}.{ext}" for ext in ("pdf", "png")]
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
