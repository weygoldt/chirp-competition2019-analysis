"""DRAFT (not in the manuscript): candidate panels for a more didactic Fig. 8, for the author to choose from.

The receiver test is the re-attack hazard (``reattack_hazard.py``, ``stats_reattack_hazard.py``): after each chase
offset, the hazard of the winner's next chase onset, with the loser's chirping as the time-varying covariate
``chirped`` (0 -> 1 at the loser's first chirp at or after max(offset - 5 s, onset)); HR 1.02 [0.86, 1.20].
These panels show the data and what the model compares, closer to the data than rates per piece.

Figure 1 (``plot_receiver_didactic_draft``), main-text candidates:

A   the data and how the model splits time: a random sample of inter-chase intervals (offset at 0), sorted by
    length; the chase that just ended (grey) with its last 5 s (light teal: a loser chirp there switches
    ``chirped`` on at the offset), the loser's chirps (ticks), the time at risk before the first counted chirp
    (black) and after it (teal), and the next chase onset (the re-attack, dot). Rows longer than the axis end
    in an arrow.
B   chirp-triggered re-attack rate, as Fig. 6: triggers are the loser's first chirps after the offset in
    intervals that were not yet chirped (the moments the covariate switches after the offset); controls are
    the same moments (same time since the offset) in other intervals of the same competition that were still
    at risk and not yet chirped (``reattack_matched``). Rate: the first re-attack after the trigger, smoothed
    with Fig. 6's kernel, per interval still at risk (controls leave the risk set at their own first chirp).
    Lines: all data; bands: 2.5-97.5 % of 1000 resamples of the competitions (bootstrap; ``--band leave_out``
    for Fig. 6's leave-10 %-out resamples). Dotted: the matched rate x 0.5, what an appeasement effect that
    halved the attack rate would look like. t < 0 is not drawn: no re-attack can precede the trigger.
C   the same triggers and controls as Kaplan-Meier curves: fraction of intervals not yet re-attacked.
D   previous inter-chase interval with and without a loser chirp in the offset window (Fig. 8 A as it is,
    ``plot_receiver_before_after.interval_panel``): chirps come with dense chasing.

Figure 2 (``plot_receiver_didactic_draft_checks``), supplement candidates and checks:

E, F  as B, C with every loser chirp between the offset and the next onset as a trigger.
G     rate ratios (after a chirp / matched) in 0-30 s: the data (first chirps, all chirps), the model's HR,
      the selection trap (controls that never chirp), and surrogates: chirps shuffled among the intervals of a
      competition (no effect), re-attacks simulated with a known hazard ratio (1, 0.5, 2).

Every number to ``$PAPER_FIGURE_DIR/plot_receiver_didactic_draft.json`` (not ``write_stats``: a draft).

    PAPER_DETECTIONS=rtdetr_gapfill_t080 PAPER_OUT=<scratch root> python plot_receiver_didactic_draft.py \
        --primary-out <scratch root>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_receiver_before_after as fig8
import reattack_hazard as rh
import reattack_matched as rm
from offset_window import offset_window
from paper_config import DATA_ROOT, FIGURES, PRIMARY_OUT, PROCESSED, SEED, figure
from style import loser_color, set_light_style

NAME = "plot_receiver_didactic_draft"
HAZARD_DIR = PROCESSED / "reattack_hazard"

CHIRP_C = loser_color
NONE_C = "k"
CTRL_C = "0.35"  # matched moments: line; band in BAND_GREY (Fig. 6's null band)
BAND_GREY = "grey"
BAND_ALPHA = 0.35
MUTED = "#6e6e6e"
CHASE_C = "0.85"
TAIL_C = tuple(0.32 * np.asarray(to_rgb(loser_color)) + 0.68)  # the chase's last 5 s: teal blended with white
LW_WAIT, LW_CHIRP = 0.85, 1.9  # A: time at risk before / after the first chirp
ANNOT_FS = "small"  # 6.7 pt at the 8-pt base
APPEASE_HR = 0.5  # the dotted reference: a chirp that halves the attack rate
# B, E: Gaussian kernel SD (s). Fig. 6 uses 0.3 s, but it smooths thousands of chirps; there are ~280 re-attacks
# in the 30 s after the triggers, and at 0.3 s the trigger curve is mostly noise (band ~1.6 /min wide on a rate
# of ~1.5 /min). ``--sigma 0.3`` draws Fig. 6's kernel.
SIGMA_S = 1.0
WINDOWS_S = (1.0, 10.0, 30.0)  # rate ratios in [0, T] after the trigger (1 s: the early excess, see the report)

# raster (A)
N_ROWS = 44
RASTER_XLIM = (-10.0, 60.0)
LEFT_EDGE_S = -5.0  # the window's left edge: chirps from offset - 5 s on count (checked against offset_window)

# Figure 1 layout [cm]
W_CM = 17.4
F1_H = 13.6
ROW2_BOTTOM, ROW2_H = 1.15, 3.55
ROW1_BOTTOM, ROW1_H = 6.35, 6.6
COL_X = (1.25, 6.95, 12.75)  # left edges of B, C, D
COL_W = 4.35
A_X, A_W = 1.25, 9.6
KEY_X = 11.25

# Figure 2 layout [cm]
F2_H = 6.5
F2_BOTTOM, F2_AX_H = 1.15, 4.3
E_X, F_X, G_X = 1.25, 6.65, 13.3
EF_W, G_W = 4.1, 3.9
G_LABEL_W = 2.35  # G's row labels sit left of its axes; its letter left of them


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def check_against_saved(iv: pd.DataFrame, saved: pd.DataFrame, atol: float = 1e-6) -> None:
    """The intervals equal those of the hazard fits (``reattack_hazard/intervals.csv``)."""
    if len(iv) != len(saved) or not (iv[rh.COMP].astype(str).to_numpy() == saved[rh.COMP].astype(str).to_numpy()).all():
        raise ValueError("intervals differ from the hazard fits' intervals.csv (rows or recordings)")
    for col in ("onset", "offset", "end", "length", "first_chirp"):
        a, b = iv[col].to_numpy(float), saved[col].to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)) or np.nanmax(np.abs(a - b)) > atol:
            raise ValueError(f"{col} differs from the hazard fits' intervals.csv")
    if not (iv["event"].to_numpy(int) == saved["event"].to_numpy(int)).all():
        raise ValueError("event differs from the hazard fits' intervals.csv")


def load(primary_out: Path, all_intervals: bool):
    window = offset_window(primary_out)
    if window[0] != LEFT_EDGE_S:
        raise ValueError(f"the offset window's left edge is {window[0]} s, the raster draws {LEFT_EDGE_S} s")
    comps = rh.load_competitions(DATA_ROOT)
    iv = rh.intervals_table(comps, window)
    check_against_saved(iv, pd.read_csv(HAZARD_DIR / "intervals.csv"))
    rh.check_against_obs(iv, pd.read_csv(fig8.OBS_DATA))
    chirps = {c.recording: c.loser_chirps for c in comps}
    use = iv if all_intervals else iv[iv["prev_interval"].notna()]  # the primary model's intervals
    return window, use, chirps


def matched(iv: pd.DataFrame, triggers: pd.DataFrame, n_boot: int, band: str, seed: int,
            n_controls: int | None, sigma: float = SIGMA_S, rule: str = "unchirped_at_s") -> dict:
    f = rm.matched_followups(iv, triggers, n_controls=n_controls, seed=seed, rule=rule)
    sm = rm.summarise(f, competitions=sorted(iv[rh.COMP].unique()), sigma=sigma, windows=WINDOWS_S)
    at = np.array([10.0, 30.0])
    res = {"followups": f, "summary": sm, "hazard": rm.hazard_per_min(sm), "survival": rm.survival(sm),
           "survival_at": rm.survival(sm, at=at), "windows": rm.window_rates(sm)}
    W = rm.resample_weights(len(sm.competitions), n_boot, seed=seed, mode=band)
    res["boot"] = rm.resampled(sm, W)
    res["boot_at"] = rm.resampled(sm, W, survival_at=at)["survival"]
    return res


def band(a: np.ndarray, q=(2.5, 97.5)) -> np.ndarray:
    return np.nanpercentile(a, q, axis=0)


def ratio_and_ci(res: dict, T: float) -> dict:
    """The matched rate ratio in [0, T] and its resampling interval."""
    k = res["summary"].windows.index(T)
    return {"value": res["windows"][f"{T:g}"]["ratio"], "ci": band(res["boot"]["ratio"][:, k]).tolist()}


def matched_numbers(res: dict) -> dict:
    f = res["followups"]
    trig, ctrl = f[f["group"] == rm.TRIGGER], f[f["group"] == rm.CONTROL]
    out = {"n_triggers": int(len(trig)), "n_triggers_without_controls": int(f.attrs["n_without_controls"]),
           "n_competitions": int(trig[rh.COMP].nunique()), "n_control_moments": int(len(ctrl)),
           "n_control_intervals": int(ctrl["interval"].nunique()),
           "controls_per_trigger_median": float(ctrl.groupby("set").size().median()),
           "trigger_s_median": float(trig["s"].median()),
           "trigger_s_quartiles": [float(x) for x in trig["s"].quantile([0.25, 0.75])]}
    for T in res["summary"].windows:
        w = res["windows"][f"{T:g}"]
        out[f"window_0_{T:g}s"] = {"rate_per_min": w["rate_per_min"], "events": w["events"],
                                   "exposure_s": w["exposure_s"], "ratio": w["ratio"],
                                   "ratio_ci": ratio_and_ci(res, T)["ci"]}
    for k, T in enumerate((10, 30)):
        out[f"survival_{T}s"] = {g: {"value": float(res["survival_at"][g][k]),
                                     "ci": band(res["boot_at"][g][:, k]).tolist()} for g in rm.GROUPS}
    return out


def surrogates(iv: pd.DataFrame, chirps: dict, n: int, seed: int, left: float) -> pd.DataFrame:
    """Matched ratios (both control rules) and the person-period Mantel-Haenszel ratio of every surrogate."""
    rng = np.random.default_rng(seed)
    scen = {"shuffled": (None, lambda: rm.shuffle_first_chirps(iv, chirps, rng, left=left))}
    for hr in (1.0, 0.5, 2.0):
        scen[f"sim_hr_{hr:g}"] = (hr, lambda hr=hr: rm.simulate_reattacks(iv, hr, rng))
    rows = []
    tiny = np.array([0.0, 1.0])  # window ratios only: no curves needed
    for name, (hr, make) in scen.items():
        for rep in range(n):
            x = make()
            trg = rm.first_chirp_triggers(x)
            row = {"scenario": name, "planted": 1.0 if hr is None else hr, "rep": rep, "n_triggers": len(trg)}
            for rule in rm.CONTROL_RULES:
                w = rm.window_rates(rm.summarise(rm.matched_followups(x, trg, rule=rule), grid=tiny))
                row[f"{rule}_10"], row[f"{rule}_30"] = w["10"]["ratio"], w["30"]["ratio"]
            row["mh"] = mh_ratio(x, chirps)
            rows.append(row)
    return pd.DataFrame(rows)


def mh_ratio(iv: pd.DataFrame, chirps: dict) -> float:
    """Mantel-Haenszel rate ratio of ``chirped`` over the person-period table (competitions x pieces): the
    frequentist analogue of the unadjusted hazard model (Fig. 8 C's pooled ratio)."""
    pp = rh.person_period(iv, chirps, covariates=("chirped",))
    c = rh.competition_rate_ratios(rh.cell_table(pp, "chirped"))
    return float(c["num"].sum() / c["den"].sum())


def surrogate_numbers(sur: pd.DataFrame) -> dict:
    out = {}
    for name, g in sur.groupby("scenario", sort=False):
        d = {"planted": float(g["planted"].iloc[0]), "n": int(len(g)),
             "n_triggers_median": float(g["n_triggers"].median())}
        for col in [c for c in g.columns if c.endswith(("_10", "_30")) or c == "mh"]:
            v = g[col].to_numpy(float)
            d[col] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                      "q": np.percentile(v, [2.5, 97.5]).tolist()}
        out[name] = d
    return out


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def raster_rows(iv: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A random sample of ``n`` intervals (seeded), sorted by length (shortest on top)."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(iv), size=min(n, len(iv)), replace=False)
    return iv.iloc[pick].sort_values("length", ascending=True)


def raster_panel(ax, rows: pd.DataFrame, chirps: dict) -> dict:
    """Panel A. Row y = 0 at the top."""
    x0, x1 = RASTER_XLIM
    counts = {"rows": len(rows), "clipped": 0, "tail_chirped": 0, "chirp_after_offset": 0, "no_chirp": 0,
              "censored": 0}
    lw_wait, lw_chirp, lw_chase = LW_WAIT, LW_CHIRP, 3.2
    for y, (_, r) in enumerate(rows.iterrows()):
        off, on, L = float(r["offset"]), float(r["onset"]) - float(r["offset"]), float(r["length"])
        tail0 = max(LEFT_EDGE_S, on)
        # the chase that ended at 0 (grey) and its counting tail (light teal)
        if on < tail0:
            ax.plot([max(on, x0), tail0], [y, y], color=CHASE_C, lw=lw_chase, solid_capstyle="butt", zorder=1)
        ax.plot([tail0, 0], [y, y], color=TAIL_C, lw=lw_chase, solid_capstyle="butt", zorder=1)
        # time at risk: black until the switch, teal after
        sw = rm.switch_times(rows.loc[[r.name]])[0]
        end = min(L, x1)
        if np.isfinite(sw):
            counts["tail_chirped" if sw == 0 else "chirp_after_offset"] += 1
        else:
            counts["no_chirp"] += 1
        a = min(sw, end)
        if a > 0:
            ax.plot([0, a], [y, y], color=NONE_C, lw=lw_wait, solid_capstyle="butt", zorder=2)
        if a < end:
            ax.plot([a, end], [y, y], color=CHIRP_C, lw=lw_chirp, solid_capstyle="butt", zorder=2)
        # loser chirps up to the next onset; grey before the counting range
        c = np.asarray(chirps[r[rh.COMP]], float) - off
        c = c[(c >= x0) & (c < min(L, x1))]
        counted = c >= tail0
        ax.plot(c[counted], np.full(counted.sum(), y), ls="", marker="|", ms=4.2, mew=0.6, color="k", zorder=3)
        ax.plot(c[~counted], np.full((~counted).sum(), y), ls="", marker="|", ms=4.2, mew=0.6, color="0.6",
                zorder=3)
        if np.isfinite(r["first_chirp"]) and float(r["first_chirp"]) - off < x1:
            ax.plot([float(r["first_chirp"]) - off], [y], ls="", marker="o", ms=3.4, mfc=CHIRP_C, mec="white",
                    mew=0.5, zorder=4)
        # the end of the interval
        if L >= x1:
            counts["clipped"] += 1
            ax.plot([x1], [y], ls="", marker=">", ms=2.8, mew=0, color=CHIRP_C if sw < x1 else NONE_C,
                    zorder=4, clip_on=False)
        elif int(r["event"]) == 1:
            ax.plot([L], [y], ls="", marker="o", ms=3.0, mfc="k", mec="white", mew=0.4, zorder=4)
        else:
            counts["censored"] += 1
            ax.plot([L], [y], ls="", marker="o", ms=3.0, mfc="white", mec="k", mew=0.6, zorder=4)
    n = len(rows)
    ax.axvline(0, color="k", lw=0.5, ls="--", zorder=0.5)
    ax.axvline(LEFT_EDGE_S, color=MUTED, lw=0.5, ls=":", zorder=0.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xticks([LEFT_EDGE_S, -10], minor=True)
    ax.set_xlabel("Time since chase offset [s]")
    ax.set_ylabel(f"{n} inter-chase intervals (random sample), by length")
    ax.text(0, -1.3, "offset", ha="center", va="bottom", fontsize=ANNOT_FS)
    ax.text(LEFT_EDGE_S, -1.3, r"$-$5 s", ha="center", va="bottom", fontsize=ANNOT_FS, color=MUTED)
    return counts


def raster_key(fig, x_cm: float, top_cm: float) -> None:
    """The symbols of A, with what the hazard model makes of them."""
    W, H = fig.get_size_inches() * 2.54
    handles = [
        Patch(color=CHASE_C, lw=0),
        Patch(color=TAIL_C, lw=0),
        Line2D([], [], ls="", marker="|", ms=4.2, mew=0.6, color="k"),
        Line2D([], [], ls="", marker="o", ms=3.4, mfc=CHIRP_C, mec="white", mew=0.5),
        Line2D([], [], color=NONE_C, lw=LW_WAIT),
        Line2D([], [], color=CHIRP_C, lw=LW_CHIRP),
        Line2D([], [], ls="", marker="o", ms=3.0, mfc="k", mec="white", mew=0.4),
        Line2D([], [], ls="", marker=">", ms=2.8, mew=0, color=NONE_C),
    ]
    labels = [
        "chase",
        "its last 5 s: a chirp here\ncounts from the offset",
        "loser chirp (grey: earlier\nin the chase, not counted)",
        "first chirp that counts; after\nthe offset a trigger in B, C",
        r"waiting, loser not yet chirped" + "\n" + r"(\texttt{chirped} = 0)",
        r"waiting after the first chirp" + "\n" + r"(\texttt{chirped} = 1)",
        "next chase onset\n(the winner re-attacks)",
        "continues beyond 60 s",
    ]
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x_cm / W, top_cm / H), frameon=False,
                     fontsize=ANNOT_FS, handlelength=1.6, handletextpad=0.6, labelspacing=0.7,
                     borderaxespad=0, borderpad=0)
    fig.text(x_cm / W, (top_cm - 5.45) / H,
             "The hazard model: re-attacks per\nminute of teal vs. per minute of\nblack waiting time, "
             "compared at\nthe same time since the offset\nand within competitions:\nHR = teal rate / black rate.",
             ha="left", va="top", fontsize=ANNOT_FS, color=MUTED, linespacing=1.25)
    return leg


def rate_panel(ax, res: dict, n_label: bool = True, appease: bool = True) -> None:
    """B (and E): kernel-smoothed re-attack rate after the trigger vs matched moments."""
    sm = res["summary"]
    t = sm.grid
    for g, line_c, band_c in ((rm.CONTROL, CTRL_C, BAND_GREY), (rm.TRIGGER, CHIRP_C, CHIRP_C)):
        lo, hi = band(res["boot"]["hazard"][g])
        ax.fill_between(t, lo, hi, color=band_c, alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(t, res["hazard"][g], color=line_c, lw=1.0, zorder=3 if g == rm.TRIGGER else 2)
    if appease:
        ax.plot(t, APPEASE_HR * res["hazard"][rm.CONTROL], color=CHIRP_C, lw=0.8, ls=(0, (1.2, 1.2)), zorder=2.5)
    ax.axvline(0, c="k", lw=0.5, ls="--")
    ax.set_xlim(0, rm.HORIZON_S)
    ax.set_xticks([0, 10, 20, 30])
    top = max(3.0, float(np.ceil(2 * np.nanmax(band(res["boot"]["hazard"][rm.TRIGGER])[1])) / 2))
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 0.01, 1))
    ax.set_xlabel("Time since chirp [s]")
    ax.set_ylabel("Re-attacks/min")


def survival_panel(ax, res: dict, appease: bool = True) -> None:
    """C (and F): Kaplan-Meier fraction not yet re-attacked after the trigger vs matched moments."""
    t = res["summary"].grid
    for g, line_c, band_c in ((rm.CONTROL, CTRL_C, BAND_GREY), (rm.TRIGGER, CHIRP_C, CHIRP_C)):
        lo, hi = band(res["boot"]["survival"][g])
        ax.fill_between(t, lo, hi, color=band_c, alpha=BAND_ALPHA, lw=0, zorder=1, step="post")
        ax.step(t, res["survival"][g], where="post", color=line_c, lw=1.0, zorder=3 if g == rm.TRIGGER else 2)
    if appease:
        ax.plot(t, res["survival"][rm.CONTROL] ** APPEASE_HR, color=CHIRP_C, lw=0.8, ls=(0, (1.2, 1.2)),
                zorder=2.5)
    ax.axvline(0, c="k", lw=0.5, ls="--")
    ax.set_xlim(0, rm.HORIZON_S)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Time since chirp [s]")
    ax.set_ylabel("Not yet re-attacked")


def label_curves(ax, res: dict, trigger_label: str) -> None:
    """Legend in the lower left of a survival panel (free there)."""
    nums = matched_numbers(res)
    handles = [(Patch(color=CHIRP_C, alpha=BAND_ALPHA, lw=0), Line2D([], [], color=CHIRP_C, lw=1.0)),
               (Patch(color=BAND_GREY, alpha=BAND_ALPHA, lw=0), Line2D([], [], color=CTRL_C, lw=1.0)),
               Line2D([], [], color=CHIRP_C, lw=0.8, ls=(0, (1.2, 1.2)))]
    labels = [f"{trigger_label} ($n$ = {nums['n_triggers']})",
              f"matched, no chirp yet ($n$ = {nums['n_control_moments']})",
              f"if a chirp halved the rate (HR {APPEASE_HR:g})"]
    ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=ANNOT_FS, borderaxespad=0.2,
              handlelength=1.5, handletextpad=0.5, labelspacing=0.35)


def forest_panel(ax, rows: list[dict]) -> None:
    """G: one row per data set; per mark a dot (estimate or surrogate median) and a whisker (interval), the
    trap as an open circle just below; the planted ratio of a surrogate as a grey tick."""
    gap_after = {r["y"] for r in rows if r.get("gap_after")}
    for r in rows:
        y = r["y"]
        if r.get("planted") is not None:
            ax.plot([r["planted"]] * 2, [y - 0.4, y + 0.4], color=MUTED, lw=1.2, zorder=1, solid_capstyle="butt")
        for m in r["marks"]:
            yy = y + m.get("dy", 0.0)
            c = m.get("color", "k")
            ax.plot(m["ci"], [yy, yy], color=c, lw=0.9, zorder=2, solid_capstyle="butt")
            filled = m.get("filled", True)
            ax.plot([m["value"]], [yy], ls="", marker="o", ms=3.2 if filled else 2.9, mfc=c if filled else "white",
                    mec="white" if filled else c, mew=0.5 if filled else 0.7, zorder=3)
    for y in gap_after:
        ax.axhline(y + 0.75, color="0.85", lw=0.5, zorder=0)
    ax.axvline(1, color="k", lw=0.5, ls=":", zorder=0.5)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3.3)
    ax.set_xticks([0.5, 1, 2])
    ax.set_xticklabels(["0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_ylim(max(r["y"] for r in rows) + 0.7, -0.7)
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=ANNOT_FS)
    ax.set_xlabel("Re-attack rate ratio, 0--30 s")


def draw_main(rows, chirps, iv, res, ev, w_ev):
    set_light_style()
    fig8._rc_tweaks()
    fig = plt.figure(figsize=(W_CM / 2.54, F1_H / 2.54))
    ax_a = fig8._axes_cm(fig, A_X, ROW1_BOTTOM, A_W, ROW1_H)
    ax_b = fig8._axes_cm(fig, COL_X[0], ROW2_BOTTOM, COL_W, ROW2_H)
    ax_c = fig8._axes_cm(fig, COL_X[1], ROW2_BOTTOM, COL_W, ROW2_H)
    ax_d = fig8._axes_cm(fig, COL_X[2], ROW2_BOTTOM, COL_W, ROW2_H)
    counts = raster_panel(ax_a, rows, chirps)
    raster_key(fig, KEY_X, ROW1_BOTTOM + ROW1_H + 0.1)
    rate_panel(ax_b, res)
    survival_panel(ax_c, res)
    label_curves(ax_c, res, "after the first chirp")
    fig8.interval_panel(ax_d, ev, w_ev)
    n = fig8.group_counts(ev[w_ev > 0])
    ax_d.legend([Line2D([], [], color=CHIRP_C, lw=fig8.CURVE_LW), Line2D([], [], color=NONE_C, lw=fig8.CURVE_LW)],
                [f"loser chirped\n($n$ = {int(n[True].sum())})", f"no chirp\n($n$ = {int(n[False].sum())})"],
                loc="upper left", frameon=False, borderaxespad=0.1, handletextpad=0.5, fontsize=ANNOT_FS)
    ax_d.set_ylabel("Cumulative fraction of chases")
    for ax in (ax_b, ax_c, ax_d):
        sns.despine(ax=ax, offset=4, trim=False)
    sns.despine(ax=ax_a, offset=4, trim=False, left=True)
    fig8._letters_cm(fig, [(ax_a, "A", 0.75, 0.55), (ax_b, "B", 0.95, 0.35), (ax_c, "C", 0.95, 0.35),
                           (ax_d, "D", 0.95, 0.35)])
    return fig, {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}, counts


def draw_checks(res_all, rows):
    set_light_style()
    fig8._rc_tweaks()
    fig = plt.figure(figsize=(W_CM / 2.54, F2_H / 2.54))
    ax_e = fig8._axes_cm(fig, E_X, F2_BOTTOM, EF_W, F2_AX_H)
    ax_f = fig8._axes_cm(fig, F_X, F2_BOTTOM, EF_W, F2_AX_H)
    ax_g = fig8._axes_cm(fig, G_X, F2_BOTTOM, G_W, F2_AX_H)
    rate_panel(ax_e, res_all)
    survival_panel(ax_f, res_all)
    label_curves(ax_f, res_all, "after any loser chirp")
    forest_panel(ax_g, rows)
    keys = [Line2D([], [], ls="", marker="o", ms=3.2, mfc="k", mec="white", mew=0.5),
            Line2D([], [], ls="", marker="o", ms=2.9, mfc="white", mec=MUTED, mew=0.7),
            Line2D([], [], ls="", marker="|", ms=6, mew=1.2, color=MUTED)]
    W, H = fig.get_size_inches() * 2.54
    fig.legend(keys, ["matched design", "trap: controls that never chirp", "planted ratio"], ncol=2,
               loc="lower right", bbox_to_anchor=(17.2 / W, (F2_BOTTOM + F2_AX_H + 0.1) / H), frameon=False,
               fontsize=ANNOT_FS, borderaxespad=0, borderpad=0, handletextpad=0.25, columnspacing=0.8,
               handlelength=0.9, labelspacing=0.25)
    for ax in (ax_e, ax_f, ax_g):
        sns.despine(ax=ax, offset=4, trim=False)
    ax_g.spines["left"].set_visible(False)
    fig8._letters_cm(fig, [(ax_e, "E", 0.95, 0.35), (ax_f, "F", 0.95, 0.35), (ax_g, "G", G_LABEL_W, 0.35)])
    return fig, {"E": ax_e, "F": ax_f, "G": ax_g}


TRAP_DY = 0.3  # G: the trap's mark sits this far below its row


def forest_rows(res, res_all, trap_ci, hr, sur_num) -> list[dict]:
    """G's rows: the model's HR; the data's matched ratios (first chirps with the trap below, all chirps,
    bootstrap intervals); the surrogates (median and 2.5-97.5 % of the replicates, design and trap)."""
    def ratio(r):
        return ratio_and_ci(r, 30.0)

    trap_mark = {"value": trap_ci["value"], "ci": trap_ci["ci"], "filled": False, "color": MUTED, "dy": TRAP_DY}
    rows = [{"label": "hazard model (HR)", "marks": [{"value": hr["median"], "ci": hr["hdi"], "color": CHIRP_C}]},
            {"label": "data: first chirps", "marks": [ratio(res), trap_mark]},
            {"label": "data: all chirps", "marks": [ratio(res_all)], "gap_after": True}]
    names = {"shuffled": "shuffled chirps", "sim_hr_1": "simulated HR 1", "sim_hr_0.5": "simulated HR 0.5",
             "sim_hr_2": "simulated HR 2"}
    for key, label in names.items():
        d = sur_num[key]
        marks = [{"value": d["unchirped_at_s_30"]["median"], "ci": d["unchirped_at_s_30"]["q"]},
                 {"value": d["never_chirped_30"]["median"], "ci": d["never_chirped_30"]["q"], "filled": False,
                  "color": MUTED, "dy": TRAP_DY}]
        rows.append({"label": label, "marks": marks, "planted": d["planted"]})
    for y, r in enumerate(rows):
        r["y"] = y + (0.5 if y >= 3 else 0.0)
    return rows


def save(fig, name: str) -> list[Path]:
    paths = [figure(f"{name}.{ext}") for ext in ("pdf", "png")]
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--primary-out", type=Path, default=PRIMARY_OUT,
                    help="run whose stats/offset_window.json holds the window (default: the primary run)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--band", choices=("bootstrap", "leave_out"), default="bootstrap")
    ap.add_argument("--n-surrogates", type=int, default=100)
    ap.add_argument("--n-controls", type=int, default=None, help="sampled controls per trigger (default all)")
    ap.add_argument("--n-rows", type=int, default=N_ROWS)
    ap.add_argument("--sigma", type=float, default=SIGMA_S, help="kernel SD of B and E (s; Fig. 6: 0.3)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--all-intervals", action="store_true",
                    help="also the first interval of each trial (default: the primary model's intervals)")
    ap.add_argument("--suffix", default="", help="appended to the output names (variants)")
    args = ap.parse_args()

    t0 = time.time()
    window, iv, chirps = load(args.primary_out, args.all_intervals)
    common = dict(n_boot=args.n_boot, band=args.band, seed=args.seed, n_controls=args.n_controls, sigma=args.sigma)
    first = rm.first_chirp_triggers(iv)
    res = matched(iv, first, **common)
    res_all = matched(iv, rm.all_chirp_triggers(iv, chirps), **common)
    res_trap = matched(iv, first, **common, rule="never_chirped")
    trap = ratio_and_ci(res_trap, 30.0)
    print(f"matched designs: {time.time() - t0:.0f} s", flush=True)
    sur = surrogates(iv, chirps, args.n_surrogates, args.seed, window[0])
    print(f"surrogates: {time.time() - t0:.0f} s", flush=True)
    hr = fig8.load_hr()
    ev = fig8.analysed_events(pd.read_csv(fig8.OBS_DATA))
    w_ev = fig8.event_weights(ev)
    rows = raster_rows(iv, args.n_rows, args.seed)

    sur_num = surrogate_numbers(sur)
    name = NAME + args.suffix
    fig, axes, counts = draw_main(rows, chirps, iv, res, ev, w_ev)
    print(fig8.axes_extents(fig, list(axes.items())))
    paths = save(fig, name)
    fig2, axes2 = draw_checks(res_all, forest_rows(res, res_all, trap, hr, sur_num))
    print(fig8.axes_extents(fig2, list(axes2.items())))
    paths += save(fig2, f"{name}_checks")

    stats = {
        "window_s": list(window), "intervals": "all" if args.all_intervals else "primary (with a previous interval)",
        "n_intervals": int(len(iv)), "band": args.band, "n_boot": args.n_boot, "seed": args.seed,
        "kernel_sigma_s": args.sigma, "n_controls": args.n_controls or "all eligible, weight 1/n",
        "model_hr": hr, "mh_ratio_person_period": mh_ratio(iv, chirps),
        "first_chirp": matched_numbers(res), "all_chirps": matched_numbers(res_all),
        "trap_never_chirped_controls": matched_numbers(res_trap),
        "raster": counts, "surrogates": sur_num,
        "previous_interval": fig8.summarise_previous(ev),
    }
    out = figure(f"{name}.json")
    out.write_text(json.dumps(stats, indent=1, default=float))
    sur.to_csv(figure(f"{name}_surrogates.csv"), index=False)
    print(json.dumps({k: stats[k] for k in ("model_hr", "mh_ratio_person_period", "first_chirp", "all_chirps",
                                            "trap_never_chirped_controls", "raster")}, indent=1, default=float))
    print("\n".join(map(str, paths + [out])))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
