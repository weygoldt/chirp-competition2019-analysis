"""Figure (supplement): the detector's three-window front end against a single window.

The released detector reads an RGB image of three spectrograms (Hann windows
of 205, 51 and 13 ms on one 4.88-Hz grid); the ablation detector the 205-ms
spectrogram in all three channels, trained identically and for the same
number of epochs (arm B of the front-end ablation, continued to epoch 16,
the released detector's best epoch; ``experiments/ablation_scenarios``).

A   The dense chirp train of the failure-mode figure C (2020-03-18-10_34,
    the ~702-Hz fish of a draw; ``v2_fig_extract.SNIPPETS["f3_dense"]``), a
    3-s excerpt (``EXCERPT``) as each detector receives it: top, the
    single-window input (one spectrogram in all three channels: grey);
    bottom, the RGB input. White circles: each detector's detections at the
    operating threshold, at their predicted emitter EOD$f$ (both detectors run
    with ``predict_span`` on the production tile grid,
    ``v2_fig_extract.py frontends``; the released detector's boxes equal the
    production run's); black triangles: the fish's chirps counted without
    the detector (``plot_v2_failure_modes.chirp_events``, as in the
    failure-mode figure); thin black lines: the tracks. The counts in the
    panel are over the whole 17.3-s train.
B   Recall on simulated chirp trains against the interval to the same fish's
    nearest chirp (benchmark sweep b: two fish 40-80 Hz apart, small chirps
    in trains of 1000-75 ms, 180 scenes; matching within 25 ms). Grey band:
    the interquartile range of the intervals of the train in A; dashed: the
    shortest interval of the training data (200 ms).
C   Correct emitter against the emitter's ΔF to the nearest other fish
    (sweep a: two fish 0.5-20 Hz apart, 270 scenes; chirps both detectors
    found, matching within 100 ms). Dashed: chance for two fish.

B and C: true positives at score >= the operating threshold (as in the
performance figure), Wilson 95 % intervals; the differences (single minus
three windows) with a paired bootstrap over scenes (``N_BOOT``) go to the
stats, also pooled (intervals < 125 ms, 125-150 ms, >= 150 ms; ΔF < 4 Hz;
all), at threshold 0.5, and for the single-window detector at epoch 22.

Inputs: the benchmark's per-label table
(``$CHIRP_DATA/runs/ablation_frontend/scenarios/paper/bench_labels.parquet``,
``experiments/ablation_scenarios/export_for_paper.py``; its sha256 is
checked), the snippet and ``f3_dense_frontends.parquet`` of
``v2_fig_extract.py``, the bundles' ``bundle.json`` (training epochs).

Environment: ``paper/.venv``::

    cd paper && unset VIRTUAL_ENV && uv run python scripts/plot_v2_frontend_ablation.py

Runtime: ~15 s.
"""

from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import paper_config
import v2_fig_common as C
from manuscript_stats import write_stats
from plot_v2_failure_modes import EVENTS, chirp_events, match_times
from style import other_color, set_light_style

NAME = "plot_v2_frontend_ablation"
THR = C.OPERATING
BENCH = C.CHIRP_DATA / "runs" / "ablation_frontend" / "scenarios" / "paper" / "bench_labels.parquet"
BENCH_SHA256 = "adb5b68f4c95a8115d46ad8d5e7243b3e72784bb5af145ddfbf7d427628c0128"
BUNDLES = {
    "rgb": C.CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_r1" / "final" / "bundle",
    "single": C.CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_longonly_r1c" / "bundles" / "epoch_16",
    "single_early": C.CHIRP_DATA / "runs" / "dualhead_r18_sim_v4_longonly_r1" / "final" / "bundle",
}
BENCH_KEY = {"rgb": "A", "single": "B16", "single_e22": "B22"}
SNIPPET = "f3_dense"
EXCERPT = (11406.0, 11409.0)  # [s] the 3-s excerpt of A (inside the 6-s candidate)
BAND = (625.0, 900.0)  # [Hz] of A (room below the fish for the scale bar)
N_BOOT = 4000
DT_EDGES_MS = [0.0, 80.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 1000.0, np.inf]
DT_KEYS = ["below_80_ms", "80_to_100_ms", "100_to_125_ms", "125_to_150_ms", "150_to_200_ms",
           "200_to_300_ms", "300_to_500_ms", "500_to_1000_ms", "1000_ms_up"]
DF_EDGES_HZ = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, np.inf]
DF_KEYS = ["below_0p5_hz", "0p5_to_1_hz", "1_to_2_hz", "2_to_3_hz", "3_to_4_hz", "4_to_6_hz",
           "6_to_8_hz", "8_to_12_hz", "12_hz_up"]
POOLED_DT = {"below_125_ms": (0.0, 0.125), "125_to_150_ms": (0.125, 0.15), "150_ms_up": (0.15, np.inf)}

# palette (Figs. 5-7, as the performance figure): the released detector in the
# indigo of the other-event colour, filled; the single-window detector grey, open
SERIES = {
    "rgb": {"color": other_color, "mfc": other_color, "mec": "k", "mew": 0.5,
            "label": "Three windows (released)"},
    "single": {"color": "#808080", "mfc": "white", "mec": "#808080", "mew": 0.8,
               "label": "Single 205-ms window"},
}
LW, EB_LW, PT_MS = 1.0, 0.8, 3.3  # as the performance figure
DET_MS = 2.6  # detection circles in A

# layout [cm]
W = 17.4
L, R = 1.5, 17.0
STRIP_H, STRIP_GAP = 2.0, 0.45
TOP = 0.75  # above the first strip: the event triangles and the letter
ROW_GAP = 1.45  # from the second strip to the plots (scale bar, letters)
PLOT_H = 4.0
BOTTOM = 1.3
PLOT_W = 6.6
PLOT_GAP = R - L - 2 * PLOT_W
H = TOP + 2 * STRIP_H + STRIP_GAP + ROW_GAP + PLOT_H + BOTTOM


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def load_bench() -> pd.DataFrame:
    sha = hashlib.sha256(BENCH.read_bytes()).hexdigest()
    if sha != BENCH_SHA256:
        msg = f"{BENCH}: sha256 {sha}, expected {BENCH_SHA256} (frozen 2026-11-26)"
        raise ValueError(msg)
    return pd.read_parquet(BENCH)


def detected(b: pd.DataFrame, model: str, thr: float, tol: str = "25") -> np.ndarray:
    m = BENCH_KEY[model]
    return (b[f"found{tol}_{m}"].to_numpy(bool)
            & (b[f"score{tol}_{m}"].fillna(-1).to_numpy(float) >= thr))


class Bootstrap:
    """Paired bootstrap over scenes: the same draws for every quantity."""

    def __init__(self, scenes: np.ndarray, n_boot: int, seed: int):
        self.names, self.idx = np.unique(scenes, return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.names), size=(n_boot, len(self.names)))
        self.w = np.stack([np.bincount(d, minlength=len(self.names)) for d in draws]).astype(float)

    def ratio(self, num: np.ndarray, den: np.ndarray, mask: np.ndarray) -> np.ndarray:
        def sums(v):
            return np.bincount(self.idx[mask], weights=v[mask].astype(float), minlength=len(self.names))

        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.w @ sums(num)) / (self.w @ sums(den))

    def diff(self, num_a, den_a, num_b, den_b, mask) -> tuple[float, float]:
        d = self.ratio(num_b, den_b, mask) - self.ratio(num_a, den_a, mask)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return float(lo), float(hi)


def compare(boot: Bootstrap, mask, num: dict, den: dict, x=None) -> dict:
    """Both detectors' proportions in ``mask`` (Wilson 95 %) and their
    difference (single minus three windows, paired bootstrap 95 %)."""
    out = {"n": int(den["rgb"][mask].sum())}
    if x is not None:
        out["x"] = float(x)
    for model in ("rgb", "single"):
        k, n = float(num[model][mask].sum()), float(den[model][mask].sum())
        lo, hi = C.wilson(k, n)
        out[model] = {"k": int(k), "n": int(n), "p": k / n, "lo": float(lo), "hi": float(hi)}
    out["single_minus_rgb"] = out["single"]["p"] - out["rgb"]["p"]
    out["single_minus_rgb_lo"], out["single_minus_rgb_hi"] = boot.diff(
        num["rgb"], den["rgb"], num["single"], den["single"], mask)
    return out


def interval_sweep(b: pd.DataFrame, boot: Bootstrap, thr: float, models=("rgb", "single")) -> dict:
    rate = b["sweep"].to_numpy() == "rate"
    dt = b["dt_same_s"].to_numpy(float)
    ones = np.ones(len(b))
    num = {"rgb": detected(b, models[0], thr), "single": detected(b, models[1], thr)}
    den = {"rgb": ones, "single": ones}
    out = {"by_interval": {}, "pooled": {}}
    for key, lo, hi in zip(DT_KEYS, DT_EDGES_MS[:-1], DT_EDGES_MS[1:], strict=True):
        m = rate & (dt * 1e3 >= lo) & (dt * 1e3 < hi)
        out["by_interval"][key] = compare(boot, m, num, den, x=np.median(dt[m]) * 1e3)
    for key, (lo, hi) in POOLED_DT.items():
        out["pooled"][key] = compare(boot, rate & (dt >= lo) & (dt < hi), num, den)
    out["pooled"]["all"] = compare(boot, rate, num, den)
    return out


def df_sweep(b: pd.DataFrame, boot: Bootstrap, thr: float, models=("rgb", "single")) -> dict:
    sweep = b["sweep"].to_numpy() == "df"
    df = b["emitter_df_hz"].to_numpy(float)
    found = {k: detected(b, m, thr, "100") for k, m in zip(("rgb", "single"), models, strict=True)}
    both = found["rgb"] & found["single"]
    num = {k: b[f"correct100_{BENCH_KEY[m]}"].to_numpy(bool)
           for k, m in zip(("rgb", "single"), models, strict=True)}
    den = {"rgb": np.ones(len(b)), "single": np.ones(len(b))}
    out = {"by_df": {}, "pooled": {}}
    for key, lo, hi in zip(DF_KEYS, DF_EDGES_HZ[:-1], DF_EDGES_HZ[1:], strict=True):
        m = sweep & both & (df >= lo) & (df < hi)
        out["by_df"][key] = compare(boot, m, num, den, x=np.median(df[m]))
    out["pooled"]["below_4_hz"] = compare(boot, sweep & both & (df < 4.0), num, den)
    out["pooled"]["all"] = compare(boot, sweep & both, num, den)
    return out


# ---------------------------------------------------------------------------
# the real chirp train (A)
# ---------------------------------------------------------------------------
def assign_to_tracks(det: pd.DataFrame, snip: dict) -> np.ndarray:
    """Nearest track to each detection's emitter EOD$f$ at its box centre
    (within the run's tolerance), else -1; the same for both detectors."""
    centre = 0.5 * (det.t_start_s.to_numpy() + det.t_end_s.to_numpy())
    best, dist = np.full(len(det), -1), np.full(len(det), np.inf)
    for tid, (tt, ff) in snip["tracks"].items():
        inside = (centre >= tt[0]) & (centre <= tt[-1])
        d = np.where(inside, np.abs(det.emitter_hz.to_numpy() - np.interp(centre, tt, ff)), np.inf)
        better = d < dist
        best[better], dist[better] = tid, d[better]
    best[dist > C.TOLERANCE_HZ] = -1
    return best


def real_train(snip: dict) -> tuple[dict, dict]:
    m = snip["meta"]
    t0, t1 = m["t_start_s"], m["t_end_s"]
    det = pd.read_parquet(C.SNIPPETS / f"{SNIPPET}_frontends.parquet")
    check = json.loads((C.SNIPPETS / f"{SNIPPET}_frontends.json").read_text())
    det = det[(det.drop_reason.astype(str) == "") & (det.score >= THR)].copy()
    centre = 0.5 * (det.t_start_s + det.t_end_s)
    det = det[(centre >= t0) & (centre <= t1)].copy()
    det["track"] = assign_to_tracks(det, snip)
    rgb = det[det.model == "rgb"]
    emitter = int(rgb.track.value_counts().drop(-1, errors="ignore").idxmax())
    events = chirp_events(snip, t0, t1, emitter)
    out = {"n_events": len(events), "duration_s": t1 - t0,
           "emitter_hz": C.track_median(snip, emitter, t0, t1),
           "event_interval_ms": {"median": float(np.median(np.diff(events)) * 1e3),
                                 "q25": float(np.percentile(np.diff(events), 25) * 1e3),
                                 "q75": float(np.percentile(np.diff(events), 75) * 1e3),
                                 "min": float(np.min(np.diff(events)) * 1e3)},
           "rgb_equals_production": int(check["check_rgb_vs_production"]["n_here"]
                                        == check["check_rgb_vs_production"]["n_production"])}
    drawn = {"events": events, "emitter": emitter, "det": det}
    for model in ("rgb", "single"):
        mine = det[(det.model == model) & (det.track == emitter)]
        centres = 0.5 * (mine.t_start_s + mine.t_end_s).to_numpy()
        ue, _ = match_times(events, np.sort(centres), EVENTS["match_s"])
        per_box = [int(((events >= r.t_start_s) & (events <= r.t_end_s)).sum()) for r in mine.itertuples()]
        out[model] = {"n_detections": len(mine), "n_matched": len(ue),
                      "fraction_matched": len(ue) / len(events),
                      "n_boxes_spanning_2_or_more": int(sum(p >= 2 for p in per_box))}
    out["single_minus_rgb_matched"] = out["single"]["n_matched"] - out["rgb"]["n_matched"]
    return out, drawn


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def strip(ax, snip: dict, drawn: dict, model: str, label: str, real: dict) -> None:
    a, b = EXCERPT
    f0, f1 = BAND
    if model == "rgb":
        C.show_rgb(ax, snip, a, b, f0, f1)
    else:
        img, ext = C.spec_crop(snip, a, b, f0, f1, 0, key="img")
        ax.imshow(np.clip(img, 0, 1), extent=(ext[0] - a, ext[1] - a, ext[2], ext[3]), aspect="auto",
                  origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="antialiased",
                  rasterized=True)
        ax.set_xlim(0, b - a)
        ax.set_ylim(f0, f1)
    for tid, (tt, ff) in snip["tracks"].items():
        keep = (tt >= a) & (tt <= b)
        if keep.any() and f0 < np.median(ff[keep]) < f1:
            ax.plot(tt[keep] - a, ff[keep], color=C.TRACK_COLOR, lw=C.TRACK_LW, zorder=3)
    d = drawn["det"]
    d = d[d.model == model]
    c = 0.5 * (d.t_start_s + d.t_end_s).to_numpy()
    inside = (c >= a) & (c <= b) & (d.emitter_hz.to_numpy() >= f0) & (d.emitter_hz.to_numpy() <= f1)
    ax.plot(c[inside] - a, d.emitter_hz.to_numpy()[inside], "o", ms=DET_MS, mfc="none", mec="white",
            mew=0.8, zorder=5)
    ev = drawn["events"]
    ev = ev[(ev >= a) & (ev <= b)]
    ax.plot(ev - a, np.full(len(ev), f1 + 0.06 * (f1 - f0)), "v", ms=2.6, color="k", mec="none",
            clip_on=False, zorder=6)
    ax.text(0.012, 0.93, label, transform=ax.transAxes, ha="left", va="top", **C.ANNOT, **C.DARK_BACK)
    k = real[model]["n_matched"]
    ax.text(0.988, 0.93, f"{k} of {real['n_events']} chirps detected", transform=ax.transAxes,
            ha="right", va="top", **C.ANNOT, **C.DARK_BACK)
    C.spec_yaxis(ax, ticks=[700, 800], ylabel=False)


def proportion_series(ax, table: dict, model: str) -> None:
    rows = [v for v in table.values() if v[model]["n"] > 0]
    x = np.array([r["x"] for r in rows])
    p = np.array([r[model]["p"] for r in rows])
    lo = np.array([r[model]["lo"] for r in rows])
    hi = np.array([r[model]["hi"] for r in rows])
    s = SERIES[model]
    ax.plot(x, p, color=s["color"], lw=LW, zorder=3, clip_on=False)
    ax.errorbar(x, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)], fmt="o", ms=PT_MS,
                mfc=s["mfc"], mec=s["mec"], mew=s["mew"], ecolor=s["color"], elinewidth=EB_LW,
                capsize=0, zorder=4, clip_on=False, label=s["label"])


def log_ticks(ax, ticks) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def main() -> None:
    set_light_style()
    C.rc_tweaks()
    bench = load_bench()
    boot = Bootstrap(bench["scene"].to_numpy(), N_BOOT, paper_config.SEED)
    rate = interval_sweep(bench, boot, THR)
    df = df_sweep(bench, boot, THR)
    rate_05 = interval_sweep(bench, boot, 0.5)
    rate_e22 = interval_sweep(bench, boot, THR, models=("rgb", "single_e22"))
    df_e22 = df_sweep(bench, boot, THR, models=("rgb", "single_e22"))
    snip = C.load_snippet(SNIPPET)
    real, drawn = real_train(snip)
    epochs = {k: json.loads((p / "bundle.json").read_text())["training"] for k, p in BUNDLES.items()}
    report = {
        "operating_threshold": THR,
        "bench": {
            "n_scenes": int(bench["scene"].nunique()),
            "n_labels": len(bench),
            "rate": {"n_scenes": int(bench.loc[bench.sweep == "rate", "scene"].nunique()),
                     "n_labels": int((bench.sweep == "rate").sum()),
                     "fastest_level_ms": float(bench.loc[bench.sweep == "rate", "level_value"].min() * 1e3),
                     "slowest_level_ms": float(bench.loc[bench.sweep == "rate", "level_value"].max() * 1e3)},
            "df": {"n_scenes": int(bench.loc[bench.sweep == "df", "scene"].nunique()),
                   "n_labels": int((bench.sweep == "df").sum()),
                   "closest_level_hz": float(bench.loc[bench.sweep == "df", "level_value"].min()),
                   "widest_level_hz": float(bench.loc[bench.sweep == "df", "level_value"].max())},
            "n_boot": N_BOOT,
        },
        "training": {"rgb_best_epoch": epochs["rgb"]["best_epoch"],
                     "rgb_epochs_run": epochs["rgb"]["epochs_run"],
                     "single_epoch": epochs["single"]["best_epoch"],
                     "single_early_best_epoch": epochs["single_early"]["best_epoch"],
                     "single_early_epochs_run": epochs["single_early"]["epochs_run"]},
        "recall": rate,
        "recall_t0_5": {"pooled": rate_05["pooled"]},
        "assign": df,
        "epoch22": {"recall_pooled": rate_e22["pooled"], "assign_pooled": df_e22["pooled"]},
        "real": real,
    }

    fig = plt.figure(figsize=(W * C.CM, H * C.CM))
    y_a1 = H - TOP - STRIP_H
    y_a2 = y_a1 - STRIP_GAP - STRIP_H
    y_plot = BOTTOM
    ax_a1 = C.axes_cm(fig, L, y_a1, R - L, STRIP_H)
    ax_a2 = C.axes_cm(fig, L, y_a2, R - L, STRIP_H)
    strip(ax_a1, snip, drawn, "single", r"Single window: 205~ms in all three channels", real)
    strip(ax_a2, snip, drawn, "rgb", r"Three windows: 205 (R), 51 (G) and 13~ms (B)", real)
    C.scalebar(ax_a2, EXCERPT[1] - EXCERPT[0] - 0.08, BAND[0] + 0.05 * (BAND[1] - BAND[0]), 0.5,
               "0.5~s", above=True)
    fig.text((L - 1.05) / W, (y_a2 + STRIP_H + STRIP_GAP / 2) / H, "Frequency [Hz]", rotation=90,
             ha="center", va="center")

    ax_b = C.axes_cm(fig, L, y_plot, PLOT_W, PLOT_H)
    ax_c = C.axes_cm(fig, L + PLOT_W + PLOT_GAP, y_plot, PLOT_W, PLOT_H)
    iv = real["event_interval_ms"]
    ax_b.axvspan(iv["q25"], iv["q75"], color="#e6e6e6", lw=0, zorder=0)
    ax_b.text(np.sqrt(iv["q25"] * iv["q75"]), 0.405, "train\nin A", ha="center", va="bottom",
              fontsize=C.ANNOT_FS, color=C.MUTED)
    ax_b.axvline(200, color="k", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_b.text(212, 0.405, "shortest interval\nin training", ha="left", va="bottom", fontsize=C.ANNOT_FS,
              color=C.MUTED)
    for model in ("single", "rgb"):
        proportion_series(ax_b, rate["by_interval"], model)
    log_ticks(ax_b, [75, 100, 150, 200, 300, 500, 1000])
    ax_b.set_xlim(66, 1500)
    ax_b.set_ylim(0.4, 1.0)
    ax_b.set_xlabel("Interval to the fish's nearest chirp [ms]")
    ax_b.set_ylabel("Recall")

    ax_c.axhline(0.5, color="k", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_c.text(21, 0.515, "chance (two fish)", ha="right", va="bottom", fontsize=C.ANNOT_FS, color=C.MUTED)
    for model in ("single", "rgb"):
        proportion_series(ax_c, df["by_df"], model)
    log_ticks(ax_c, [0.3, 1, 3, 10, 20])
    ax_c.set_xlim(0.2, 30)
    ax_c.set_ylim(0.4, 1.0)
    ax_c.set_xlabel(r"$\Delta f$ to the nearest other fish [Hz]")
    ax_c.set_ylabel("Correct emitter")
    handles, labels = ax_c.get_legend_handles_labels()
    order = [labels.index(SERIES[m]["label"]) for m in ("rgb", "single")]
    ax_c.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                frameon=False, handletextpad=0.4)
    for ax in (ax_b, ax_c):
        sns.despine(ax=ax, offset=5, trim=False)

    C.letters_cm(fig, [(ax_a1, "A", 1.2, 0.35), (ax_b, "B", 1.2, 0.35), (ax_c, "C", 1.2, 0.35)])
    print(C.axes_extents(fig, [("A single", ax_a1), ("A rgb", ax_a2), ("B", ax_b), ("C", ax_c)]))
    paths = C.save(fig, NAME)
    print(json.dumps(report, indent=1, default=float))
    print("\n".join(map(str, paths)))
    write_stats("v2_frontend_ablation", report)


if __name__ == "__main__":
    main()
