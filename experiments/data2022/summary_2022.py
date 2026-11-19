"""Summary plots of the 2022 chirps (threshold 0.8, gap-filled tracks, per-fish correction), with 2019/2020 for comparison.

2022: ``competition2022_v2_gapfill_t080_corrected`` (``make_2022_chirps.py``): every
detection >= 0.8 with ``p_chirp``, its probability of being a chirp (the expert's
agreement in its stratum); "expected chirps" = the sum of ``p_chirp``. 18 trials of
``raw/meta.csv`` (6 pairs x 3 bouts a, b, c).
2019/2020: the paper's primary run (``competition2019_v2_20261113_gapfill_t080``,
``chirp`` = kept and score >= 0.8), the 33 analysed trials of ``trial_summary.csv``
(no draws, without the 4 excluded recordings); no correction (agreement at 0.8 there:
0.90, real_v1).

Figures (``$CHIRP_DATA/data2022/summary/``):

1. ``counts_winner_loser.png``: chirps per hour of dark phase, winner vs loser, one line per trial;
2. ``chirp_space.png``: decoder height vs FWHM (calibrated, ``params_ok``), winner vs loser;
3. ``rate_over_time.png``: chirps per 10 min over the recording, winner vs loser;
4. ``bouts.png``: the three bouts of each 2022 pair.

Numbers: ``trials.csv`` (per trial and fish), ``summary.json``.

    .venv/bin/python experiments/data2022/summary_2022.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, REC_2019, REC_2022  # noqa: E402

RUN_2022 = DATA / "detections" / "competition2022_v2_gapfill_t080_corrected"
RUN_2019 = DATA / "detections" / "competition2019_v2_20261113_gapfill_t080"
OUT = DATA / "data2022" / "summary"
EXCLUDED_2019 = ("2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00")
DARK_H = 3.0
COL = {"winner": "#2b8cbe", "loser": "#d95f0e"}


def load_2022() -> pd.DataFrame:
    meta = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"}).set_index("recording")
    rows = []
    for rec in meta.index:
        d = pd.read_parquet(RUN_2022 / rec / "chirps_rtdetr_t080.parquet")
        d = d[d["status"].isin(["winner", "loser"])].copy()
        d["pair"] = meta.loc[rec, "Pairing"][:-1]
        d["bout"] = meta.loc[rec, "Pairing"][-1]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_2019() -> pd.DataFrame:
    t = pd.read_csv(REC_2019 / "trial_summary.csv")
    t = t[(t["draw"] == 0) & ~t["recording"].isin(EXCLUDED_2019)]
    rows = []
    for r in t.itertuples():
        d = pd.read_parquet(RUN_2019 / r.recording / "chirps_rtdetr_gapfill_t080.parquet")
        d = d[d["chirp"]].copy()
        d["status"] = np.select([d["track_id"] == r.win_ID, d["track_id"] == r.lose_ID], ["winner", "loser"], "other")
        d = d[d["status"] != "other"]
        d["recording"] = r.recording
        d["p_chirp"] = 1.0
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Median of x with weights w (the value where the cumulative weight reaches half)."""
    if len(x) == 0:
        return float("nan")
    o = np.argsort(x)
    c = np.cumsum(np.asarray(w, float)[o])
    return float(np.asarray(x)[o][np.searchsorted(c, 0.5 * c[-1])])


def per_trial(d: pd.DataFrame, recordings: list[str], dataset: str) -> pd.DataFrame:
    dark = d[d["t_center_s"] < DARK_H * 3600]
    rows = []
    for rec in recordings:
        for s in ("winner", "loser"):
            g = dark[(dark["recording"] == rec) & (dark["status"] == s)]
            rows.append({"dataset": dataset, "recording": rec, "status": s, "detections": len(g),
                         "expected_chirps": float(g["p_chirp"].sum()),
                         "expected_low": float(g["p_chirp_low"].sum()) if "p_chirp_low" in g else np.nan,
                         "expected_high": float(g["p_chirp_high"].sum()) if "p_chirp_high" in g else np.nan})
    return pd.DataFrame(rows)


def fig_counts(t19: pd.DataFrame, t22: pd.DataFrame, meta: pd.DataFrame, path: Path) -> dict:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, constrained_layout=True)
    out = {}
    floor = 0.1
    for ax, t, name in ((axes[0], t19, "2019/2020"), (axes[1], t22, "2022")):
        w = t[t["status"] == "winner"].set_index("recording")
        lo = t[t["status"] == "loser"].set_index("recording")
        y_w, y_l = w["expected_chirps"] / DARK_H, lo["expected_chirps"] / DARK_H
        more = int((y_l > y_w).sum())
        out[name] = {"trials": len(w), "loser_more": more, "median_winner_per_h": float(y_w.median()),
                     "median_loser_per_h": float(y_l.median())}
        for rec in w.index:
            pair = meta.loc[rec, "Pairing"][:-1] if rec in meta.index else ""
            c = f"C{int(pair) - 1}" if pair else "0.45"
            ax.plot([0, 1], [max(y_w[rec], floor), max(y_l[rec], floor)], "-o", color=c, alpha=0.7, ms=4, lw=1)
            if name == "2022":
                ax.plot([-0.08], [max(w.loc[rec, "detections"] / DARK_H, floor)], "_", color="0.6", ms=8)
                ax.plot([1.08], [max(lo.loc[rec, "detections"] / DARK_H, floor)], "_", color="0.6", ms=8)
        ax.set_yscale("log")
        ax.set_xticks([0, 1], ["winner", "loser"])
        ax.set_xlim(-0.35, 1.35)
        ax.grid(axis="y", alpha=0.3, which="both")
        note = ("detections >= 0.8 (agreement 0.90)" if name == "2019/2020"
                else "expected chirps, one colour per pair; grey: raw detections >= 0.8")
        ax.set_title(f"{name}: {len(w)} trials, loser chirps more in {more}\n{note}", fontsize=9)
    from matplotlib.lines import Line2D

    axes[1].legend(handles=[Line2D([], [], color=f"C{k - 1}", marker="o", ms=4, label=f"pair {k}") for k in range(1, 7)],
                   fontsize=7, frameon=False, loc="upper left", ncol=2)
    axes[0].set_ylabel("chirps per hour of dark phase (log; floor 0.1)")
    fig.suptitle("Losers chirp far more than winners, in 2019/2020 and in 2022", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return out


def fig_space(d19: pd.DataFrame, d22: pd.DataFrame, path: Path) -> dict:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True, constrained_layout=True)
    out = {}
    for ax, d, name in ((axes[0], d19, "2019/2020"), (axes[1], d22, "2022")):
        ok = d[d["params_ok"].astype(bool) & d["height_hz"].gt(0) & d["fwhm_s"].gt(0)]
        for s in ("loser", "winner"):
            g = ok[ok["status"] == s]
            alpha = np.clip(g["p_chirp"].to_numpy(), 0.03, 1) * (0.15 if name == "2019/2020" else 0.5)
            ax.scatter(1e3 * g["fwhm_s"], g["height_hz"], s=3, color=COL[s], alpha=alpha, edgecolors="none",
                       label=f"{s} ({g['p_chirp'].sum():,.0f} chirps)")
            out[f"{name}_{s}"] = {"chirps": float(g["p_chirp"].sum()),
                                  "median_height_hz": weighted_median(g["height_hz"].to_numpy(), g["p_chirp"].to_numpy()),
                                  "median_fwhm_ms": 1e3 * weighted_median(g["fwhm_s"].to_numpy(), g["p_chirp"].to_numpy())}
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("chirp width, FWHM (ms)")
        ax.set_title(name + ("" if name == "2019/2020" else ": each point shaded by its probability of being a chirp"),
                     fontsize=9)
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], ls="", marker="o", ms=5, color=COL[s],
                          label=f"{s} ({out[f'{name}_{s}']['chirps']:,.0f} chirps)") for s in ("loser", "winner")]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("chirp height above baseline (Hz)")
    big = {name: float(np.average(d.loc[d["params_ok"].astype(bool), "height_hz"] >= 200,
                                  weights=d.loc[d["params_ok"].astype(bool), "p_chirp"]))
           for d, name in ((d19, "2019/2020"), (d22, "2022"))}
    out["share_big_ge_200hz"] = big
    fig.suptitle(f"Chirp height against width (decoder): in 2022 almost all chirps are small "
                 f"({100 * big['2022']:.0f} % >= 200 Hz, 2019/2020 {100 * big['2019/2020']:.0f} %)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return out


def fig_time(d19: pd.DataFrame, d22: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    bins = np.arange(0, 4.0 * 3600 + 1, 600)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
    for ax, d, name, end_h in ((axes[0], d19, "2019/2020", 6.0), (axes[1], d22, "2022", 4.0)):
        b = np.arange(0, end_h * 3600 + 1, 600)
        for s in ("winner", "loser"):
            per = []
            for _rec, g in d[d["status"] == s].groupby("recording"):
                h, _ = np.histogram(g["t_center_s"], bins=b, weights=g["p_chirp"])
                per.append(h)
                ax.plot(b[:-1] / 3600 + 1 / 12, h + 0.5, color=COL[s], alpha=0.15, lw=0.7)
            ax.plot(b[:-1] / 3600 + 1 / 12, np.median(per, axis=0) + 0.5, color=COL[s], lw=2.5,
                    label=f"{s}: median over trials")
        ax.axvspan(0, DARK_H, color="0.85", zorder=0)
        ax.text(0.1, 0.97, "dark", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
        ax.text(DARK_H + 0.1, 0.97, "light", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("recording time (h)")
        ax.set_title(f"{name}" + (" (expected chirps; weights after 3 h extrapolated)" if name == "2022" else ""),
                     fontsize=9)
        ax.grid(alpha=0.3, which="both")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    axes[0].set_ylabel("chirps per 10 min (+0.5, log)")
    fig.suptitle("Chirping happens in the dark phase and stops when the light comes on", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bouts(t22: pd.DataFrame, meta: pd.DataFrame, path: Path) -> dict:
    import matplotlib.pyplot as plt

    t = t22.merge(meta["Pairing"], left_on="recording", right_index=True)
    t["pair"], t["bout"] = t["Pairing"].str[:-1], t["Pairing"].str[-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True, constrained_layout=True)
    for ax, s in zip(axes, ("loser", "winner"), strict=True):
        for pair, g in t[t["status"] == s].groupby("pair"):
            g = g.sort_values("bout")
            ax.plot(range(len(g)), np.maximum(g["expected_chirps"] / DARK_H, 0.1), "-o", color=f"C{int(pair) - 1}",
                    label=f"pair {pair}", lw=1.5, ms=5)
        ax.set_xticks([0, 1, 2], ["bout a", "bout b", "bout c"])
        ax.set_yscale("log")
        ax.set_title(f"{s}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, which="both")
    axes[0].set_ylabel("expected chirps per hour of dark phase (log)")
    axes[1].legend(fontsize=8, frameon=False, ncol=2)
    lo = t[t["status"] == "loser"].pivot_table(index="pair", columns="bout", values="expected_chirps")
    fewer = int((lo["c"] < lo["a"]).sum())
    fig.suptitle(f"2022, the same pairs on three consecutive days (same winner each time): the loser chirps less "
                 f"in bout c than in bout a in {fewer} of {len(lo)} pairs", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"loser_fewer_in_c_than_a": fewer, "pairs": len(lo), "loser_per_bout": lo.round(1).to_dict()}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(REC_2022 / "raw" / "meta.csv").rename(columns={"Unnamed: 0": "recording"}).set_index("recording")
    d22, d19 = load_2022(), load_2019()
    t22 = per_trial(d22, list(meta.index), "2022")
    t19 = per_trial(d19, sorted(d19["recording"].unique()), "2019/2020")
    pd.concat([t19, t22]).to_csv(OUT / "trials.csv", index=False)
    summary = {"counts": fig_counts(t19, t22, meta, OUT / "counts_winner_loser.png"),
               "chirp_space": fig_space(d19, d22, OUT / "chirp_space.png")}
    fig_time(d19, d22, OUT / "rate_over_time.png")
    summary["bouts"] = fig_bouts(t22, meta, OUT / "bouts.png")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
