"""Chase onset / offset-triggered chirp rates of 2022, drawn like the 2019 figure, and both years side by side.

Input: the result files of the unchanged 2019 script
``paper/scripts/analyze_event_time_rates_trial.py --ag-events`` run on the 2022 dataset
folder (``build_dataset.py``): ``<run>/data/processed/event_time_rates/{l,w}chirps@{agon,agoff}.json``
(Gaussian kernel, sigma 0.3 s; jackknifed rate over trials; baseline = the same with
the chase times jittered by up to 120 s). The 2019 files of the paper's primary run are
read for the comparison (read only).

Figures (``<run>/figures/``):

- ``chase_triggered_rates_2022.png``: as the chase columns of the 2019 figure
  (``plot_event_time_rates_winner_loser_chirp.py``, which cannot run on 2022 because it
  expects contact results): per centring a rate panel (chirps / min, median and 95 % band
  of the jackknifed rate, grey = jittered baseline) and the per-trial rates (each trial
  scaled to its maximum);
- ``chase_triggered_rates_2019_vs_2022.png``: loser and winner rates divided by their
  baseline, 2019 and 2022, same code.

    .venv/bin/python experiments/paper2022/plot_chase_triggered_rates.py [--run chases_model_v2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA

OUT = DATA / "paper2022" / "output"
PAPER_2019 = Path("/home/weygoldt/wrk/analyses/chirp_competition_upgrade/paper/output/rtdetr_gapfill_t080/data/"
                  "processed/event_time_rates")
COLORS = {"lchirps": "#d95f0e", "wchirps": "#2b8cbe"}
WHO = {"lchirps": "loser's chirps", "wchirps": "winner's chirps"}
WHEN = {"agon": "chase onset", "agoff": "chase offset"}


def load(folder: Path, chirps: str, center: str) -> dict:
    with open(folder / f"{chirps}@{center}.json") as fh:
        return json.load(fh)


def summary(d: dict) -> dict:
    """Peak after 0 and mean baseline (chirps / min), as the 2019 plot prints them."""
    t = np.array(d["conv_time"])
    rate = np.percentile(d["jackknifed_eventrate"], 50, axis=0) * 60
    base = np.percentile(d["bootstrapped_baseline_jittered"], 50, axis=0) * 60
    pos = t > 0
    k = int(np.argmax(rate[pos]))
    return {"peak_time_s": float(t[pos][k]), "peak_per_min": float(rate[pos][k]), "baseline_per_min": float(base.mean()),
            "trials": len(d["trial_identifiers"]), "centres": int(sum(len(x) > 0 for x in d["centered_times"]))}


def rate_panel(ax, d: dict, color: str) -> None:
    t = np.array(d["conv_time"])
    b = np.percentile(d["bootstrapped_baseline_jittered"], [2.5, 50, 97.5], axis=0) * 60
    r = np.percentile(d["jackknifed_eventrate"], [2.5, 50, 97.5], axis=0) * 60
    ax.fill_between(t, b[0], b[2], color="0.75", alpha=0.6, lw=0)
    ax.plot(t, b[1], color="0.5", lw=1)
    ax.fill_between(t, r[0], r[2], color=color, alpha=0.35, lw=0)
    ax.plot(t, r[1], color=color, lw=1.2)
    ax.axvline(0, color="k", ls=":", lw=0.8)


def trial_panel(ax, d: dict) -> None:
    t = np.array(d["conv_time"])
    per = np.array(d["per_trial_rates"])
    per = per / (per.max(axis=1, keepdims=True) + 1e-12)
    ax.imshow(per, aspect="auto", origin="lower", cmap="Greys", extent=[t[0], t[-1], 0, len(per)],
              interpolation="none")
    ax.axvline(0, color="C3", ls=":", lw=0.8)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="chases_model_v2")
    args = ap.parse_args()
    run = OUT / args.run
    folder = run / "data" / "processed" / "event_time_rates"
    figs = run / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.6, 1, 1.6, 1])
    for row, chirps in enumerate(("lchirps", "wchirps")):
        for col, center in enumerate(("agon", "agoff")):
            d = load(folder, chirps, center)
            ax_r = fig.add_subplot(grid[2 * row, col])
            ax_t = fig.add_subplot(grid[2 * row + 1, col], sharex=ax_r)
            rate_panel(ax_r, d, COLORS[chirps])
            trial_panel(ax_t, d)
            s = summary(d)
            ax_r.set_title(f"{WHO[chirps]} around the {WHEN[center]}\n({s['trials']} trials; peak "
                           f"{s['peak_per_min']:.1f}/min at +{s['peak_time_s']:.1f} s, baseline "
                           f"{s['baseline_per_min']:.1f}/min)", fontsize=8)
            ax_r.set_xlim(-30, 30)
            ax_r.tick_params(labelbottom=False)
            if col == 0:
                ax_r.set_ylabel("chirps / min")
                ax_t.set_ylabel("trials")
            if row == 1:
                ax_t.set_xlabel(f"time from the {WHEN[center]} (s)")
    fig.suptitle("2022: chirps around chase onsets and offsets (2019 analysis code, unchanged; locked model "
                 "chases, chirps >= 0.8)", fontsize=9)
    fig.savefig(figs / "chase_triggered_rates_2022.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    for row, chirps in enumerate(("lchirps", "wchirps")):
        for col, center in enumerate(("agon", "agoff")):
            ax = axes[row, col]
            for year, f, ls in (("2019", PAPER_2019, "-"), ("2022", folder, "-")):
                d = load(f, chirps, center)
                t = np.array(d["conv_time"])
                ratio = np.array(d["jackknifed_eventrate"]) / np.median(d["bootstrapped_baseline_jittered"], axis=0)
                q = np.percentile(ratio, [2.5, 50, 97.5], axis=0)
                color = "0.3" if year == "2019" else COLORS[chirps]
                ax.fill_between(t, q[0], q[2], color=color, alpha=0.2, lw=0)
                ax.plot(t, q[1], color=color, ls=ls, lw=1.5,
                        label=f"{year} ({len(d['trial_identifiers'])} trials)")
                s = summary(d)
                print(f"{year} {chirps}@{center}: trials {s['trials']}, peak {s['peak_per_min']:.2f}/min at "
                      f"+{s['peak_time_s']:.1f} s, baseline {s['baseline_per_min']:.2f}/min, peak/baseline "
                      f"{s['peak_per_min'] / s['baseline_per_min']:.2f}")
            ax.axhline(1, color="0.6", lw=0.8)
            ax.axvline(0, color="k", ls=":", lw=0.8)
            ax.set_xlim(-30, 30)
            ax.set_title(f"{WHO[chirps]} around the {WHEN[center]}", fontsize=9)
            if col == 0:
                ax.set_ylabel("rate / jittered baseline")
            if row == 1:
                ax.set_xlabel(f"time from the {WHEN[center]} (s)")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Chase-triggered chirp rates, 2019 (hand-annotated chases) and 2022 (model chases): same analysis "
                 "code", fontsize=9)
    fig.savefig(figs / "chase_triggered_rates_2019_vs_2022.png", dpi=200)
    plt.close(fig)
    print(f"-> {figs}")


if __name__ == "__main__":
    main()
