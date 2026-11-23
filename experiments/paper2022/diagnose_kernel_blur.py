"""Why are the 2022 chase-triggered chirp kernels blunter than 2019's? Timing noise vs event definition.

1. Timing noise: the model's onset / offset errors against the author's labels (leave one
   clip out, ``video2022/chases/model/final_v2/chases.csv``). If a detected time is the
   true one plus an error e, the kernel around detected times is the true kernel averaged
   over the errors: k_det(t) = mean_e k_true(t + e). Applied to the 2019 kernel (rate /
   jittered baseline, paper primary run): does 2019 blurred by our errors look like 2022?
2. Event definition: gaps between consecutive chases (offset to next onset) in 2019
   (BORIS, the 15 annotated trials) and 2022 (locked model chases, merged as in the
   2022 dataset folder): does the model cut episodes that the 2019 annotation keeps whole?

Read only (2019 files, model outputs).

    .venv/bin/python experiments/paper2022/diagnose_kernel_blur.py

Output: ``$CHIRP_DATA/paper2022/output/chases_model_v2/figures/diagnose_kernel_blur.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA

RUN = DATA / "paper2022" / "output" / "chases_model_v2"
RATES_2022 = RUN / "data" / "processed" / "event_time_rates"
RATES_2019 = Path("/home/weygoldt/wrk/analyses/chirp_competition_upgrade/paper/output/rtdetr_gapfill_t080/data/"
                  "processed/event_time_rates")
REC_2019 = Path("/mnt/data2/2019_tube_competition")
LOCO = DATA / "video2022" / "chases" / "model" / "final_v2" / "chases.csv"
DATASET_2022 = DATA / "paper2022" / "dataset"
BAD_2019 = ("2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00")


def ratio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as fh:
        d = json.load(fh)
    t = np.array(d["conv_time"])
    return t, np.median(d["jackknifed_eventrate"], axis=0) / np.median(d["bootstrapped_baseline_jittered"], axis=0)


def blur(t: np.ndarray, k: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """mean over errors e of k(t + e) (outside the window: 1, the baseline)."""
    return np.mean([np.interp(t + e, t, k, left=1.0, right=1.0) for e in errors], axis=0)


def gaps_2019() -> np.ndarray:
    ts = pd.read_csv(REC_2019 / "trial_summary.csv")
    ts = ts[(ts["group"] >= 5) & (ts["draw"] == 0) & ~ts["recording"].isin(BAD_2019)]
    out = []
    for rec in ts["recording"]:
        f = REC_2019 / rec / ("-".join(rec.split("-")[:3]) + ".csv")
        if not f.exists():
            continue
        b = pd.read_csv(f)
        on = np.sort(b.loc[b["Behavior"] == 0, "Start (s)"].to_numpy())
        off = np.sort(b.loc[b["Behavior"] == 1, "Start (s)"].to_numpy())
        pairs = [(a, off[off > a][0]) for a in on if (off > a).any()]
        out += [pairs[k + 1][0] - pairs[k][1] for k in range(len(pairs) - 1)]
    return np.array(out)


def gaps_2022() -> np.ndarray:
    out = []
    for f in sorted(DATASET_2022.glob("*/2022-*.csv")):
        b = pd.read_csv(f)
        on = np.sort(b.loc[b["Behavior"] == 0, "Start (s)"].to_numpy())
        off = np.sort(b.loc[b["Behavior"] == 1, "Start (s)"].to_numpy())
        out += list(on[1:] - off[:-1])
    return np.array(out)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loco = pd.read_csv(LOCO)
    m = loco[(loco["method"] == "model") & loco["found"]]
    err = {"agon": m["onset_err_s"].dropna().to_numpy(), "agoff": m["offset_err_s"].dropna().to_numpy()}
    for k, e in err.items():
        q = np.percentile(np.abs(e), [50, 75, 90])
        print(f"{k}: {len(e)} labelled chases, |error| median {q[0]:.2f} s, 75 % {q[1]:.2f} s, 90 % {q[2]:.2f} s, "
              f"bias {np.median(e):+.2f} s")
    g19, g22 = gaps_2019(), gaps_2022()
    for name, g in (("2019 BORIS", g19), ("2022 model", g22)):
        print(f"{name}: {len(g)} gaps between chases, median {np.median(g):.1f} s, share < 2 s {np.mean(g < 2):.2f}, "
              f"< 5 s {np.mean(g < 5):.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, center in enumerate(("agon", "agoff")):
        ax = axes[0, col]
        t19, k19 = ratio(RATES_2019 / f"lchirps@{center}.json")
        t22, k22 = ratio(RATES_2022 / f"lchirps@{center}.json")
        ax.plot(t19, k19, color="0.3", lw=1.5, label="2019 (BORIS chases)")
        ax.plot(t19, blur(t19, k19, err[center]), color="0.3", ls="--", lw=1.5,
                label="2019 blurred with the model's timing errors")
        ax.plot(t22, k22, color="#d95f0e", lw=1.8, label="2022 (model chases)")
        ax.axhline(1, color="0.7", lw=0.8)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlim(-30, 30)
        ax.set_title(f"loser's chirps around the chase {'onset' if center == 'agon' else 'offset'}", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=7, frameon=False)
        print(f"{center}: peak ratio 2019 {k19.max():.2f}, 2019 blurred {blur(t19, k19, err[center]).max():.2f}, "
              f"2022 {k22.max():.2f}")
    axes[0, 0].set_ylabel("rate / jittered baseline")
    ax = axes[1, 0]
    bins = np.arange(-4, 4.01, 0.25)
    ax.hist(err["agon"], bins=bins, color="0.5", alpha=0.7, label=f"onset (n = {len(err['agon'])})")
    ax.hist(err["agoff"], bins=bins, histtype="step", color="k", lw=1.5, label=f"offset (n = {len(err['agoff'])})")
    ax.set_xlabel("model - author (s), labelled chases, leave one clip out")
    ax.set_ylabel("chases")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 1]
    bins = np.r_[np.arange(0, 30.01, 1), np.inf]
    for name, g, color in (("2019 BORIS", g19, "0.3"), ("2022 model", g22, "#d95f0e")):
        ax.hist(np.minimum(g, 30.5), bins=[*bins[:-1].tolist(), 31], density=True, histtype="step", lw=1.6,
                color=color, label=f"{name} (n = {len(g)}, median {np.median(g):.0f} s)")
    ax.set_xlabel("gap between consecutive chases (s; last bin: >= 30)")
    ax.set_ylabel("share")
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Why are the 2022 kernels blunter? Timing errors (top: blur test) and chase gaps (bottom right)",
                 fontsize=10)
    out = RUN / "figures" / "diagnose_kernel_blur.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
