"""Chirps of the loser (and winner) around chase onsets and offsets, relative to random moments in the same trials.

Chases in which the winner chases, from one or more sources: model runs
(``c9_chase_model_apply.py``, ``chases/model/<name>/events.csv``) and, with ``--old``,
the old rule-based detector (``c1_chases.py`` run, identities from
``identity/<run>/events_identity.csv``). Chirps: the corrected 0.8 dataset
(``c4_chirps_around_chases.CHIRPS``), weighted by ``p_chirp``.

Because the chirp rate differs a lot between trials (pair 3 chirps most), every rate
is divided by what random moments with both fish in view in the *same* trial give:
per bin, sum over trials of the chirps around the chases / sum over trials of
(chases x that trial's chirp rate at random moments x bin width). 1 = as at random.
For the first model run, its chases are also split by the model's confidence (mean
probability >= ``CONFIDENT`` or below).

    .venv/bin/python experiments/video2022/c10_chirps_model_chases.py model_v2 model_v1 --old

Output: ``$CHIRP_DATA/video2022/chases/model/<first name>/chirps_around_chases.png``,
``chirps_sources.png``, ``chirps_around_chases.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c4_chirps_around_chases as c4
import c8_chase_model as c8
import identity_tracks as it

CONFIDENT = 0.9
WINDOW_S = 10.0
BIN_S = 0.5
N_REF = 2000
SEED = 20261120
EDGES = np.arange(-WINDOW_S, WINDOW_S + BIN_S / 2, BIN_S)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def random_rates(recs: list[str], run: str, rng: np.random.Generator) -> dict[tuple[str, str], float]:
    """(recording, winner / loser) -> chirps per second at random moments with both fish in view."""
    out = {}
    for rec in recs:
        ch = pd.read_parquet(c4.CHIRPS / rec / "chirps_rtdetr_t080.parquet")
        det = pd.read_parquet(it.OUT / run / f"{rec}_detections.parquet", columns=["frame", "t_s", "identity"])
        both = det.groupby("frame").agg(t=("t_s", "first"), n=("identity", "nunique"))
        both = both.loc[both["n"] == 2, "t"].to_numpy()
        ref = rng.choice(both, min(N_REF, len(both)), replace=False) if len(both) else both
        for s in ("winner", "loser"):
            c = ch[ch["status"] == s]
            counts = c4.peri(ref, c["t_center_s"].to_numpy(), c["p_chirp"].to_numpy(), EDGES)
            out[(rec, s)] = float(counts.sum() / max(len(ref), 1) / (2 * WINDOW_S))
    return out


def ratio(ev: pd.DataFrame, col: str, who: str, base: dict) -> tuple[np.ndarray, int]:
    """Chirps of ``who`` around ``ev[col]``, relative to random moments in the same trials, per bin."""
    obs, exp = np.zeros(len(CENTERS)), 0.0
    for rec, g in ev.groupby("recording"):
        ch = pd.read_parquet(c4.CHIRPS / rec / "chirps_rtdetr_t080.parquet")
        c = ch[ch["status"] == who]
        obs += c4.peri(g[col].to_numpy(), c["t_center_s"].to_numpy(), c["p_chirp"].to_numpy(), EDGES)
        exp += len(g) * base[(rec, who)] * BIN_S
    return obs / exp if exp > 0 else np.full(len(CENTERS), np.nan), len(ev)


def winner_chases(source: str, run: str) -> pd.DataFrame:
    if source == "old detector":
        ev = pd.read_csv(it.IDENT / run / "events_identity.csv")
    else:
        ev = pd.read_csv(c8.OUT / source / "events.csv")
    return ev[ev["chaser"] == "winner"]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="+", help="model runs; the first one gets the confidence split")
    ap.add_argument("--old", action="store_true", help="also the old rule-based detector")
    ap.add_argument("--run", default="poses_r3")
    args = ap.parse_args()
    root = c8.OUT / args.names[0]
    sources = list(args.names) + (["old detector"] if args.old else [])
    evs = {s: winner_chases(s, args.run) for s in sources}
    recs = sorted(set().union(*(set(e["recording"]) for e in evs.values())))
    base = random_rates(recs, args.run, np.random.default_rng(SEED))
    first = evs[args.names[0]]
    split = {f"{args.names[0]}, confident (p >= {CONFIDENT})": first[first["p_mean"] >= CONFIDENT],
             f"{args.names[0]}, less confident (p < {CONFIDENT})": first[first["p_mean"] < CONFIDENT]}
    rows = []
    for label, ev in [*evs.items(), *split.items()]:
        for who in ("loser", "winner"):
            for align, col in (("onset", "t_start"), ("offset", "t_end")):
                r, n = ratio(ev, col, who, base)
                rows.append({"chases": label, "who": who, "align": align, "n": n, **dict(enumerate(r))})
    d = pd.DataFrame(rows)
    d.to_csv(root / "chirps_around_chases.csv", index=False)
    nb = len(CENTERS)

    def curve(label: str, who: str, align: str) -> tuple[np.ndarray, int]:
        r = d[(d["chases"] == label) & (d["who"] == who) & (d["align"] == align)].iloc[0]
        return r[list(range(nb))].to_numpy(float), int(r["n"])

    colors = ["#b30000", "#fc8d59", "0.45", "#7570b3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, align in zip(axes, ("onset", "offset"), strict=True):
        for k, s in enumerate(sources):
            y, n = curve(s, "loser", align)
            ax.step(CENTERS, y, where="mid", color=colors[k % len(colors)], lw=2 if k == 0 else 1.4,
                    label=f"{s} (n = {n})")
        ax.axhline(1, color="0.6", lw=0.8)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlabel(f"time from the chase {align} (s)")
        ax.set_title(f"aligned to chase {align}s", fontsize=9)
    axes[0].set_ylabel("loser's chirps, relative to random\nmoments in the same trials (1 = same)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Loser's chirps around chases in which the winner chases: which chase times line up best?",
                 fontsize=10)
    fig.savefig(root / "chirps_sources.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    for r, who in enumerate(("loser", "winner")):
        for col, align in enumerate(("onset", "offset")):
            ax = axes[r, col]
            for label, color in zip(split, ("#b30000", "#fdae6b"), strict=True):
                y, n = curve(label, who, align)
                ax.step(CENTERS, y, where="mid", color=color, lw=2, label=f"{label.split(', ')[1]} (n = {n})")
            ax.axhline(1, color="0.6", lw=0.8)
            ax.axvline(0, color="k", ls=":", lw=0.8)
            if r == 1:
                ax.set_xlabel(f"time from the chase {align} (s)")
            if col == 0:
                ax.set_ylabel(f"{who}'s chirps, relative to\nrandom moments (same trials)")
        axes[r, 0].legend(fontsize=7, frameon=False)
    axes[0, 0].set_title("aligned to chase onsets", fontsize=9)
    axes[0, 1].set_title("aligned to chase offsets", fontsize=9)
    fig.suptitle(f"Chirps around the chases of {args.names[0]} in which the winner chases, by the model's confidence",
                 fontsize=10)
    fig.savefig(root / "chirps_around_chases.png", dpi=150)
    plt.close(fig)

    for label in [*sources, *split]:
        for align in ("onset", "offset"):
            y, n = curve(label, "loser", align)
            pre, post = y[(CENTERS > -5) & (CENTERS < 0)].mean(), y[(CENTERS > 0) & (CENTERS < 5)].mean()
            jump = y[(CENTERS > 0) & (CENTERS < 1.5)].mean() - y[(CENTERS > -1.5) & (CENTERS < 0)].mean()
            print(f"{label:38s} {align:6s} n={n:5d}  loser chirps x random: 5 s before {pre:.2f}, "
                  f"5 s after {post:.2f}, step across 0 (1.5 s each side) {jump:+.2f}")
    print(f"-> {root}")


if __name__ == "__main__":
    main()
