"""How much does each labelled item move the 2022 agreement estimates of real_v1?

The real_v1 session weights each labelled candidate by ``w = N_cell /
n_labelled_cell`` (cell = stratum x recording) and reports, per score bin,
the share of detections the expert judged chirps ("agreement"). In 2022 the
detections are very unevenly spread over recordings, so a few items can
carry most of a bin's weight. This script shows, per dataset and score bin:

- the weighted agreement as reported (same formula as
  ``experiments/real_labels/score_session.py``);
- the share of the bin's weight carried by its heaviest item and by its
  heaviest recording;
- the agreement with each recording left out in turn (range over recordings);
- the estimated number of detections per recording in the bin (N_hat) and
  that recording's own labelled agreement.

    .venv/bin/python experiments/data2022/a1_influence.py

Output: ``$CHIRP_DATA/data2022/a1/influence/`` (``by_bin.csv``,
``by_recording.csv``, ``heaviest_items.csv``, ``influence.png``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA

SCORED = DATA / "labels" / "real_v1" / "report" / "candidates_scored.csv"
OUT = DATA / "data2022" / "a1" / "influence"
POSITIVE = ("chirp", "chirp_emitter_unclear")
BINS = ["0.01-0.05", "0.05-0.2", "0.2-0.5", "0.5-0.7", "0.7-0.9", "0.9-1"]
DATASETS = {"competition2019": "2019/2020", "competition2022": "2022"}


def load_main(path: Path = SCORED) -> pd.DataFrame:
    """Labelled main-stratum candidates with their design weight ``w`` and decision ``y``."""
    t = pd.read_csv(path)
    t = t[t["stratum"].str.startswith("main/") & t["labelled"]].copy()
    t["y"] = t["label"].isin(POSITIVE).astype(float)
    return t.reset_index(drop=True)


def agreement(t: pd.DataFrame) -> float:
    """Weighted share of chirps: sum(w y) / sum(w)."""
    w = t["w"].to_numpy()
    return float((w * t["y"].to_numpy()).sum() / w.sum()) if w.sum() > 0 else np.nan


def by_bin(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, name in DATASETS.items():
        for b in BINS:
            g = t[(t["dataset"] == ds) & (t["score_bin"] == b)]
            if g.empty:
                continue
            w_rec = g.groupby("recording")["w"].sum()
            loo = [agreement(g[g["recording"] != r]) for r in w_rec.index]
            rows.append({
                "dataset": name, "score_bin": b, "n_items": len(g), "n_recordings": g["recording"].nunique(),
                "N_hat": g["w"].sum(), "agreement": agreement(g),
                "agreement_unweighted": g["y"].mean(),
                "heaviest_item_weight_share": g["w"].max() / g["w"].sum(),
                "heaviest_recording": w_rec.idxmax(),
                "heaviest_recording_weight_share": w_rec.max() / w_rec.sum(),
                "leave_one_recording_out_min": np.nanmin(loo), "leave_one_recording_out_max": np.nanmax(loo),
                "agreement_without_heaviest_recording": agreement(g[g["recording"] != w_rec.idxmax()]),
            })
    return pd.DataFrame(rows)


def by_recording(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["dataset", "score_bin", "recording"])
    out = g.agg(n_items=("y", "size"), n_chirp=("y", "sum"), N_hat=("w", "sum"))
    out["agreement"] = g.apply(agreement, include_groups=False)
    out = out.reset_index()
    out["dataset"] = out["dataset"].map(DATASETS)
    return out


def heaviest_items(t: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    cols = ["dataset", "score_bin", "stratum", "recording", "candidate_id", "t_center_s", "score",
            "label", "w", "cell_N", "cell_n_labelled", "emitter_hz", "track_hz", "height_hz", "fwhm_s"]
    rows = []
    for _, g in t.groupby(["dataset", "score_bin"]):
        g = g.assign(weight_share=g["w"] / g["w"].sum())
        rows.append(g.sort_values("w", ascending=False).head(k)[[*cols, "weight_share"]])
    out = pd.concat(rows)
    out["dataset"] = out["dataset"].map(DATASETS)
    return out


def plot(t: pd.DataFrame, table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, (ds, name) in zip(axes, DATASETS.items(), strict=True):
        x = np.arange(len(BINS))
        rows = table[table["dataset"] == name].set_index("score_bin").reindex(BINS)
        for i, b in enumerate(BINS):
            g = t[(t["dataset"] == ds) & (t["score_bin"] == b)]
            if g.empty:
                continue
            # each labelled item: area ~ its share of the bin's weight
            share = g["w"] / g["w"].sum()
            jitter = np.random.default_rng(i).uniform(-0.28, 0.28, len(g))
            yy = np.where(g["y"] > 0, 1.0, 0.0) + np.random.default_rng(i + 99).uniform(-0.04, 0.04, len(g))
            ax.scatter(i + jitter, yy, s=8 + 900 * share, alpha=0.45,
                       c=np.where(g["y"] > 0, "#2b8cbe", "#d95f0e"), edgecolors="none")
        ax.plot(x, rows["agreement"], "k-o", lw=1.5, ms=4, label="weighted agreement (reported)")
        ax.plot(x, rows["agreement_unweighted"], color="0.5", ls="--", marker="s", ms=3, lw=1,
                label="unweighted share of labelled items")
        ax.vlines(x, rows["leave_one_recording_out_min"], rows["leave_one_recording_out_max"],
                  color="k", lw=3, alpha=0.25, label="range, one recording left out")
        ax.set_xticks(x, BINS)
        ax.set_xlabel("detector score bin")
        ax.set_title(f"{name}: each dot one labelled detection (area = its weight)", fontsize=10)
        ax.set_ylim(-0.12, 1.12)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("judged a chirp by the expert\n(1 = chirp, 0 = not; line = weighted share)")
    axes[1].legend(loc="center left", fontsize=8, frameon=False)
    fig.suptitle("In 2022, single heavy items from one recording decide the agreement of every bin above 0.5",
                 fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_main()
    table = by_bin(t)
    table.to_csv(OUT / "by_bin.csv", index=False)
    by_recording(t).to_csv(OUT / "by_recording.csv", index=False)
    heaviest_items(t).to_csv(OUT / "heaviest_items.csv", index=False)
    plot(t, table, OUT / "influence.png")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
