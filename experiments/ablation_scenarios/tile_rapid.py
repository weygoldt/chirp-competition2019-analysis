"""What the two front ends show the network during rapid chirping.

    uv run python experiments/ablation_scenarios/tile_rapid.py [--image ID] [--arm-b Bc16]

One benchmark tile with a fast chirp train, as the PNGs the models receive:
top, the single-window input (arm B: the 0.2048-s window in all three
channels, i.e. a grey image) and the RGB input (arm A: 0.2048 / 0.0512 /
0.0128 s in R / G / B), each with its own model's detections (score >= 0.5,
after post-processing; white boxes) and the true chirp centres (grey
triangles); bottom, arm A's three channels one by one. Cropped to the train
and the emitter's band. Default tile: the train with the largest difference
between arm A and the budget-matched arm B (Bc16) among 203 trains of >= 6
chirps in the 85-120 ms levels (median difference over those trains: 0).
Writes $ROOT/tile_rapid.png and .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # bench_figure (style)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bench_figure import CM, style  # noqa: E402
from paths import DATA  # noqa: E402

from chirpdetector.data import load_dataset  # noqa: E402
from chirpdetector.data.contract import read_image  # noqa: E402

ROOT = DATA / "runs/ablation_frontend/scenarios"
TILE = "ablation_bench_v1_rate_0100ms-030619_f0000440"


def kept(arm: str, image_id: str) -> pd.DataFrame:
    f = pd.read_parquet(ROOT / f"bench/{arm}/predictions_flagged.parquet")
    f = f[(f["recording"] == image_id) & (f["score"] >= 0.5) & (f["drop_reason"].astype(str) == "")]
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=TILE)
    p.add_argument("--arm-b", default="Bc16")
    p.add_argument("--labels", type=Path, default=ROOT / "bench/analysis_A_Bc16/labels.parquet")
    args = p.parse_args()
    style()
    da = load_dataset(DATA / "datasets/ablation_bench_v1_A", check_files=False)
    db = load_dataset(DATA / "datasets/ablation_bench_v1_B", check_files=False)
    ia = read_image(da.image_path(args.image))
    ib = read_image(db.image_path(args.image))
    assert np.array_equal(ib[..., 0], ib[..., 1]) and np.array_equal(ib[..., 0], ib[..., 2])
    assert np.array_equal(ia[..., 0], ib[..., 0]), "channel 0 must be the same window in both"
    geo = da.manifest.image(args.image).geometry
    extent = (
        geo.t_first_s - 0.5 * geo.dt_s,
        geo.t_first_s + (geo.width_px - 0.5) * geo.dt_s,
        geo.f_lowest_hz - 0.5 * geo.df_hz,
        geo.f_lowest_hz + (geo.height_px - 0.5) * geo.df_hz,
    )
    lab = pd.read_parquet(args.labels)
    lab = lab[lab["recording"] == args.image]
    em = lab["emitter_id"].value_counts().idxmax()
    train = lab[lab["emitter_id"] == em].sort_values("t_center_s")
    t_lo = max(train["t_start_s"].min() - 0.15, extent[0])
    t_hi = min(train["t_end_s"].max() + 0.15, extent[1])
    f_lo, f_hi = train["f_low_hz"].min() - 25, train["f_high_hz"].max() + 25
    preds = {"A": kept("A", args.image), args.arm_b: kept(args.arm_b, args.image)}
    found = {"A": int(train["found_A_after_025"].sum()), args.arm_b: int(train[f"found_{args.arm_b}_after_025"].sum())}

    fig = plt.figure(figsize=(17.4 * CM, 11.0 * CM), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])
    ax_t = top.subplots(1, 2, sharey=True)
    ax_b = bottom.subplots(1, 3, sharey=True)

    def show(ax, img, rgb: bool):
        # image row 0 is the highest frequency (dataset contract)
        ax.imshow(img if rgb else img[..., 0], origin="upper", aspect="auto", extent=extent,
                  interpolation="nearest", cmap=None if rgb else "gray", vmin=0, vmax=255)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(f_lo, f_hi)

    def mark(ax, arm):
        # true chirps: grey triangles at the top; the model's detections
        # (box midpoints): white triangles at the bottom
        span = f_hi - f_lo
        ax.plot(train["t_center_s"], np.full(len(train), f_hi - 0.04 * span), "v", color="0.65", ms=3.5, mec="none")
        d = preds[arm]
        mid = 0.5 * (d["t_start_s"] + d["t_end_s"])
        mid = mid[(mid >= t_lo) & (mid <= t_hi)]
        ax.plot(mid, np.full(len(mid), f_lo + 0.04 * span), "^", color="white", ms=3.5, mec="none")

    show(ax_t[0], ib, rgb=False)
    mark(ax_t[0], args.arm_b)
    ax_t[0].set_title(f"a  Single window: model B finds {found[args.arm_b]} of {len(train)}")
    show(ax_t[1], ia, rgb=True)
    mark(ax_t[1], "A")
    ax_t[1].set_title(f"b  RGB, three windows: model A finds {found['A']} of {len(train)}")
    ax_t[0].set_ylabel("frequency (Hz)")
    windows = ["0.2048 s (R)", "0.0512 s (G)", "0.0128 s (B)"]
    for ch, ax in enumerate(ax_b):
        ax.imshow(ia[..., ch], origin="upper", aspect="auto", extent=extent, interpolation="nearest",
                  cmap="gray", vmin=0, vmax=255)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(f_lo, f_hi)
        ax.set_title(f"{'cde'[ch]}  RGB channel {ch}: {windows[ch]}")
        ax.set_xlabel("time (s)")
    ax_b[0].set_ylabel("frequency (Hz)")
    dts = np.diff(train["t_center_s"].to_numpy())
    top.suptitle(
        f"One fish chirping every {1000 * np.median(dts):.0f} ms ({len(train)} small chirps, SNR "
        f"{train['snr_db'].median():.0f} dB). Grey: true chirps; white: detections",
        fontsize=8, x=0.01, ha="left",
    )
    for ax in ax_t:
        ax.set_xlabel("time (s)")
    out = ROOT / "tile_rapid.png"
    fig.savefig(out, dpi=300)
    info = {
        "image_id": args.image, "emitter_id": int(em), "n_chirps": len(train), "found": found,
        "interval_ms_median": float(1000 * np.median(dts)), "snr_db_median": float(train["snr_db"].median()),
        "emitter_df_hz": float(train["emitter_df_hz"].median()), "crop": [t_lo, t_hi, f_lo, f_hi],
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
