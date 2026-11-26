"""One sim_v4 test tile as both arms' front ends render it (the PNGs the models see).

    uv run python experiments/ablation/tile_comparison.py A_DATASET B_DATASET OUT_PNG [--image ID]

Rows: arm A (sim_v4: 0.2048 / 0.0512 / 0.0128 s windows) and arm B
(sim_v4_longonly: 0.2048 s in all channels); columns: channels 0, 1, 2 and
|A - B| per channel. Ground-truth chirp boxes (green) and ignore regions
(grey, dashed) are drawn on channel 0. Without --image, picks the first test
tile (sorted id) holding both a small and a big visible chirp with footprint
SNR >= 0 dB and a neighbouring fish within 20 Hz of an emitter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from chirpdetector.data import load_dataset
from chirpdetector.data.contract import read_image


def pick(ds) -> str:
    b = ds.boxes
    test = set(ds.manifest.image_ids("test"))
    c = b[b["image_id"].isin(test) & (b["kind"] == "chirp") & (b["snr_db"] >= 0)]
    for image_id, g in sorted(c.groupby("image_id"), key=lambda kv: kv[0]):
        fam = set(g["family"])
        if {"small", "big"} <= fam and (g["min_df_live_hz"] < 20).any():
            return image_id
    return sorted(c["image_id"])[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--image", default=None)
    args = p.parse_args()
    a = load_dataset(args.a, check_files=False)
    b = load_dataset(args.b, check_files=False)
    image_id = args.image or pick(a)
    ia = read_image(a.image_path(image_id)).astype(np.int16)
    ib = read_image(b.image_path(image_id)).astype(np.int16)
    geo = a.manifest.image(image_id).geometry
    rec = a.manifest.image(image_id)
    t0 = rec.extra.get("frame_start", 0)
    boxes = a.boxes[a.boxes["image_id"] == image_id]
    wa = [a.manifest.spectrogram.reference_window_s, *a.manifest.spectrogram.extra_windows_s]
    extent = (
        geo.t_first_s - 0.5 * geo.dt_s,
        geo.t_first_s + (geo.width_px - 0.5) * geo.dt_s,
        geo.f_lowest_hz - 0.5 * geo.df_hz,
        geo.f_lowest_hz + (geo.height_px - 0.5) * geo.df_hz,
    )
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.6), sharex=True, sharey=True, constrained_layout=True)
    rows = [
        ("A (released): ", ia, [f"{w * 1000:.1f} ms window" for w in wa]),
        ("B (long only): ", ib, ["204.8 ms window"] * 3),
    ]
    for r, (name, img, titles) in enumerate(rows):
        for ch in range(3):
            ax = axes[r, ch]
            # image row 0 is the highest frequency (dataset contract); "lower" flipped the band
            ax.imshow(img[..., ch], origin="upper", aspect="auto", cmap="magma", vmin=0, vmax=255, extent=extent, interpolation="nearest")
            ax.set_title(f"{name}channel {ch}, {titles[ch]}", fontsize=10, loc="left")
    diff_max = []
    for ch in range(3):
        d = np.abs(ia[..., ch] - ib[..., ch])
        diff_max.append(int(d.max()))
        ax = axes[2, ch]
        ax.imshow(d, origin="upper", aspect="auto", cmap="gray_r", vmin=0, vmax=255, extent=extent, interpolation="nearest")
        ax.set_title(f"|A - B|, channel {ch}: max {d.max()}, mean {d.mean():.2f} (of 255)", fontsize=10, loc="left")
    for ax in axes[:2, 0]:
        for bx in boxes.itertuples(index=False):
            chirp = bx.kind == "chirp"
            ax.add_patch(
                Rectangle(
                    (bx.t_start_s, bx.f_low_hz), bx.t_end_s - bx.t_start_s, bx.f_high_hz - bx.f_low_hz,
                    fill=False, lw=1.2, ec="#1baf7a" if chirp else "#c3c2b7", ls="-" if chirp else "--",
                )
            )
    for ax in axes[-1]:
        ax.set_xlabel("time in scene (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (Hz)")
    fig.suptitle(
        f"{image_id} (sim_v4 test): the same tile in both front ends; "
        f"green = chirp boxes, dashed grey = ignore regions (identical in both datasets)",
        fontsize=11,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    info = {
        "image_id": image_id,
        "abs_diff_max_per_channel": diff_max,
        "b_channels_equal": bool(np.array_equal(ib[..., 0], ib[..., 1]) and np.array_equal(ib[..., 0], ib[..., 2])),
        "n_chirp_boxes": int((boxes["kind"] == "chirp").sum()),
        "n_ignore": int((boxes["kind"] == "ignore").sum()),
        "chirps": boxes[boxes["kind"] == "chirp"][["family", "excursion_hz", "duration_s", "snr_db", "min_df_live_hz"]].round(3).to_dict(orient="records"),
    }
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
