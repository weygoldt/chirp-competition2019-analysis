"""Does the benchmark's colour rule keep faint real chirps visible?

The blind benchmark's labeller colours every panel from the 5th percentile
of its dB values to its maximum (chirpdetector.labeling.view._panel). On
real recordings the maximum is a fish's baseline, so the colour range is the
baseline-to-noise range of the view, and a faint chirp gets only the part of
the colour map that its level above the noise is of that range.

This script renders the labeller's overview panel (the benchmark view, see
common.py) for

- every hand-labelled real chirp (the 365 labels of eval/real_labels_365:
  335 audian labels + 30 confirmed boxes, 4 recordings), and
- noise-only windows: random times inside the chirp-screened background
  segments of library v2 (inactive phase), with a pseudo candidate box on a
  fish's baseline (as a real candidate's view) or on its 2nd harmonic,
- a sample of the simulated benchmark's items (for the range comparison),

and records, per item, the dB statistics the two rules depend on: the
benchmark's colour limits, the percentiles of the background pixels (away
from every live track and its 2nd/3rd harmonic), and the chirp's level (99th
percentile inside its label box padded by 0.1 s, away from the tracks).
Colour positions for any rule follow from these (``summarise``).

    uv run python experiments/real_labels/contrast_check.py measure
    uv run python experiments/real_labels/contrast_check.py summarise
    uv run python experiments/real_labels/contrast_check.py sheets

Outputs: ``$CHIRP_DATA/labels/real_v1/review/contrast/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BACKGROUND,
    DATA,
    REAL_LABELS,
    ROOT,
    box_mask,
    colour_index,
    dataset_of,
    recording_dir,
    track_mask,
    tracks_dir,
    view_config,
)

OUT = ROOT / "review" / "contrast"
SEED = 20261114
N_NOISE = 120
N_SIM = 60


def _item(rec: str, k: int, box: tuple[float, float, float, float]):  # noqa: ANN202
    from chirpdetector.labeling.queue import QueueItem

    t0, t1, f0, f1 = (float(v) for v in box)
    return QueueItem(
        item_id=f"{rec}#c{k}",
        trial=rec,
        det_index=k,
        t_start_s=t0,
        t_end_s=t1,
        f_low_hz=f0,
        f_high_hz=f1,
        n_candidates=0,
        delta_f_hz=None,
        df_bin="",
    )


def _stats(view, box, other_boxes) -> dict:  # noqa: ANN001
    """dB statistics of the overview panel of one rendered view."""
    p = view.overview
    tm = track_mask(p, view.tracks)
    away = ~tm
    others = (
        box_mask(p, other_boxes, pad_s=0.3, pad_hz=20.0)
        if len(other_boxes)
        else np.zeros(p.db.shape, bool)
    )
    noise = away & ~others
    if box is not None:
        noise &= ~box_mask(p, [box], pad_s=0.3, pad_hz=20.0)
    q = np.percentile(p.db[noise], [5, 10, 50, 95, 99, 99.9]) if noise.any() else [np.nan] * 6
    row = {
        "bench_lo": p.clim[0],
        "bench_hi": p.clim[1],
        "bench_range_db": p.clim[1] - p.clim[0],
        "away_p10": float(np.percentile(p.db[away], 10)) if away.any() else np.nan,
        "away_p5": float(np.percentile(p.db[away], 5)) if away.any() else np.nan,
        "noise_p5": q[0],
        "noise_p10": q[1],
        "noise_p50": q[2],
        "noise_p95": q[3],
        "noise_p99": q[4],
        "noise_p999": q[5],
        "n_noise_px": int(noise.sum()),
        "n_tracks": len(view.tracks),
        "extent": json.dumps([float(v) for v in p.extent]),
    }
    if box is not None:
        region = box_mask(p, [box], pad_s=0.1) & away
        if not region.any():
            region = box_mask(p, [box], pad_s=0.1)
        row["chirp_p99"] = float(np.percentile(p.db[region], 99))
        row["chirp_max"] = float(p.db[region].max())
        row["snr_db"] = row["chirp_p99"] - row["noise_p50"]
    return row


def measure_real(cfg) -> pd.DataFrame:  # noqa: ANN001
    from chirpdetector.labeling.view import render_view
    from gridtools.io import load_tracks, open_recording

    rows = []
    for f in sorted(REAL_LABELS.glob("*/*/traces-grid1-editable-labels.csv")):
        rec = f.parent.name
        ds = dataset_of(rec)
        lab = pd.read_csv(f)
        chirps = lab[lab["category"] == "chirp"].reset_index(drop=True)
        boxes = chirps[["t_start_s", "t_end_s", "f_low_hz", "f_high_hz"]].to_numpy(float)
        recording = open_recording(recording_dir(ds, rec))
        tracks = load_tracks(tracks_dir(ds, rec))
        for k, box in enumerate(boxes):
            item = _item(rec, k, box)
            view = render_view(recording, tracks, item, cfg)
            others = np.delete(boxes, k, axis=0)
            row = {"kind": "real_chirp", "recording": rec, "k": k, "box": json.dumps(box.tolist())}
            row |= _stats(view, box, others)
            rows.append(row)
        print(f"{rec}: {len(boxes)} labelled chirps", flush=True)
    return pd.DataFrame(rows)


def measure_noise(cfg, rng: np.random.Generator) -> pd.DataFrame:  # noqa: ANN001
    from chirpdetector.labeling.view import render_view
    from gridtools.io import load_tracks, open_recording

    m = pd.read_parquet(BACKGROUND / "manifest.parquet")
    m = m[m["dataset"].str.startswith("competition") & m["accept"].astype(bool)]
    pick = m.iloc[rng.choice(len(m), size=N_NOISE, replace=False)]
    rows = []
    cache: dict[str, tuple] = {}
    for n, seg in enumerate(pick.itertuples()):
        rec, ds = seg.recording, seg.dataset
        if rec not in cache:
            cache[rec] = (open_recording(recording_dir(ds, rec)), load_tracks(tracks_dir(ds, rec)))
        recording, tracks = cache[rec]
        tc = float(rng.uniform(seg.start_s + 3.0, seg.stop_s - 3.0))
        live = sorted(tracks.frequencies_at(tc, 10.0).values())
        if not live:
            continue
        harmonic = n % 4 == 3  # a quarter on a 2nd harmonic (no fundamental near)
        f = float(live[rng.integers(len(live))]) * (2.0 if harmonic else 1.0)
        box = (tc - 0.02, tc + 0.02, f - 10.0, f + 120.0)
        view = render_view(recording, tracks, _item(rec, n, box), cfg)
        row = {
            "kind": "noise_harmonic" if harmonic else "noise",
            "recording": rec,
            "k": n,
            "box": json.dumps(list(box)),
            "segment_id": seg.segment_id,
        }
        row |= _stats(view, None, [])
        rows.append(row)
    print(f"noise: {len(rows)} windows", flush=True)
    return pd.DataFrame(rows)


def measure_sim(cfg, rng: np.random.Generator) -> pd.DataFrame:  # noqa: ANN001
    """Items of the simulated benchmark (true chirps and negatives alike)."""
    from chirpdetector.labeling.view import render_view
    from gridtools.io import load_tracks, open_recording

    cand = pd.read_parquet(DATA / "labels" / "candidates" / "human_benchmark_v2_extra.parquet")
    pick = cand.iloc[rng.choice(len(cand), size=N_SIM, replace=False)]
    rows = []
    for n, c in enumerate(pick.itertuples()):
        rec_dir = Path(c.recording)
        box = (c.t_start_s, c.t_end_s, c.f_low_hz, c.f_high_hz)
        view = render_view(open_recording(rec_dir), load_tracks(rec_dir), _item(rec_dir.name, n, box), cfg)
        row = {"kind": "sim_item", "recording": rec_dir.name, "k": n, "box": json.dumps(list(box))}
        row |= _stats(view, None, [])
        rows.append(row)
    return pd.DataFrame(rows)


def measure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = view_config()
    rng = np.random.default_rng(SEED)
    real = measure_real(cfg)
    noise = measure_noise(cfg, rng)
    sim = measure_sim(cfg, rng)
    pd.concat([real, noise, sim], ignore_index=True).to_csv(OUT / "measurements.csv", index=False)
    print(f"wrote {OUT / 'measurements.csv'}")


RULES = {
    "benchmark (p5..max)": None,
    "background p10 + 30 dB": (10.0, 30.0),
    "background p10 + 40 dB": (10.0, 40.0),
}


def _clim(row: pd.Series, rule) -> tuple[float, float]:  # noqa: ANN001
    if rule is None:
        return row["bench_lo"], row["bench_hi"]
    pct, span = rule
    floor = row["away_p10"] if pct == 10.0 else row["away_p5"]
    return floor, floor + span


def summarise() -> None:
    d = pd.read_csv(OUT / "measurements.csv")
    real = d[d["kind"] == "real_chirp"].copy()
    noise = d[d["kind"].str.startswith("noise")].copy()
    lines = ["# Contrast check: benchmark colour rule on real data", ""]
    lines += [
        "Overview panel of the benchmark view (5 s x 600 Hz, 204.8 ms window).",
        "dB values: electrode-summed power. 'noise' = pixels away from every live",
        "track (+-12 Hz, harmonics 1-3) and from every labelled chirp (+-0.3 s).",
        "Colour position c in [0, 1] (magma; c = 0 black).",
        "",
        "## dB ranges",
        "",
        "| items | n | benchmark range (max - p5), median [p10, p90] dB | noise p95 - p50 dB |",
        "|---|---|---|---|",
    ]
    for kind, g in d.groupby("kind"):
        r = g["bench_range_db"]
        lines.append(
            f"| {kind} | {len(g)} | {r.median():.1f} [{r.quantile(0.1):.1f}, {r.quantile(0.9):.1f}] "
            f"| {(g['noise_p95'] - g['noise_p50']).median():.1f} |"
        )
    real["snr_q"] = pd.qcut(real["snr_db"], 4, labels=["Q1 (faintest)", "Q2", "Q3", "Q4"])
    lines += ["", "Real chirps: level above the noise median (chirp p99 - noise p50):", ""]
    lines.append(
        f"median {real['snr_db'].median():.1f} dB; quartile edges "
        + ", ".join(f"{v:.1f}" for v in real["snr_db"].quantile([0.25, 0.5, 0.75]))
        + f" dB; min {real['snr_db'].min():.1f} dB"
    )
    lines += [
        "",
        "## Colour positions of real chirps by rule",
        "",
        "margin = c(chirp p99) - c(noise p99) of the same view: how far the chirp's",
        "brightest pixels stand above the brightest 1 % of its noise.",
        "",
        "| rule | SNR quartile | c(noise p50) | c(noise p99) | c(chirp p99) | margin median | margin < 0.10 |",
        "|---|---|---|---|---|---|---|",
    ]
    summary = {}
    for name, rule in RULES.items():
        for q, g in real.groupby("snr_q", observed=True):
            cn50 = np.array([colour_index(r["noise_p50"], _clim(r, rule)) for _, r in g.iterrows()])
            cn99 = np.array([colour_index(r["noise_p99"], _clim(r, rule)) for _, r in g.iterrows()])
            cc = np.array([colour_index(r["chirp_p99"], _clim(r, rule)) for _, r in g.iterrows()])
            margin = cc - cn99
            summary[(name, str(q))] = float(np.median(margin))
            lines.append(
                f"| {name} | {q} | {np.median(cn50):.2f} | {np.median(cn99):.2f} | "
                f"{np.median(cc):.2f} | {np.median(margin):.2f} | {np.mean(margin < 0.1):.0%} |"
            )
    lines += [
        "",
        "## Noise-only windows (background library v2, inactive phase, chirp-screened)",
        "",
        "c(noise p99.9): the brightest 0.1 % of background pixels; a chirp-like blob",
        "needs a few hundred pixels, so values near the chirps' c would mean noise",
        "can look like chirps.",
        "",
        "| rule | windows | c(noise p50) | c(noise p99) | c(noise p99.9) median [max] |",
        "|---|---|---|---|---|",
    ]
    for name, rule in RULES.items():
        for kind, g in noise.groupby("kind"):
            c50 = np.array([colour_index(r["noise_p50"], _clim(r, rule)) for _, r in g.iterrows()])
            c99 = np.array([colour_index(r["noise_p99"], _clim(r, rule)) for _, r in g.iterrows()])
            c999 = np.array([colour_index(r["noise_p999"], _clim(r, rule)) for _, r in g.iterrows()])
            lines.append(
                f"| {name} | {kind} ({len(g)}) | {np.median(c50):.2f} | {np.median(c99):.2f} | "
                f"{np.median(c999):.2f} [{c999.max():.2f}] |"
            )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def sheets(n_faint: int = 12, n_noise: int = 8) -> None:
    """Side-by-side contact sheets: benchmark rule | background p10 + 30 dB."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chirpdetector.labeling.view import hint_box, render_view
    from gridtools.io import load_tracks, open_recording

    d = pd.read_csv(OUT / "measurements.csv")
    cfg = view_config()
    real = d[d["kind"] == "real_chirp"].sort_values("snr_db").head(n_faint)
    rng = np.random.default_rng(SEED + 1)
    noise = d[d["kind"].str.startswith("noise")]
    noise = noise.iloc[rng.choice(len(noise), size=n_noise, replace=False)]
    for name, rows in (("faint_real_chirps", real), ("noise_windows", noise)):
        fig, axes = plt.subplots(len(rows), 2, figsize=(16, 2.6 * len(rows)), squeeze=False)
        for i, (_, r) in enumerate(rows.iterrows()):
            rec = r["recording"]
            ds = dataset_of(rec)
            box = json.loads(r["box"])
            item = _item(rec, int(r["k"]), box)
            view = render_view(
                open_recording(recording_dir(ds, rec)), load_tracks(tracks_dir(ds, rec)), item, cfg
            )
            p = view.overview
            away = ~track_mask(p, view.tracks)
            clims = {
                "benchmark: p5 .. max": p.clim,
                "background-anchored: p10 + 30 dB": (
                    float(np.percentile(p.db[away], 10)),
                    float(np.percentile(p.db[away], 10)) + 30.0,
                ),
            }
            for j, (title, clim) in enumerate(clims.items()):
                ax = axes[i, j]
                t0, t1, f0, f1 = p.extent
                ax.imshow(p.db, origin="lower", aspect="auto", extent=p.extent,
                          cmap="magma", vmin=clim[0], vmax=clim[1], interpolation="nearest")
                h = hint_box(item, cfg)
                ax.add_patch(plt.Rectangle((h[0], h[2]), h[1] - h[0], h[3] - h[2], fill=False,
                                           ec=(1, 1, 1, 0.45), ls=(0, (6, 5)), lw=1))
                ax.set_xlim(t0, t1)
                ax.set_ylim(f0, f1)
                extra = (
                    f"SNR {r['snr_db']:.1f} dB" if r["kind"] == "real_chirp" else r["kind"]
                )
                ax.set_title(
                    f"{title}  [{clim[0]:.0f}, {clim[1]:.0f}] dB   {rec} t={item.t_center_s:.2f} s   {extra}",
                    fontsize=8,
                )
                ax.tick_params(labelsize=7)
        fig.tight_layout()
        path = OUT / f"sheet_{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    {"measure": measure, "summarise": summarise, "sheets": sheets}[sys.argv[1]]()
