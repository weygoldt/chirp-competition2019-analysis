"""Gallery of real harmonic pairs and re-attributed rows (<= 20 panels).

Panels are drawn with the pipeline's `render_window` (the production
renderer: spectrogram recomputed with the detector's front end and click
removal) from tables post-processed with the production rules at the
panel's threshold (analyse.py `at`). Red dotted = flagged (here: the
dropped copy), solid = chirp; a re-attributed row is drawn at the
harmonic with "2nd harmonic of id k".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA, REC_2019, REC_2022  # noqa: E402


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gridtools.io import open_recording
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from analyse import (  # noqa: E402
    DRY_RECS,
    OUT,
    PROD,
    RUNS,
    THRESHOLDS,
    TRACK_DIRS,
    at,
    load_dry,
    load_labelled,
)

from chirpdetector.bundle import load_bundle  # noqa: E402
from chirpdetector.pipeline.windows import render_window  # noqa: E402

BUNDLE = f"{DATA}/runs/{RUNS['dual-head']}/final/bundle"
RAW = {
    **DRY_RECS,
    "2019-12-09-10_00": str(REC_2019 / "2019-12-09-10_00"),
    "2020-06-03-10_10": str(REC_2019 / "2020-06-03-10_10"),
    "2022-05-12-10_00": str(REC_2022 / "raw/2022-05-12-10_00"),
    "2022-06-01-10_00": str(REC_2022 / "raw/2022-06-01-10_00"),
}
HALF = 1.5


def render_table(flagged: pd.DataFrame, t: float) -> pd.DataFrame:
    """The columns `render_window` reads, from a flagged table."""
    again = flagged["reattributed"].astype(str).to_numpy() == "harmonic"
    reason = flagged["drop_reason"].astype(str).to_numpy()
    track = flagged["emitter_id"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    track[again] = flagged["harmonic_of_id"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)[again]
    out = pd.DataFrame(
        {
            "t_start_s": flagged["t_start_s"],
            "t_end_s": flagged["t_end_s"],
            "f_low_hz": flagged["f_low_hz"],
            "f_high_hz": flagged["f_high_hz"],
            "score": flagged["score"],
            "emitter_hz": flagged["emitter_hz"],
            "track_id": track,
            "operating": flagged["score"].to_numpy() >= t,
            "kept": reason == "",
            "pp_drop_reason": reason,
            "pp_reattributed": np.where(again, "harmonic", ""),
            "pp_harmonic_f_low_hz": flagged["harmonic_f_low_hz"],
            "pp_harmonic_f_high_hz": flagged["harmonic_f_high_hz"],
            "pp_harmonic_emitter_hz": flagged["harmonic_emitter_hz"],
            "assign_status": np.where(np.isfinite(track), "assigned", "unassigned"),
            "assign_confidence": np.nan,
        }
    )
    out["chirp"] = out["operating"] & out["kept"]
    return out


def pick(pairs: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Spread over copy-vs-fundamental score difference, recordings mixed."""
    p = pairs[pairs["partner_row"].notna()].copy()
    if len(p) <= n:
        return p
    p["diff"] = p["copy_score"] - p["fund_score"]
    p = p.sort_values("diff")
    idx = np.unique(np.linspace(0, len(p) - 1, n).round().astype(int))
    return p.iloc[idx]


def main() -> None:
    gallery = OUT / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(BUNDLE)
    panels = []  # (data, recording, t_center, threshold, note)
    dry, dtracks, _ = load_dry()
    lab, _, ltracks = load_labelled("dual-head")
    sources = {"dry": (dry, dtracks), "labelled": (lab, ltracks)}
    # re-attributed rows in real data at any threshold
    for key, (preds, tracks) in sources.items():
        for t in THRESHOLDS:
            f = at(preds, t, tracks, PROD)
            ra = f[f["reattributed"].astype(str) == "harmonic"]
            for r in ra.head(2).itertuples(index=False):
                panels.append((key, r.recording, 0.5 * (r.t_start_s + r.t_end_s), t, f"re-attributed at {t}"))
    # harmonic pairs at 0.5
    for key, n in (("dry", 10), ("labelled", 6)):
        name = "dryrun_dual-head" if key == "dry" else "labelled_dual-head"
        pairs = pd.read_csv(OUT / "tables" / f"real_pairs_{name}.csv")
        if key == "dry":  # both recordings
            other = pairs[pairs["recording"] == "2020-05-18-10_00"]
            chosen = pd.concat([pick(pairs[pairs["recording"] != "2020-05-18-10_00"], n - min(2, len(other))), other.head(2)])
        else:
            chosen = pick(pairs, n)
        for r in chosen.itertuples(index=False):
            note = "copy dropped (fundamental detected)" if pd.notna(r.partner_row) else "copy dropped"
            panels.append((key, r.recording, r.t_center_s, 0.5, f"{note}; copy {r.copy_score:.2f} vs fundamental {r.fund_score:.2f}"))
    panels = panels[:20]
    flagged_cache: dict = {}
    files = []
    for k, (key, rec, tc, t, note) in enumerate(panels):
        preds, tracks = sources[key]
        if (key, t) not in flagged_cache:
            flagged_cache[(key, t)] = at(preds, t, tracks, PROD)
        f = flagged_cache[(key, t)]
        table = render_table(f[f["recording"] == rec], t)
        stem = gallery / f"{k:02d}_{rec}_{tc:.1f}"
        render_window(
            stem,
            open_recording(RAW[rec]),
            bundle,
            table,
            tracks[rec],
            tc - HALF,
            tc + HALF,
            title=f"{rec} {tc:.1f} s, threshold {t}: {note}.",
            device="cpu",
        )
        files.append(stem.with_suffix(".png"))
        print(stem.name, note)
    # contact sheet, two columns
    images = [Image.open(p) for p in files]
    w = 1100
    scaled = [im.resize((w, int(im.height * w / im.width))) for im in images]
    h = max(im.height for im in scaled)
    rows = (len(scaled) + 1) // 2
    sheet = Image.new("RGB", (2 * w, rows * h), "white")
    for i, im in enumerate(scaled):
        sheet.paste(im, ((i % 2) * w, (i // 2) * h))
    sheet.save(OUT / "gallery_contact_sheet.png")
    pd.DataFrame(panels, columns=["data", "recording", "t_center_s", "threshold", "note"]).to_csv(gallery / "panels.csv", index=False)


if __name__ == "__main__":
    main()
