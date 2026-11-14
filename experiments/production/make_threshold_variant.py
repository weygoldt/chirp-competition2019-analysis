"""A detection set of a production run at a stricter operating threshold.

Robustness check decided by the author (2026-11-14): the analyses stay at the
production operating threshold 0.5 (precision 0.938 / recall 0.992 on the
365 real labels), and are repeated at 0.9 (0.981 / 0.827) to show whether the
conclusions depend on the less certain detections.

Only the operating threshold changes. Post-processing flags (harmonic and
duplicate copies) and the assignment do not depend on it and are taken
unchanged from the source run: a row counts as a chirp if ``score >=
threshold`` and it was kept by post-processing (``kept``; equivalently
``pp_drop_reason == ""``). Re-attribution of harmonic copies was evaluated at
the source run's threshold (0.5); such rows stay only if their score also
passes the new threshold.

For every recording of the source run this writes, into OUT/<recording>/:
``chirps_rtdetr<suffix>.parquet`` (the source table with ``operating`` and
``chirp`` recomputed), the legacy arrays ``chirp_times_rtdetr<suffix>.npy`` /
``chirp_ids_rtdetr<suffix>.npy`` (rows that are chirps and assigned, sorted by
box centre, as in the pipeline), and ``meta.json`` (the source meta with the
new threshold, counts, outputs and a ``variant`` record). Then copy them into
the recording directories with ``copy_detections_to_trials.py --run OUT
--suffix <suffix>``.

Check built in: ``--verify`` recomputes the source's own threshold and
requires the legacy arrays to equal the source's exactly.

    python experiments/production/make_threshold_variant.py \
        --run $CHIRP_DATA/detections/competition2019_v2_20261113_gapfill \
        --source-suffix _gapfill --threshold 0.9 --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA  # noqa: E402


def legacy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sel = df[df["chirp"] & df["track_id"].notna()].sort_values("t_center_s", kind="stable")
    return sel["t_center_s"].to_numpy(np.float64), sel["track_id"].to_numpy(np.float64)


def recompute(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["operating"] = out["score"] >= threshold
    out["chirp"] = out["operating"] & out["kept"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=DATA / "detections" / "competition2019_v2_20261113_gapfill")
    ap.add_argument("--source-suffix", default="_gapfill")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>_t<threshold*100>")
    ap.add_argument("--verify", action="store_true", help="check the source threshold reproduces the source arrays first")
    args = ap.parse_args()
    tag = f"_t{round(args.threshold * 100):03d}"
    suffix = args.source_suffix + tag
    out_root = args.out or args.run.with_name(args.run.name + tag)
    recs = sorted(p.name for p in args.run.iterdir() if (p / "meta.json").is_file())

    if args.verify:
        for rec in recs:
            src = args.run / rec
            meta = json.loads((src / "meta.json").read_text())
            df = pd.read_parquet(src / f"chirps_rtdetr{args.source_suffix}.parquet")
            t, ids = legacy(recompute(df, float(meta["operating_threshold"])))
            t0 = np.load(src / f"chirp_times_rtdetr{args.source_suffix}.npy")
            i0 = np.load(src / f"chirp_ids_rtdetr{args.source_suffix}.npy")
            if not (np.array_equal(t, t0) and np.array_equal(ids, i0)):
                raise SystemExit(f"verify failed for {rec}: recomputing the source threshold does not reproduce its legacy arrays")
        print(f"verify: source threshold reproduces the legacy arrays of all {len(recs)} recordings")

    total_src = total_new = 0
    for rec in recs:
        src = args.run / rec
        meta = json.loads((src / "meta.json").read_text())
        df = pd.read_parquet(src / f"chirps_rtdetr{args.source_suffix}.parquet")
        new = recompute(df, args.threshold)
        t, ids = legacy(new)
        dst = out_root / rec
        dst.mkdir(parents=True, exist_ok=True)
        new.to_parquet(dst / f"chirps_rtdetr{suffix}.parquet", index=False)
        np.save(dst / f"chirp_times_rtdetr{suffix}.npy", t)
        np.save(dst / f"chirp_ids_rtdetr{suffix}.npy", ids)
        n_src = int(meta["legacy"]["n"])
        meta["variant"] = {
            "kind": "operating threshold",
            "source_run": str(args.run),
            "source_files": f"chirps_rtdetr{args.source_suffix}.parquet",
            "source_operating_threshold": meta["operating_threshold"],
            "operating_threshold": args.threshold,
            "rule": "chirp = score >= threshold & kept (post-processing and assignment unchanged from the source)",
            "legacy_n_source": n_src,
            "created": datetime.now().isoformat(timespec="seconds"),
            "script": "experiments/production/make_threshold_variant.py",
        }
        meta["operating_threshold"] = args.threshold
        meta["run"] = out_root.name
        meta["legacy"]["n"] = len(t)
        meta["legacy"]["times"] = f"chirp_times_rtdetr{suffix}.npy"
        meta["legacy"]["ids"] = f"chirp_ids_rtdetr{suffix}.npy"
        meta.setdefault("counts", {})
        meta["counts"]["operating"] = int(new["operating"].sum())
        meta["counts"]["chirps"] = int(new["chirp"].sum())
        meta["counts"]["legacy"] = len(t)
        meta["outputs"] = {
            "chirps": {"file": f"chirps_rtdetr{suffix}.parquet"},
            "times": {"file": f"chirp_times_rtdetr{suffix}.npy"},
            "ids": {"file": f"chirp_ids_rtdetr{suffix}.npy"},
        }
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        total_src += n_src
        total_new += len(t)
    (out_root / "README.md").write_text(
        f"# {out_root.name}\n\nThe run `{args.run.name}` at operating threshold {args.threshold} "
        f"(source: {meta['variant']['source_operating_threshold']}); made by "
        "`experiments/production/make_threshold_variant.py`. Post-processing and assignment unchanged. "
        f"Legacy chirps over all {len(recs)} recordings: {total_src:,} -> {total_new:,}.\n"
    )
    print(f"{len(recs)} recordings -> {out_root} (suffix {suffix}); legacy chirps {total_src:,} -> {total_new:,}")


if __name__ == "__main__":
    main()
