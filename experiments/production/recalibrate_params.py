"""A detection run with the chirp parameters of a recalibrated decoder bundle.

Quick fix decided by the author (2026-11-27): template library v4 holds three
octave-error templates (61, 1598, 391; the chirp-chamber zero-crossing trace
reads twice the baseline EODf during the chirp). Template 1598 is in the
decoder's validation split and set the top of the high-sigma height
calibration and the ``uncertain`` thresholds. The calibration was refitted
without it (``chirpdetector train params-recalibrate``; same weights), and
this script carries the new calibration into the stored detections: no
re-detection, no new decoder pass.

The decoder's calibrated columns are a function of its raw output (model-space
mean and sigma per parameter) and the bundle's calibration
(`chirpdetector.params.inference.ParamsEstimator.estimate_inputs`). The tables
store the raw mean (``*_raw_*``: ``exp(mean) - offset``) and the calibrated
value and sd (``sd = (value + offset) * calibrated sigma``), from which the raw
sigma follows (``sigma = calibrated sigma / sigma_scale * slope of the bias
curve at the calibrated mean``; the sigma stratum is the one whose bias curve
and edges reproduce the stored row). For every row the script then

1. applies the *old* bundle's calibration to the recovered raw output and
   requires every stored value, sd and flag back (relative difference below
   ``--rtol``, flags exactly); the run is refused otherwise;
2. applies the *new* bundle's calibration: new values, sds and ``uncertain``
   flags; ``edge``, ``band`` and ``size`` do not depend on the calibration and
   are kept, raw columns stay as they are.

Undershoot is floored at 0 Hz; where the raw undershoot is 0 the raw mean is
only known up to a bound (``mean <= h(calibrated mean)`` or ``<= 0``). Those
rows are evaluated at the bound; the new calibrated undershoot is exact if it
is 0 there (the inverse curve is monotone and floored). Otherwise it lies
between 0 and the value at the bound: such rows keep the stored 0 Hz (sd
recomputed at it) and are counted (``undershoot_undetermined``, with the
largest bound); the run is refused if a bound exceeds
``--max-undetermined-hz`` (0.2 Hz; the 2019 runs: 3 rows, bound 0.13 Hz, sd
0.3-0.4 Hz).

For every recording of the source run this writes, into OUT/<recording>/, the
table with the new parameter columns (same file name), the unchanged legacy
arrays and ``meta.json`` (the source meta with ``models.decoder`` set to the new
bundle and a ``params_recalibration`` record). Then copy them into the
recording directories with ``copy_detections_to_trials.py --run OUT --suffix
<suffix> --replace-from <source run>`` (the author decided on 2026-11-27 to
replace the trial copies; the originals stay in the source run).

    python experiments/production/recalibrate_params.py \
        --run $CHIRP_DATA/detections/competition2019_v2_20261113_gapfill_t080 \
        --suffix _gapfill_t080 \
        --new-bundle $CHIRP_DATA/params/runs/v4-dualhead-sim-v4-r2-recal1/bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from chirpdetector.params import load_params_bundle
from chirpdetector.params.bundle import ParamsBundle
from chirpdetector.params.calibration import MonotoneMap
from chirpdetector.params.inference import _RAW_NAMES, _SD_NAMES, FLAG_REASONS
from chirpdetector.params.model import LOG_OFFSETS, from_model_space, natural_sd
from chirpdetector.params.truth import PARAMETERS
from chirpdetector.training.provenance import code_state, git_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA

OFFSETS = np.array([LOG_OFFSETS[p] for p in PARAMETERS])
SD = [_SD_NAMES[p] for p in PARAMETERS]
RAW = [_RAW_NAMES[p] for p in PARAMETERS]
UNDERSHOOT = PARAMETERS.index("undershoot_hz")


def curves(
    bundle: ParamsBundle, name: str
) -> tuple[tuple[MonotoneMap, ...], np.ndarray]:
    """Bias curves of a parameter (one per sigma stratum) and the strata's inner edges."""
    cal = bundle.calibration
    if cal.stratum_maps.get(name):
        return tuple(cal.stratum_maps[name]), np.asarray(cal.sigma_edges[name], float)
    return (cal.maps.get(name, MonotoneMap()),), np.array([])


def recover_raw(
    df: pd.DataFrame, bundle: ParamsBundle
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw mean and sigma (model space, ``(n, 4)``) of the rows, and where the mean is a bound.

    ``bound[i, j]``: the raw value is floored at 0, the mean is the largest one
    consistent with the row (see module docstring).
    """
    value = df[list(PARAMETERS)].to_numpy(np.float64)
    sd = df[SD].to_numpy(np.float64)
    raw = df[RAW].to_numpy(np.float64)
    cal_sigma = sd / (value + OFFSETS)
    n = len(df)
    mean = np.full((n, len(PARAMETERS)), np.nan)
    sigma = np.full_like(mean, np.nan)
    bound = np.zeros_like(mean, dtype=bool)
    for j, name in enumerate(PARAMETERS):
        maps, edges = curves(bundle, name)
        scale = float(bundle.calibration.sigma_scale.get(name, 1.0))
        floored_cal = value[:, j] <= 0
        floored_raw = raw[:, j] <= 0
        if (floored_cal | floored_raw).any() and j != UNDERSHOOT:
            raise SystemExit(f"{name}: floored values, only undershoot can be floored")
        with np.errstate(divide="ignore"):
            cal_mean = np.where(floored_cal, -np.inf, np.log(value[:, j] + OFFSETS[j]))
            obs_mean = np.where(floored_raw, np.nan, np.log(raw[:, j] + OFFSETS[j]))
        chosen = np.full(n, -1)
        for k, h in enumerate(maps):
            if h.x and floored_cal.any() and h.x[0] <= 0:
                raise SystemExit(
                    f"{name}: a floored calibrated value lies inside a bias curve"
                )
            # slope 1 below the first knot (floored rows); right-hand slope at a knot otherwise
            slope = np.where(
                floored_cal, 1.0, h.slope(np.where(floored_cal, 0.0, cal_mean))
            )
            s_k = cal_sigma[:, j] / scale * slope
            in_stratum = (
                np.digitize(s_k, edges) == k if edges.size else np.ones(n, bool)
            )
            maps_back = np.where(
                floored_raw | floored_cal,
                True,
                np.isclose(
                    h.inverse(np.nan_to_num(obs_mean)), cal_mean, rtol=0, atol=1e-9
                ),
            )
            take = in_stratum & maps_back & (chosen < 0)
            chosen[take] = k
            sigma[take, j] = s_k[take]
            # the mean: observed, or from the calibrated value, or a bound
            m = np.where(floored_raw, np.nan, obs_mean)
            from_cal = floored_raw & ~floored_cal
            m = np.where(from_cal, h(np.where(from_cal, cal_mean, 0.0)), m)
            both = floored_raw & floored_cal
            m = np.where(both, np.minimum(0.0, h(np.zeros(n))), m)
            mean[take, j] = m[take]
            bound[take, j] = both[take]
        if (chosen < 0).any():
            raise SystemExit(
                f"{name}: {int((chosen < 0).sum())} rows match no sigma stratum"
            )
    return mean, sigma, bound


def estimate(
    mean: np.ndarray,
    sigma: np.ndarray,
    bundle: ParamsBundle,
    other_flags: list[list[str]],
) -> pd.DataFrame:
    """The calibrated columns, ``params_ok`` and ``params_flag``, as the estimator computes them."""
    cal_mean, cal_sigma = bundle.calibration.apply(mean, sigma)
    values = from_model_space(cal_mean)
    sds = natural_sd(values, cal_sigma)
    thresholds = bundle.domain.get("sigma_threshold", {})
    uncertain = np.zeros(len(mean), dtype=bool)
    for j, name in enumerate(PARAMETERS):
        if name in thresholds:
            uncertain |= ~(cal_sigma[:, j] <= float(thresholds[name]))
    flags = []
    for reasons, unsure in zip(other_flags, uncertain, strict=True):
        r = set(reasons) | ({"uncertain"} if unsure else set())
        flags.append(",".join(x for x in FLAG_REASONS if x in r))
    out = pd.DataFrame({p: values[:, j] for j, p in enumerate(PARAMETERS)})
    for j, name in enumerate(SD):
        out[name] = sds[:, j]
    out["params_ok"] = [not f for f in flags]
    out["params_flag"] = flags
    return out


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.maximum(np.abs(a), np.abs(b))
    diff = np.where(scale > 0, np.abs(a - b) / np.where(scale > 0, scale, 1.0), 0.0)
    return float(np.nanmax(diff)) if diff.size else 0.0


def recalibrate(
    df: pd.DataFrame,
    old: ParamsBundle,
    new: ParamsBundle,
    rtol: float,
    max_undetermined_hz: float,
) -> tuple[pd.DataFrame, dict]:
    has = df["height_hz"].notna().to_numpy()
    rows = df.loc[has]
    other = [
        [r for r in str(f).split(",") if r and r != "uncertain"]
        for f in rows["params_flag"]
    ]
    unknown = {r for rs in other for r in rs} - set(FLAG_REASONS)
    if unknown:
        raise SystemExit(f"unknown params_flag reasons {sorted(unknown)}")
    mean, sigma, bound = recover_raw(rows, old)

    # 1. the old calibration gives the stored table back
    again = estimate(mean, sigma, old, other)
    worst = 0.0
    for col in [*PARAMETERS, *SD]:
        stored = rows[col].to_numpy(np.float64)
        worst = max(worst, relative_difference(again[col].to_numpy(), stored))
    raw_again = from_model_space(np.where(np.isfinite(mean), mean, 0.0))
    worst_raw = relative_difference(raw_again, rows[RAW].to_numpy(np.float64))
    flags_equal = bool(
        (
            again["params_flag"].to_numpy()
            == rows["params_flag"].astype(str).to_numpy()
        ).all()
    )
    if worst > rtol or worst_raw > rtol or not flags_equal:
        raise SystemExit(
            f"the old calibration does not reproduce the stored columns: max rel. diff {worst:.2e} "
            f"(raw {worst_raw:.2e}), flags equal {flags_equal}"
        )

    # 2. the new calibration
    fresh = estimate(mean, sigma, new, other)
    upper = fresh["undershoot_hz"].to_numpy()
    undetermined = bound[:, UNDERSHOOT] & (upper > 0)
    largest = float(upper[undetermined].max()) if undetermined.any() else 0.0
    if largest > max_undetermined_hz:
        raise SystemExit(
            f"{int(undetermined.sum())} floored undershoots lie in [0, {largest:.3f}] Hz under the new "
            f"calibration, more than --max-undetermined-hz {max_undetermined_hz}"
        )
    if undetermined.any():
        # keep the stored 0 Hz (the lower end); sd = (value + offset) * calibrated sigma at it
        sd = fresh["undershoot_sd_hz"].to_numpy().copy()
        sd[undetermined] /= upper[undetermined] + OFFSETS[UNDERSHOOT]
        fresh.loc[undetermined, "undershoot_sd_hz"] = sd[undetermined]
        fresh.loc[undetermined, "undershoot_hz"] = 0.0
    out = df.copy()
    for col in [*PARAMETERS, *SD, "params_ok", "params_flag"]:
        values = fresh[col].to_numpy()
        if col == "params_ok":
            out[col] = out[col].astype(bool)
        out.loc[has, col] = values
    old_ok = rows["params_ok"].to_numpy(bool)
    new_ok = fresh["params_ok"].to_numpy(bool)
    ratio = fresh["height_hz"].to_numpy() / rows["height_hz"].to_numpy()
    summary = {
        "rows_with_params": int(has.sum()),
        "rows_without_params": int((~has).sum()),
        "verify_max_rel_diff": worst,
        "verify_max_rel_diff_raw": worst_raw,
        "verify_flags_equal": flags_equal,
        "undershoot_mean_bounded": int(bound[:, UNDERSHOOT].sum()),
        "undershoot_undetermined": int(undetermined.sum()),
        "undershoot_undetermined_max_hz": largest,
        "params_ok_before": int(old_ok.sum()),
        "params_ok_after": int(new_ok.sum()),
        "params_ok_lost": int((old_ok & ~new_ok).sum()),
        "params_ok_gained": int((~old_ok & new_ok).sum()),
        "height_ratio_median": float(np.median(ratio)),
        "height_ratio_min": float(np.min(ratio)),
        "height_ratio_max": float(np.max(ratio)),
        "height_changed_over_1pct": int((np.abs(ratio - 1) > 0.01).sum()),
    }
    return out, summary


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(bundle: ParamsBundle) -> dict:
    return {
        "path": str(bundle.path),
        "weights_sha256": bundle.weights_sha256,
        "bundle_json_sha256": sha256(bundle.path / "bundle.json"),
        "created_at": bundle.created_at,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="source detection run")
    ap.add_argument(
        "--suffix",
        default="",
        help="file suffix of the source run (e.g. _gapfill_t080)",
    )
    ap.add_argument(
        "--new-bundle",
        type=Path,
        default=DATA / "params" / "runs" / "v4-dualhead-sim-v4-r2-recal1" / "bundle",
    )
    ap.add_argument("--out", type=Path, default=None, help="default: <run>_recal1")
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="largest relative difference allowed in the check",
    )
    ap.add_argument(
        "--max-undetermined-hz",
        type=float,
        default=0.2,
        help="largest new undershoot allowed for a floored raw undershoot (kept at 0 Hz)",
    )
    args = ap.parse_args()
    out_root = args.out or args.run.with_name(args.run.name + "_recal1")
    if out_root.exists():
        raise SystemExit(f"{out_root} exists")
    new = load_params_bundle(args.new_bundle)
    recal = new.training.get("recalibration")
    if not recal:
        raise SystemExit(f"{args.new_bundle} is not a recalibrated bundle")
    old = load_params_bundle(recal["source_bundle"])
    if new.weights_sha256 != old.weights_sha256:
        raise SystemExit("the new bundle has other weights than its source")
    recs = sorted(p.name for p in args.run.iterdir() if (p / "meta.json").is_file())
    files = (
        f"chirps_rtdetr{args.suffix}.parquet",
        f"chirp_times_rtdetr{args.suffix}.npy",
        f"chirp_ids_rtdetr{args.suffix}.npy",
    )
    totals: dict[str, float] = {}
    staging = out_root.with_name(out_root.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    for rec in recs:
        src = args.run / rec
        meta = json.loads((src / "meta.json").read_text())
        decoder = meta.get("models", {}).get("decoder", {})
        if decoder.get("weights_sha256") not in (None, old.weights_sha256):
            raise SystemExit(
                f"{rec}: detected with decoder weights {decoder.get('weights_sha256')}, not the source bundle's"
            )
        df = pd.read_parquet(src / files[0])
        patched, summary = recalibrate(
            df, old, new, args.rtol, args.max_undetermined_hz
        )
        dst = staging / rec
        dst.mkdir(parents=True)
        patched.to_parquet(dst / files[0], index=False)
        for name in files[1:]:
            shutil.copyfile(src / name, dst / name)
        meta["params_recalibration"] = {
            "kind": "decoder calibration refitted without octave-error templates; columns recomputed from the stored raw output",
            "source_run": str(args.run),
            "source_files": {name: sha256(src / name) for name in files},
            "decoder_at_detection": decoder,
            "old_bundle": describe(old),
            "new_bundle": describe(new),
            "excluded_template_ids": recal["excluded_template_ids"],
            "reason": recal["reason"],
            "columns": [*PARAMETERS, *SD, "params_ok", "params_flag"],
            "unchanged": [
                *RAW,
                "edge/band/size reasons of params_flag",
                "all other columns",
            ],
            **summary,
            "created": datetime.now().isoformat(timespec="seconds"),
            "script": "experiments/production/recalibrate_params.py",
            "code": {
                **code_state(),
                "script": git_state(Path(__file__).resolve().parent),
            },
        }
        meta["models"]["decoder"] = {
            **decoder,
            **describe(new),
            "recalibrated_from": describe(old),
        }
        meta["run"] = out_root.name
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        for key, value in summary.items():
            if isinstance(value, bool) or key.startswith("height_ratio_median"):
                continue
            if key.startswith("verify") or key in (
                "height_ratio_max",
                "undershoot_undetermined_max_hz",
            ):
                totals[key] = max(totals.get(key, value), value)
            elif key == "height_ratio_min":
                totals[key] = min(totals.get(key, value), value)
            else:
                totals[key] = totals.get(key, 0) + value
        print(
            f"{rec}: ok {summary['params_ok_before']} -> {summary['params_ok_after']}, "
            f"height changed >1% {summary['height_changed_over_1pct']}, check {summary['verify_max_rel_diff']:.1e}"
        )
    (staging / "README.md").write_text(
        f"# {out_root.name}\n\nThe run `{args.run.name}` with the chirp parameters of the recalibrated decoder "
        f"bundle `{new.path}` (calibration refitted without templates {recal['excluded_template_ids']}, same "
        f"weights `{new.weights_sha256[:16]}`); made by `experiments/production/recalibrate_params.py` from the "
        "stored raw decoder output (no re-detection). Everything else is the source run's.\n\n"
        f"Over {len(recs)} recordings: rows with parameters {int(totals['rows_with_params']):,}; params_ok "
        f"{int(totals['params_ok_before']):,} -> {int(totals['params_ok_after']):,} "
        f"(lost {int(totals['params_ok_lost']):,}, gained {int(totals['params_ok_gained']):,}); calibrated height "
        f"changed by more than 1 % in {int(totals['height_changed_over_1pct']):,} rows. Check: the old calibration "
        f"reproduced every stored value and sd within {totals['verify_max_rel_diff']:.1e} (relative) and every flag.\n"
    )
    staging.rename(out_root)
    print(f"{len(recs)} recordings -> {out_root}")
    print(
        json.dumps(
            {
                k: (
                    v
                    if k.startswith(("verify", "height_ratio")) or k.endswith("_hz")
                    else int(v)
                )
                for k, v in totals.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
