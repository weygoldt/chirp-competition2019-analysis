"""Score the 2022 labelling session real_v2_2022 (run only once labels exist).

"Agreement with the expert" is the share of detections the author judged
chirps; the "found share" is the share of the chirps the author marked in the
recall windows that the detector found. Neither is precision or recall: on
real recordings the expert's labels are not ground truth (spec ground rule).

Reads the truth file (``labels/real_v2_2022/truth/real_v2_2022_truth.parquet``),
the labeller's label files (``labels/real_v2_2022_labels/labels/*.csv``) and,
for the windows, the t001 detections. Reports:

1. **Agreement** ``A = sum(w y) / sum(w)`` over labelled candidates, ``y = 1``
   for ``chirp`` / ``chirp_emitter_unclear``, ``0`` for ``not_chirp``,
   ``w = N_stratum / n_labelled_stratum`` (stratum = score bin x status x
   group x size, pooled over recordings; unlabelled items are taken as
   missing at random within their stratum). By score bin, at thresholds
   0.5-0.95 (domain estimation), and by status, group, group x status and
   size. ``N_hat_chirps``: estimated chirps among the frame's detections of
   that domain.
2. **Found share** from the finished windows: a mark is found at threshold
   ``t`` if a kept detection with score >= t matches it (real_v1's rule,
   ``score_session.match_marks``: centres within 100 ms, the mark +-15 Hz
   overlapping the box, one-to-one); ``F(t) = sum(w found) / sum(w marks)``
   with ``w = N_slots_group / n_done_group``. Also chirps per window (and per
   minute) and detections per window inside the zones.
3. **Test-retest** on the hidden repeats (``score_session.retest_table``).
4. **Emitter agreement** (``score_session.emitter_table``).
5. **Sensitivity: pooled with real_v1's 2022 items**: a composite of the two
   sessions' estimates, ``lambda A_new + (1 - lambda) A_real_v1`` with
   ``lambda`` = the sessions' share of the effective sample size (Kish) in
   the domain, so that real_v1's few heavy items count for little. The two
   frames differ only by real_v1's own items.

95 % percentile bootstrap intervals (2,000 resamples) within the design's
strata (candidates) or groups (windows).

    .venv/bin/python experiments/real_labels_2022/score_2022.py [--labels DIR] [--out DIR]

Output: ``labels/real_v2_2022/report/`` (``report.md``, ``metrics.json``,
tables, ``thresholds.png``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design2022 as d

sys.path.insert(1, str(d.REAL_V1_CODE))
import score_session as s1

SEED = 11
BY = ([], ["status"], ["group"], ["group", "status"], ["size"])


# candidates -------------------------------------------------------------------


def candidate_items(truth: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Every candidate with its last decision and the design weight."""
    t = s1.candidate_table(truth, labels)
    n_lab = t[t["labelled"]].groupby("stratum").size()
    t["n_labelled_stratum"] = t["stratum"].map(n_lab).fillna(0).astype(int)
    t["w"] = np.where(t["labelled"], t["cell_N"] / t["n_labelled_stratum"].clip(lower=1), 0.0)
    return t


def score_domains(main: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    s = main["score"].to_numpy(float)
    out = [(b, (main["score_bin"] == b).to_numpy()) for b in d.SCORE_NAMES]
    out += [(f">= {t:g}", s >= t) for t in d.THRESHOLDS]
    out += [("0.5-0.9", (s >= 0.5) & (s < 0.9))]
    return out


def boot_ratios(num: np.ndarray, den: np.ndarray, index: np.ndarray) -> np.ndarray:
    """sum(num) / sum(den) in every resample (NaN where the domain is empty)."""
    n, dd = num[index].sum(axis=1), den[index].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dd > 0, n / np.where(dd > 0, dd, 1.0), np.nan)


def _ci(r: np.ndarray) -> tuple[float, float]:
    r = r[np.isfinite(r)]
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))) if r.size else (np.nan, np.nan)


def strata_index(main: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    return s1.boot_matrix([g.index.to_numpy() for _, g in main.groupby("stratum")], rng)


def domains(main: pd.DataFrame, by: list[str]):
    """(labels, mask) of every combination of the ``by`` columns x score domain."""
    combos = [((), np.ones(len(main), bool))] if not by else [
        (key, np.all([(main[c] == v).to_numpy() for c, v in zip(by, key, strict=True)], axis=0))
        for key, _ in main.groupby(by)]
    for key, m0 in combos:
        for name, dm in score_domains(main):
            m = m0 & dm
            if m.any():
                labels = {"by": "+".join(by) or "all", "group": "all", "status": "all", "size": "all"}
                labels |= dict(zip(by, key, strict=True))
                yield {**labels, "scores": name}, m


def agreement_table(main: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    index = strata_index(main, rng)
    w, y = main["w"].to_numpy(), main["y"].to_numpy()
    rows = []
    for by in BY:
        for labels, m in domains(main, by):
            num, den = w * y * m, w * m
            lo, hi = _ci(boot_ratios(num, den, index))
            rows.append({**labels, "n": int(m.sum()), "n_eff": d.kish_n_eff(w[m]),
                         "N_hat": float(den.sum()), "agreement": float(num.sum() / den.sum()),
                         "ci_low": lo, "ci_high": hi, "N_hat_chirps": float(num.sum()),
                         "agreement_unweighted": float(y[m].mean())})
    return pd.DataFrame(rows)


# windows ----------------------------------------------------------------------


def window_items(truth: pd.DataFrame, labels: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Per finished window: marks, matches at every threshold; group weights."""
    win, marks = s1.window_table(truth, labels)
    rec = s1.recall_tables(win, marks, rng)  # per-window matching of real_v1 (its pooling unused)
    per = rec["per_window"]
    if per.empty:
        return {"win": win, "marks": marks, "per_window": per, "pairs": rec["pairs"]}
    per = per.merge(win[["candidate_id", "group", "cell_N"]], on="candidate_id", how="left")
    n_done = per.groupby("group").size()
    per["w"] = per["cell_N"] / per["group"].map(n_done)
    return {"win": win, "marks": marks, "per_window": per, "pairs": rec["pairs"]}


def window_tables(per: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    if per.empty:
        return pd.DataFrame(), pd.DataFrame()
    index = s1.boot_matrix([g.index.to_numpy() for _, g in per.groupby("group")], rng)
    w, m = per["w"].to_numpy(), per["marks"].to_numpy(float)
    found_rows, rate_rows = [], []
    for g in ("all", *d.GROUPS):
        dm = np.ones(len(per), bool) if g == "all" else (per["group"] == g).to_numpy()
        if not dm.any():
            continue
        lo, hi = _ci(boot_ratios(w * m * dm, w * dm, index))
        rate = float((w * m * dm).sum() / (w * dm).sum())
        rate_rows.append({"group": g, "windows": int(dm.sum()), "marks": int(m[dm].sum()),
                          "windows_with_marks": int((m[dm] > 0).sum()),
                          "chirps_per_window": rate, "ci_low": lo, "ci_high": hi,
                          "chirps_per_minute": rate * 60.0 / 2.0,
                          **{f"detections_per_window_{t:g}": float(
                              (w * per[f"det_zone_{t}"].to_numpy(float) * dm).sum() / (w * dm).sum())
                             for t in d.THRESHOLDS}})
        for t in d.THRESHOLDS:
            f = per[f"found_{t}"].to_numpy(float)
            ft = per[f"found_time_only_{t}"].to_numpy(float)
            dz = per[f"det_zone_{t}"].to_numpy(float)
            dzm = per[f"det_zone_matched_{t}"].to_numpy(float)
            lo, hi = _ci(boot_ratios(w * f * dm, w * m * dm, index))
            den = (w * m * dm).sum()
            found_rows.append({
                "group": g, "threshold": t, "windows": int(dm.sum()), "marks": int(m[dm].sum()),
                "found": int(f[dm].sum()),
                "found_share": float((w * f * dm).sum() / den) if den > 0 else np.nan,
                "ci_low": lo, "ci_high": hi,
                "found_share_time_only": float((w * ft * dm).sum() / den) if den > 0 else np.nan,
                "zone_detections": int(dz[dm].sum()),
                "zone_agreement": float((w * dzm * dm).sum() / (w * dz * dm).sum())
                if (w * dz * dm).sum() > 0 else np.nan})
    return pd.DataFrame(found_rows), pd.DataFrame(rate_rows)


# sensitivity: pooled with real_v1 ----------------------------------------------


def real_v1_candidates() -> pd.DataFrame:
    """real_v1's labelled 2022 candidates with score >= 0.5, with its design weights."""
    truth = pd.read_parquet(d.REAL_V1_TRUTH)
    t = s1.design_weights(s1.candidate_table(truth, s1.read_labels(d.REAL_V1_LABELS)))
    t = t[(t["dataset"] == "competition2022") & t["stratum"].str.startswith("main/") & t["labelled"]
          & (t["score"] >= d.SCORE_EDGES[0])].reset_index(drop=True)
    t["score_bin"] = d.score_bin(t["score"])
    t["group"] = d.group_of(t["recording"])
    return t


def real_v1_windows(rng: np.random.Generator) -> pd.DataFrame:
    """real_v1's finished 2022 windows (activity-weighted draws, weight 1 / pi)."""
    truth = pd.read_parquet(d.REAL_V1_TRUTH)
    truth = truth[truth["dataset"] == "competition2022"]
    win, marks = s1.window_table(truth, s1.read_labels(d.REAL_V1_LABELS))
    per = s1.recall_tables(win, marks, rng)["per_window"]
    if per.empty:
        return per
    per["group"] = d.group_of(per["recording"])
    per["w"] = per["weight"]
    return per


def composite(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> dict:
    """Composite ``sum_k lambda_k R_k`` of ratio estimates ``R_k = sum(num_k) /
    sum(den_k)``, ``lambda_k`` proportional to ``n_eff_k``; parts: (num, den,
    bootstrap index, n_eff). The first part is the new session."""
    present = [den.sum() > 0 and neff > 0 for _, den, _, neff in parts]
    if not any(present):
        return {"pooled": np.nan, "pooled_ci_low": np.nan, "pooled_ci_high": np.nan, "lambda_new": np.nan}
    neffs = np.array([p[3] if ok else 0.0 for p, ok in zip(parts, present, strict=True)])
    lam = neffs / neffs.sum()
    est, boot = 0.0, np.zeros(s1.N_BOOT)
    for (num, den, index, _), lam_k, ok in zip(parts, lam, present, strict=True):
        if ok:
            est += lam_k * num.sum() / den.sum()
            boot += lam_k * boot_ratios(num, den, index)  # NaN where a resample's domain is empty
    lo, hi = _ci(boot)
    return {"pooled": float(est), "pooled_ci_low": lo, "pooled_ci_high": hi, "lambda_new": float(lam[0])}


def pooled_agreement(new: pd.DataFrame, old: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx_new, idx_old = strata_index(new, rng), strata_index(old, rng)
    wn, yn, wo, yo = new["w"].to_numpy(), new["y"].to_numpy(), old["w"].to_numpy(), old["y"].to_numpy()
    old_masks = {}
    for by in BY[:4]:
        for labels, m in domains(old, by):
            old_masks[tuple(labels.values())] = m
    rows = []
    for by in BY[:4]:
        for labels, m in domains(new, by):
            mo = old_masks.get(tuple(labels.values()), np.zeros(len(old), bool))
            row = {**labels, "n_new": int(m.sum()), "agreement_new": float((wn * yn)[m].sum() / wn[m].sum()),
                   "n_real_v1": int(mo.sum()),
                   "agreement_real_v1": float((wo * yo)[mo].sum() / wo[mo].sum()) if mo.any() else np.nan,
                   "n_eff_new": d.kish_n_eff(wn[m]), "n_eff_real_v1": d.kish_n_eff(wo[mo])}
            row |= composite([(wn * yn * m, wn * m, idx_new, row["n_eff_new"]),
                              (wo * yo * mo, wo * mo, idx_old, row["n_eff_real_v1"])])
            rows.append(row)
    return pd.DataFrame(rows)


def mark_n_eff(w: np.ndarray, marks: np.ndarray) -> float:
    """Kish's effective number of marks: every mark carries its window's weight."""
    s2 = float((w**2 * marks).sum())
    return float((w * marks).sum() ** 2 / s2) if s2 > 0 else 0.0


def pooled_found(new: pd.DataFrame, old: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Found share of the new windows, of real_v1's 2022 windows, and their composite."""
    if new.empty or old.empty:
        return pd.DataFrame()
    sessions = {"new": (new, s1.boot_matrix([g.index.to_numpy() for _, g in new.groupby("group")], rng)),
                "real_v1": (old, s1.boot_matrix([np.arange(len(old))], rng))}
    rows = []
    for g in ("all", *d.GROUPS):
        for t in d.THRESHOLDS:
            row, parts = {"group": g, "threshold": t}, []
            for tag, (per, index) in sessions.items():
                m = np.ones(len(per), bool) if g == "all" else (per["group"] == g).to_numpy()
                w, mk = per["w"].to_numpy(float), per["marks"].to_numpy(float) * m
                num, den = w * per[f"found_{t}"].to_numpy(float) * m, w * mk
                row |= {f"marks_{tag}": int(mk.sum()),
                        f"found_share_{tag}": float(num.sum() / den.sum()) if den.sum() > 0 else np.nan}
                parts.append((num, den, index, mark_n_eff(w, mk)))
            rows.append(row | composite(parts))
    return pd.DataFrame(rows)


# report -----------------------------------------------------------------------


def _get(df: pd.DataFrame, **kv) -> pd.Series | None:
    m = np.ones(len(df), bool)
    for k, v in kv.items():
        m &= (df[k] == v).to_numpy()
    return df[m].iloc[0] if m.any() else None


def _fmt(r: pd.Series | None, col: str) -> str:
    if r is None or not np.isfinite(r[col]):
        return "-"
    return f"{r[col]:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"


def headline(agree: pd.DataFrame, found: pd.DataFrame) -> list[str]:
    lines = []
    for t in (0.8, 0.9):
        a = _get(agree, by="all", scores=f">= {t:g}")
        f = _get(found, group="all", threshold=t) if len(found) else None
        lines.append(f"- **Threshold {t:g}**: agreement {_fmt(a, 'agreement')} "
                     f"({int(a['n']) if a is not None else 0} candidates); found share "
                     f"{_fmt(f, 'found_share')} ({int(f['marks']) if f is not None else 0} marked chirps).")
    for t in (0.8, 0.9):
        parts = []
        for st in ("winner", "loser"):
            for g in d.GROUPS:
                r = _get(agree, by="group+status", group=g, status=st, scores=f">= {t:g}")
                parts.append(f"{st} {g} {_fmt(r, 'agreement')}")
        lines.append(f"- Agreement at {t:g} by fish: " + "; ".join(parts) + ".")
    return lines


def thresholds_plot(agree: pd.DataFrame, found: pd.DataFrame, path: Path, fake: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    colours = {"all": "0.15", "pair3": "#b5446e", "rest": "#2a7ab0"}
    offsets = {"all": 0.0, "pair3": -0.008, "rest": 0.008}
    for g in ("all", *d.GROUPS):
        a = agree[(agree["by"] == ("all" if g == "all" else "group")) & agree["scores"].str.startswith(">=")]
        a = a[a["group"] == g]
        x = a["scores"].str.removeprefix(">= ").astype(float) + offsets[g]
        axes[0].errorbar(x, a["agreement"], yerr=[a["agreement"] - a["ci_low"], a["ci_high"] - a["agreement"]],
                         marker="o", ms=4, capsize=2, color=colours[g], label=g)
        if len(found):
            f = found[found["group"] == g]
            axes[1].errorbar(f["threshold"] + offsets[g], f["found_share"],
                             yerr=[f["found_share"] - f["ci_low"], f["ci_high"] - f["found_share"]],
                             marker="o", ms=4, capsize=2, color=colours[g], label=g)
    axes[0].set(xlabel="detector threshold (score >=)", ylabel="share (95 % CI)",
                title="detections the expert judged chirps", ylim=(-0.02, 1.02))
    axes[1].set(xlabel="detector threshold (score >=)", title="marked chirps the detector found")
    axes[0].legend(frameon=False, fontsize=8)
    title = "2022 detections vs the expert (real_v2_2022)"
    if fake:
        title = "FAKE LABELS (smoke test, numbers mean nothing): " + title
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, default=d.LABELS_OUT)
    ap.add_argument("--truth", type=Path, default=d.TRUTH)
    ap.add_argument("--out", type=Path, default=d.ROOT / "report")
    ap.add_argument("--no-real-v1", action="store_true", help="skip the pooled sensitivity check")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    truth = pd.read_parquet(args.truth)
    labels = s1.read_labels(args.labels)
    fake = bool((labels["labeler"].astype(str) == "FAKE").any())
    t = candidate_items(truth, labels)
    main_items = t[t["stratum"].str.startswith("main/") & t["labelled"]].reset_index(drop=True)
    agree = agreement_table(main_items, rng)
    wi = window_items(truth, labels, rng)
    found, rates = window_tables(wi["per_window"], rng)
    retest = s1.retest_table(t, wi["win"], wi["marks"])
    emit = s1.emitter_table(t, wi["pairs"])
    sens_a = sens_f = pd.DataFrame()
    if not args.no_real_v1:
        sens_a = pooled_agreement(main_items, real_v1_candidates(), rng)
        sens_f = pooled_found(wi["per_window"], real_v1_windows(rng), rng)

    design = t[t["stratum"].str.startswith("main/")]
    strata = design.groupby("stratum").agg(N=("cell_N", "first"), labelled=("labelled", "sum"))
    progress = {
        "fake_labels": fake,
        "candidates_labelled": int(t["labelled"].sum()), "candidates_total": len(t),
        "windows_done": int(wi["win"]["done"].sum()), "windows_total": len(wi["win"]),
        "marks": len(wi["marks"]),
        "strata_without_labels": int((strata["labelled"] == 0).sum()),
        "detections_in_strata_without_labels": int(strata.loc[strata["labelled"] == 0, "N"].sum()),
        "labelling_minutes_sum_of_item_times": float(
            (t["elapsed_s"].sum() + wi["win"]["elapsed_done_s"].sum()) / 60.0),
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    agree.to_csv(out / "agreement.csv", index=False)
    found.to_csv(out / "found_share.csv", index=False)
    rates.to_csv(out / "chirp_rates.csv", index=False)
    wi["per_window"].to_csv(out / "windows.csv", index=False)
    emit.to_csv(out / "emitter_agreement.csv", index=False)
    sens_a.to_csv(out / "pooled_real_v1_agreement.csv", index=False)
    sens_f.to_csv(out / "pooled_real_v1_found_share.csv", index=False)
    t.to_csv(out / "candidates_scored.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({"progress": progress, "retest": retest}, indent=1,
                                                 default=float))
    if len(agree):
        thresholds_plot(agree, found, out / "thresholds.png", fake)

    def table(df: pd.DataFrame, by: str, cols: list[str]) -> str:
        return s1._md(df[df["by"] == by][cols]) if len(df) else "-"

    acols = ["scores", "n", "n_eff", "N_hat", "agreement", "ci_low", "ci_high", "N_hat_chirps"]
    md = ["# real_v2_2022: 2022 detections vs the expert", ""]
    if fake:
        md += ["**FAKE LABELS: a smoke test of the scorer. These numbers mean nothing.**", ""]
    md += ["Agreement = share of detections the expert judged chirps; found share = share of the "
           "chirps the expert marked in the windows that the detector found. Not precision or recall: "
           "the expert's labels are not ground truth.", "",
           f"Progress: {json.dumps(progress)}", "", "## In short", "", *headline(agree, found), "",
           "## Agreement, all 2022 detections (score >= 0.5, kept, dark phase)", "",
           table(agree, "all", acols), "",
           "## By the assigned fish's trial status", "",
           table(agree, "status", ["status", *acols]), "",
           "## By group (pair 3 = 2022-06-20/22/24; rest = the other recordings)", "",
           table(agree, "group", ["group", *acols]), "",
           "## By group and status", "", table(agree, "group+status", ["group", "status", *acols]), "",
           "## By chirp size (decoder height < / >= 200 Hz)", "", table(agree, "size", ["size", *acols]), "",
           "## Found share (recall windows, uniform over the dark phase)", "",
           s1._md(found) if len(found) else "(no finished windows)", "",
           "## Chirps per window (2 s) and detections inside the zones", "",
           s1._md(rates) if len(rates) else "-", "",
           "## Test-retest (hidden repeats)", "", "```", json.dumps(retest, indent=1, default=float), "```", "",
           "## Emitter agreement", "", s1._md(emit) if len(emit) else "-", ""]
    if len(sens_a):
        sc = ["by", "group", "status", "scores", "n_new", "agreement_new", "n_real_v1", "agreement_real_v1",
              "n_eff_new", "n_eff_real_v1", "pooled", "pooled_ci_low", "pooled_ci_high"]
        md += ["## Sensitivity: pooled with real_v1's 2022 candidates", "",
               "Composite weighted by each session's effective sample size in the domain.", "",
               s1._md(sens_a[sens_a["scores"].str.startswith(">=") | sens_a["scores"].isin(d.SCORE_NAMES)][sc]),
               "", "## Sensitivity: found share pooled with real_v1's 2022 windows", "",
               s1._md(sens_f) if len(sens_f) else "-", ""]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md[: 12 + len(headline(agree, found))]))


if __name__ == "__main__":
    main()
