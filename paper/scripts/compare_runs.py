"""Compare the manuscript's numbers across detection sets (paper reruns).

Reads the output roots of ``run_all.sh`` for several ``PAPER_DETECTIONS``
values (default: ``rtdetr`` = v2 as produced, ``rtdetr_gapfill`` = v2 with
the trial fish's track gaps filled, ``rtdetr_gapfill_t090`` = the same at
operating threshold 0.9, the robustness check) plus the v1 detections for
the chirp counts, and writes one markdown table set to
``$CHIRP_DATA/paper_rerun/compare/comparison.md``:

1. chirp counts in the 33 analysed trials (whole recording, first 3 h),
2. every reported posterior (``reported_posteriors_median_hdi.csv``),
3. hurdle multipliers, echo responses and temporal allocation (parsed from
   the run logs, where the scripts print them),
4. the event-rate extrema printout (raw lines).

Runs that are missing or incomplete are reported as such, not skipped
silently.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paper_config import CHIRP_DATA, DATA_ROOT

RUNS = [
    ("rtdetr", "v2, 0.5"),
    ("rtdetr_gapfill", "v2 gap-filled, 0.5"),
    ("rtdetr_gapfill_t090", "v2 gap-filled, 0.9"),
]
BAD_RECORDINGS = ["2020-05-21-10_00", "2020-05-25-10_00", "2020-05-28-10_00", "2020-06-10-10_00"]
ROOT = CHIRP_DATA / "paper_rerun"


def counts() -> str:
    ts = pd.read_csv(DATA_ROOT / "trial_summary.csv", index_col=0)
    ts = ts[(ts["draw"] == 0) & ~ts["recording"].isin(BAD_RECORDINGS)]
    srcs = [("rcnn", "v1")] + RUNS
    rows = []
    for _, r in ts.iterrows():
        for src, lab in srcs:
            f = DATA_ROOT / r["recording"] / f"chirp_times_{src}.npy"
            if not f.exists():
                continue
            t = np.load(f)
            i = np.load(DATA_ROOT / r["recording"] / f"chirp_ids_{src}.npy")
            for status, fid in (("winner", r["win_ID"]), ("loser", r["lose_ID"])):
                sel = i == fid
                rows.append({"src": lab, "status": status, "rec": r["recording"],
                             "whole": int(sel.sum()), "first_3h": int((sel & (t < 3 * 3600)).sum())})
    x = pd.DataFrame(rows)
    order = [lab for _, lab in srcs if lab in set(x["src"])]
    out = ["## 1. Chirps in the 33 analysed trials", ""]
    for col, title in (("whole", "whole recording"), ("first_3h", "first 3 h")):
        tot = x.pivot_table(index="status", columns="src", values=col, aggfunc="sum")[order]
        med = x.pivot_table(index="status", columns="src", values=col, aggfunc="median")[order]
        zero = x[(x["status"] == "winner") & (x[col] == 0)].groupby("src").size().reindex(order, fill_value=0)
        out += [f"**{title}** (total; per-trial median in brackets)", "",
                "| | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        for st in ("winner", "loser"):
            out.append(f"| {st} | " + " | ".join(f"{int(tot.loc[st, c]):,} ({med.loc[st, c]:.0f})" for c in order) + " |")
        out.append("| winners with 0 chirps | " + " | ".join(str(int(zero[c])) for c in order) + " |")
        out.append("")
    return "\n".join(out)


def posteriors() -> str:
    frames = []
    for run, lab in RUNS:
        f = ROOT / run / "data" / "processed" / "reported_posteriors_median_hdi.csv"
        if not f.exists():
            frames.append(None)
            continue
        d = pd.read_csv(f)
        d[lab] = [f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in zip(d["median"], d["hdi_low"], d["hdi_high"])]
        frames.append(d.set_index(["quantity", "unit"])[[lab]])
    present = [f for f in frames if f is not None]
    missing = [lab for (_, lab), f in zip(RUNS, frames) if f is None]
    t = pd.concat(present, axis=1)
    out = ["## 2. Reported posteriors (median [95% HDI])", ""]
    if missing:
        out += [f"Missing: {', '.join(missing)}.", ""]
    cols = list(t.columns)
    out += ["| quantity | unit | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for (q, u), r in t.iterrows():
        out.append(f"| {q} | {u} | " + " | ".join("" if pd.isna(v) else v for v in r) + " |")
    return "\n".join(out) + "\n"


def grep(run: str, log: str, patterns: list[str]) -> list[str]:
    f = ROOT / run / "logs" / log
    if not f.exists():
        return [f"(no {log})"]
    lines = f.read_text(errors="replace").splitlines()
    return [ln.strip() for ln in lines if any(re.search(p, ln) for p in patterns)]


def parsed() -> str:
    out = ["## 3. Hurdle, echo responses, temporal allocation (from the run logs)", ""]
    for run, lab in RUNS:
        out += [f"### {lab}", "", "```"]
        out += grep(run, "stats_bayesian_hurdle_glmm_chirprate_deviation_during_chasing_offsets.log",
                    [r"status\[[01]\] \(", r"odds ratio", r"rate ratio"])
        out += grep(run, "stats_bayesian_logreg_echo_responses_new.log", [r"\\item \$P\("])
        out += grep(run, "plot_event_trial_progression_correlations_all_fish.log",
                    [r"first_h median", r"Per-trial count", r"Total events"])
        out += ["```", ""]
    return "\n".join(out)


def extrema() -> str:
    out = ["## 4. Event-rate extrema (raw printout of the src_paper plot)", ""]
    for run, lab in RUNS:
        out += [f"### {lab}", "", "```"]
        out += grep(run, "plot_event_time_rates_winner_loser_chirp_src_paper.log",
                    [r"▲|▼|peak|trough|region|^=+ |Panel|panel"])[:60]
        out += ["```", ""]
    return "\n".join(out)


def main() -> None:
    out_dir = ROOT / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        "# Manuscript numbers across detection sets",
        "",
        "Runs: " + "; ".join(f"`{r}` = {lab}" for r, lab in RUNS) + ". v1 = `rcnn` (counts only).",
        "",
        counts(), posteriors(), parsed(), extrema(),
    ])
    (out_dir / "comparison.md").write_text(text)
    print(text[:6000])
    print(f"\n... written to {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    os.environ.setdefault("PAPER_DETECTIONS", "rtdetr")
    main()
