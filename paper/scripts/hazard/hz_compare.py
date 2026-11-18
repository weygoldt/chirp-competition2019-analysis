"""Tables of the hazard analysis across detection sets, from the stats files.

New in the paper port. Reads ``paper/output/<run>/stats/chase_hazard.json``
(written by ``chirp_chase_hazard.py``) and ``boris_led_check.json``
(``hz_led_check.py``) for v1 (``rcnn``, the old detections rerun with the
ported code) and the v2 runs, and writes the markdown tables that
``paper/REPORT_hazard_v2.md`` quotes to
``paper/output/compare/hazard_comparison.md``. No number is computed here
except formatting; a run without a stats file shows "—".

    python3 paper/scripts/hazard/hz_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = [
    ("rcnn", "v1"),
    ("rtdetr_gapfill", "v2, 0.5"),
    ("rtdetr_gapfill_t080", "v2, 0.8 (primary)"),
    ("rtdetr_gapfill_t090", "v2, 0.9"),
]
ROOT = Path(__file__).resolve().parents[2] / "output"
OUT = ROOT / "compare" / "hazard_comparison.md"
DASH = "—"

OFFSET_ACCOUNTS = [  # hz_acausal_sim offset accounts, in its order
    ("causal_only", "causal only"), ("burst_only", "burst only"),
    ("causal_plus_burst", "causal + burst"), ("causal_lag_plus_burst", "causal 0-1.5 s + burst"),
    ("causal_het_plus_burst", "causal, dyads differ, + burst"),
    ("causal_plus_burst_drift", "causal + burst, drift"),
    ("causal_plus_burst_matched", "causal + burst, HR matched"),
    ("causal_plus_bout", "causal + bout"), ("common_trigger", "common trigger"),
    ("symmetric_cluster", "symmetric cluster"),
]
ONSET_ACCOUNTS = [
    ("onset_no_effect", "no effect"), ("onset_start_matched", "no effect, chase start matched"),
    ("onset_het_no_effect", "no mean effect, dyads differ"),
    ("onset_hidden_short", "short latent state"), ("onset_causal", "causal HR 0.5"),
    ("onset_causal_start", "causal HR 0.5, chase start matched"),
]


def load(run: str, namespace: str) -> dict | None:
    f = ROOT / run / "stats" / f"{namespace}.json"
    return json.loads(f.read_text())["values"] if f.exists() else None


def get(tree: dict | None, path: str):
    for key in path.split("."):
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def p_str(p: float | None) -> str:
    if p is None:
        return ""
    if p < 1e-3:
        return f", p = {p:.0e}"
    return f", p = {p:.3f}" if p < 0.1 else f", p = {p:.2f}"


def ci(d: dict | None) -> str:
    """'ratio [low, high], p = …' of a {ratio, low, high[, p]} dict."""
    if not d or "ratio" not in d:
        return DASH
    s = f"{d['ratio']:.2f}"
    if "low" in d and "high" in d:
        s += f" [{d['low']:.2f}, {d['high']:.2f}]"
    return s + p_str(d.get("p"))


def num(x, fmt: str = "{:,.0f}") -> str:
    return DASH if x is None else fmt.format(x)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def by_run(stats: dict, label: str, fn) -> list[str]:
    return [label] + [fn(stats[run]) for run, _ in RUNS]


def kind_rows(stats: dict, k: str) -> list[list[str]]:
    def dyads(path):
        def f(s):
            d = get(s, f"{k}.{path}")
            return DASH if not d else f"{d.get('informative', 0)}: {d.get('above', 0)} / " \
                                      f"{d.get('below', 0)}"
        return f

    def sens(name, backend):
        def f(s):
            d = get(s, f"{k}.sensitivity.{name}")
            return DASH if not d else f"{ci(d.get(backend))} ({num(d.get('n_dyads'))} dyads)"
        return f

    def jit(s):
        d = get(s, f"{k}.jitter_p")
        return DASH if not d else ", ".join(f"{n} {v:.3f}" for n, v in sorted(d.items()))

    def lodo(s):
        d = get(s, f"{k}.lodo_fe")
        return DASH if not d else f"{d['min']:.2f} to {d['max']:.2f}"

    def het(s):
        d = get(s, f"{k}.heterogeneity")
        return DASH if not d else f"{d.get('dyad_sd', float('nan')):.2f} (G test{p_str(d.get('g_p'))})"

    def support(s):
        d = get(s, f"{k}.support_led_dyads")
        return DASH if not d else f"{d['events']:.0f} of {d['events_all']:.0f}"

    rows = [
        by_run(stats, "events / risk-set bins", lambda s: DASH if get(s, f"{k}.n_events") is None
               else f"{get(s, f'{k}.n_events'):,.0f} / {get(s, f'{k}.n_bins'):,.0f}"),
        by_run(stats, "**1 s: FE HR** (dyad-clustered)", lambda s: ci(get(s, f"{k}.tau1s.fe"))),
        by_run(stats, "1 s: RE mean HR over dyads", lambda s: ci(get(s, f"{k}.tau1s.re"))),
        by_run(stats, "1 s: GLMM OR (one common effect)", lambda s: ci(get(s, f"{k}.tau1s.glmm"))),
        by_run(stats, "between-dyad SD of log HR", het),
        by_run(stats, "informative dyads: above / below 1", dyads("dyads_tau1s")),
        by_run(stats, "**guard 0.5-1.5 s: FE HR**", lambda s: ci(get(s, f"{k}.guard.fe"))),
        by_run(stats, "guard: RE mean HR", lambda s: ci(get(s, f"{k}.guard.re"))),
        by_run(stats, "guard: GLMM OR", lambda s: ci(get(s, f"{k}.guard.glmm"))),
        by_run(stats, "guard: informative dyads above / below", dyads("dyads_guard")),
        by_run(stats, "kernel mean OR 0-2 s (GLMM)", lambda s: ci(get(s, f"{k}.kernel_glmm.mean"))),
        by_run(stats, "leave one dyad out, FE HR range", lodo),
        by_run(stats, "jitter test p (FE refits)", jit),
        by_run(stats, "without the LED-repaired dyads, FE", sens("drop_led_corrected", "fe")),
        by_run(stats, "without close-EODf dyads, FE", sens("drop_close_eodf", "fe")),
        by_run(stats, "without sender-ambiguous dyads, FE", sens("drop_sender_ambiguous", "fe")),
        by_run(stats, "only unambiguous chirps, FE", sens("unambiguous_chirps", "fe")),
        by_run(stats, "events with a chirp from the LED-repaired dyads", support),
    ]
    for name in sorted({n for run, _ in RUNS for n in (get(stats[run], f"{k}.sensitivity") or {})
                        if n.startswith("shift_")}):
        rows.append(by_run(stats, f"{name.replace('_', ' ')}, FE", sens(name, "fe")))
    return rows


def dyad_rows(stats: dict, k: str, path: str) -> list[list[str]]:
    recs = sorted({r for run, _ in RUNS for r in (get(stats[run], f"{k}.{path}.per_dyad") or {})})

    def cell(s, rec):
        d = get(s, f"{k}.{path}.per_dyad.{rec}")
        if not d:
            return DASH
        if "oe" not in d:  # no bins with the indicator on
            return f"{d.get('observed', 0):.0f}/{d.get('expected', 0):.1f} (u)"
        mark = "" if not d.get("informative") else (
            " +" if d["low"] > 1 else (" −" if d["high"] < 1 else ""))
        return (f"{d['observed']:.0f}/{d['expected']:.1f} = {d['oe']:.2f} "
                f"[{d['low']:.2f}, {d['high']:.2f}]{mark}" + ("" if d.get("informative") else " (u)"))

    return [[rec.replace("_", "-")[5:]] + [cell(stats[run], rec) for run, _ in RUNS]
            for rec in recs]


def main() -> None:
    stats = {run: load(run, "chase_hazard") for run, _ in RUNS}
    led = next((x for x in (load(run, "boris_led_check") for run, _ in RUNS) if x), None)
    head = [""] + [lab for _, lab in RUNS]
    L = ["# Hazard analysis across detection sets", "",
         "Generated by `paper/scripts/hazard/hz_compare.py` from "
         "`paper/output/<run>/stats/chase_hazard.json`. Ratios with 95 % intervals; "
         "FE = cloglog GLM with dyad intercepts and dyad-clustered t(G-1) interval, RE = "
         "random-effects mean over dyads, GLMM = logit with dyad random intercept (assumes one "
         "common effect). v1 = the old RCNN detections rerun with the ported code.", ""]
    missing = [lab for run, lab in RUNS if stats[run] is None]
    if missing:
        L += [f"No stats file yet: {', '.join(missing)}.", ""]

    L += ["## Data", ""]
    L += table(head, [
        by_run(stats, label, lambda s, key=key: num(get(s, f"data.{key}")))
        for label, key in (("dyads", "n_dyads"), ("chases", "n_chases"),
                           ("loser chirps", "n_sub_chirps"),
                           ("loser chirps inside chases", "n_sub_chirps_in_chase"),
                           ("winner chirps", "n_dom_chirps"),
                           ("unassigned distinct detections", "n_unassigned_distinct_window"))])
    for k, title in (("offset", "(a) chase offsets: loser chirp in the preceding window"),
                     ("onset", "(b) chase onsets (re-attack): loser chirp in the preceding window")):
        L += ["", f"## {title}", ""]
        L += table(head, kind_rows(stats, k))
        L += ["", f"Per dyad, {k}s, 1 s window: observed / expected events in bins with a loser "
              "chirp in the preceding 1 s (expected = no chirp effect; exact 95 % CI; + / − = "
              "informative dyad with CI above / below 1; (u) = uninformative, < 2 expected).", ""]
        L += table(["dyad"] + head[1:], dyad_rows(stats, k, "dyads_tau1s"))
        L += ["", f"Per dyad, {k}s, guard window 0.5-1.5 s:", ""]
        L += table(["dyad"] + head[1:], dyad_rows(stats, k, "dyads_guard"))

    L += ["", "## Simulations (hz_sim)", ""]
    sim_rows = [
        by_run(stats, "offset: trigger share reproducing the FE HR",
               lambda s: num(get(s, "simulation.offset.trigger_share"), "{:.0%}")),
        by_run(stats, "offset: offsets preceded by a trigger chirp",
               lambda s: num(get(s, "simulation.offset.trigger_offsets_frac"), "{:.0%}")),
    ]
    for k in ("offset", "onset"):
        sim_rows += [
            by_run(stats, f"{k}: recovery, true HR → FE median [95 % range]",
                   lambda s, k=k: DASH if not get(s, f"simulation.{k}.recovery") else
                   f"{get(s, f'simulation.{k}.recovery.truth'):.2f} → "
                   f"{get(s, f'simulation.{k}.recovery.fe'):.2f} "
                   f"[{get(s, f'simulation.{k}.recovery.fe_low'):.2f}, "
                   f"{get(s, f'simulation.{k}.recovery.fe_high'):.2f}]"),
            by_run(stats, f"{k}: FPR FE cluster / RE, dyads differ, no mean effect",
                   lambda s, k=k: DASH if not get(s, f"simulation.{k}.fpr_het_no_effect") else
                   f"{get(s, f'simulation.{k}.fpr_het_no_effect.fe_cluster'):.0%} / "
                   f"{get(s, f'simulation.{k}.fpr_het_no_effect.re'):.0%}"),
        ]
    for k, sc, lab in (("offset", "lag_05", "offset: offsets marked 0.5 s late"),
                       ("offset", "annotation", "offset: 0.3 s boundary error on a 0.5 s lattice"),
                       ("offset", "detection", "offset: missed chirps"),
                       ("onset", "hidden_state_short", "onset: short latent state (3 s)"),
                       ("onset", "hidden_state", "onset: slow latent state (30 s)"),
                       ("onset", "annotation", "onset: 0.3 s boundary error on a 0.5 s lattice")):
        sim_rows.append(by_run(stats, f"{lab}, true HR → FE median [95 % range]",
                               lambda s, k=k, sc=sc: DASH
                               if not get(s, f"simulation.{k}.{sc}") else
                               f"{num(get(s, f'simulation.{k}.{sc}.truth'), '{:.2f}')} → "
                               f"{get(s, f'simulation.{k}.{sc}.hr'):.2f} "
                               f"[{get(s, f'simulation.{k}.{sc}.low'):.2f}, "
                               f"{get(s, f'simulation.{k}.{sc}.high'):.2f}]"))
    L += table(head, sim_rows)

    L += ["", "## Acausal control (hz_acausal)", ""]
    ac_rows = []
    for k in ("offset", "onset"):
        ac_rows += [
            by_run(stats, f"{k}: HR_past / HR_future, 1 s (FE)",
                   lambda s, k=k: ci(get(s, f"acausal.{k}.pf_1s.ratio"))),
            by_run(stats, f"{k}: HR_past / HR_future, 0.5-1.5 s (FE)",
                   lambda s, k=k: ci(get(s, f"acausal.{k}.pf_guard.ratio"))),
            by_run(stats, f"{k}: kernel before / after (FE)",
                   lambda s, k=k: ci(get(s, f"acausal.{k}.kernel.ratio"))),
        ]
    ac_rows += [
        by_run(stats, "offset: restricted real / mirror, 1 s, W = 2 s",
               lambda s: ci(get(s, "acausal.offset.mirror_w_2000ms.tau1s"))),
        by_run(stats, "offset: restricted real / mirror, guard, W = 2 s",
               lambda s: ci(get(s, "acausal.offset.mirror_w_2000ms.guard"))),
        by_run(stats, "offset: restricted real / mirror, guard, W = 5 s",
               lambda s: ci(get(s, "acausal.offset.mirror_w_5000ms.guard"))),
    ]
    L += table(head, ac_rows)
    for k, accounts, stat_list in (
            ("offset", OFFSET_ACCOUNTS, (("hr_primary", "standard 1 s HR"),
                                         ("ratio_pf", "past/future 1 s"),
                                         ("ratio_grd", "past/future guard"),
                                         ("ratio_k", "kernel ratio"))),
            ("onset", ONSET_ACCOUNTS, (("on_hr_primary", "standard 1 s HR"),
                                       ("on_ratio_pf", "past/future 1 s"),
                                       ("on_ratio_grd", "past/future guard")))):
        for run, lab in RUNS:
            sim = get(stats[run], f"acausal.{k}.simulation")
            if not sim:
                continue
            L += ["", f"Acausal simulation ({k}, {lab}): account median [95 % range]; "
                  "exceedance = share of the account's datasets at or beyond the observed value "
                  "(< 2.5 % inconsistent, 2.5-10 % borderline).", ""]
            obs = [next((sim[a][st]["observed"] for a, _ in accounts
                         if st in sim.get(a, {})), None) for st, _ in stat_list]
            hdr = ["account"] + [f"{sl} (obs {num(o, '{:.2f}')})" for (_, sl), o
                                 in zip(stat_list, obs, strict=True)]
            rows = []
            for a, alab in accounts:
                cells = []
                for st, _ in stat_list:
                    d = sim.get(a, {}).get(st)
                    cells.append(DASH if not d else
                                 f"{d['median']:.2f} [{d['q025']:.2f}, {d['q975']:.2f}]; "
                                 f"{d['exceed_frac']:.0%}")
                rows.append([alab] + cells)
            L += table(hdr, rows)

    if led:
        L += ["", "## BORIS/LED conversion: paper converter minus repaired map", "",
              f"{led['n_trials']:.0f} trials with BORIS and LED files ({led['n_trials_analysed']:.0f} "
              f"analysed by the paper scripts); difference > 0.1 s in the first 3 h in "
              f"{led['n_trials_max_gt_0_1s_3h']:.0f}; in all others at most "
              f"{led['max_abs_3h_s_other_trials'] * 1e3:.0f} ms.", ""]
        rows = []
        for rec, d in sorted(led["trials"].items()):
            if d["max_abs_3h_s"] <= 0.1:
                continue
            rows.append([rec.replace("_", "-"), f"{d['n_events_3h']:.0f}",
                         f"{d['median_diff_3h_s']:+.2f}", f"{d['max_abs_3h_s']:.2f}",
                         f"{d['n_gt_0_5s_3h']:.0f}", num(d.get("first_gt_0_5s_t"), "{:,.0f}"),
                         f"{d['ref_led_resid_med_ms']:.0f}", f"{d['robust_led_resid_med_ms']:.0f}"])
        L += table(["trial", "events (0-3 h)", "median difference (s)", "max |difference| (s)",
                    "events off by > 0.5 s", "first such event (grid s)",
                    "LED residual, paper map (ms)", "LED residual, repaired map (ms)"], rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
