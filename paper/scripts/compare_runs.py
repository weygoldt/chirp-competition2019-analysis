"""Compare the manuscript's numbers across detection sets (paper reruns).

Reads the stats files that the analysis modules write (``write_stats``,
``paper/output/<run>/stats/<namespace>.json``) for several
``PAPER_DETECTIONS`` values (``rtdetr`` = v2 as produced, ``rtdetr_gapfill`` =
v2 with the trial fish's track gaps filled, ``rtdetr_gapfill_t080`` = the same
at the manuscript's operating threshold 0.8, ``rtdetr_gapfill_t090`` = at 0.9)
and writes ``paper/output/compare/comparison.md``: one table per namespace,
one row per number, one column per run. The v1 chirp counts are in the
``counts`` namespace (written by ``compare_detection_counts.py``).

A run without a stats file of a namespace shows "—" there; runs without any
stats files are listed at the top, not skipped silently.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = [
    ("rtdetr", "v2, 0.5"),
    ("rtdetr_gapfill", "v2 gap-filled, 0.5"),
    ("rtdetr_gapfill_t080", "v2 gap-filled, 0.8 (primary)"),
    ("rtdetr_gapfill_t090", "v2 gap-filled, 0.9"),
]
ROOT = Path(__file__).resolve().parents[1] / "output"


def leaves(tree, prefix: str = "") -> dict:
    """Dotted key -> number or list of numbers."""
    if isinstance(tree, dict):
        return {k: v for key, item in tree.items() for k, v in leaves(item, f"{prefix}{key}.").items()}
    return {prefix[:-1]: tree}


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.4g}"


def load(run: str) -> dict[str, dict]:
    """Namespace -> flattened values of one run."""
    return {f.stem: leaves(json.loads(f.read_text())["values"])
            for f in sorted((ROOT / run / "stats").glob("*.json"))}


def main() -> None:
    stats = {run: load(run) for run, _ in RUNS}
    labels = [label for _, label in RUNS]
    out = ["# Manuscript numbers across detection sets", "",
           "Runs: " + "; ".join(f"`{run}` = {label}" for run, label in RUNS) + ".", ""]
    empty = [label for (run, label) in RUNS if not stats[run]]
    if empty:
        out += [f"No stats files: {', '.join(empty)}.", ""]
    for namespace in sorted(set().union(*stats.values())):
        keys = sorted(set().union(*(stats[run].get(namespace, {}) for run, _ in RUNS)))
        out += [f"## {namespace}", "", "| key | " + " | ".join(labels) + " |",
                "|---|" + "---|" * len(RUNS)]
        for key in keys:
            cells = [fmt(stats[run].get(namespace, {}).get(key)) for run, _ in RUNS]
            out.append(f"| {key} | " + " | ".join(cells) + " |")
        out.append("")
    path = ROOT / "compare" / "comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"written to {path}")


if __name__ == "__main__":
    main()
