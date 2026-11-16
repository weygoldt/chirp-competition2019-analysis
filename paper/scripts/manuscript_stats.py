"""The numbers the manuscript cites, from the analysis modules into the text.

Every analysis script that computes a number the manuscript cites writes it
with ``write_stats(namespace, values)`` to ``$PAPER_OUT/stats/<namespace>.json``
(one file per script, nested dictionaries of numbers). ``collect`` merges the
files of one run into ``manuscript/stats.json``, which the text reads through
``#stat("<namespace>.<key>...")`` (``manuscript/stats.typ``). The file records
where every number came from: per input file its sha256, the script that
wrote it and the commit of that script. ``check`` (run by the manuscript's
``make``) collects again and fails on any difference, so the text never
builds on numbers that no longer match the analysis outputs.

Standard library only: ``check`` runs outside the paper environment.

    python3 paper/scripts/manuscript_stats.py collect rtdetr_gapfill_t080
    python3 paper/scripts/manuscript_stats.py check manuscript/stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PAPER = SCRIPTS.parent
REPO = PAPER.parent
DEFAULT_OUT = REPO / "manuscript" / "stats.json"

# Keys become parts of the dotted paths of #stat("a.b.c"), so no dots.
_KEY = re.compile(r"^[A-Za-z0-9_]+$")


# ------------------------------------------------------------------ writing


def _clean(value, where: str):
    """``value`` as plain JSON: dicts of numbers, lists of numbers, numbers.

    numpy scalars and arrays are converted; NaN, infinities, booleans,
    strings and None are refused, as is a key that is not a plain word.
    """
    if hasattr(value, "tolist"):  # numpy scalar or array
        value = value.tolist()
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not _KEY.match(key):
                raise ValueError(f"{where}: key {key!r} is not a word of letters, digits and _")
            out[key] = _clean(item, f"{where}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_clean(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {value!r} is not a number")
    if not math.isfinite(value):
        raise ValueError(f"{where}: {value} is not finite")
    return value


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(SCRIPTS), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def write_stats(namespace: str, values: dict) -> Path:
    """Write the numbers one analysis script contributes to the manuscript.

    The file is ``$PAPER_OUT/stats/<namespace>.json``. It also records which
    script wrote it (``sys.argv[0]``), the detections of the run, the commit
    of ``paper/scripts`` and whether that directory had uncommitted changes
    (``collect`` refuses such files). A namespace belongs to one script:
    writing another script's namespace raises.
    """
    import paper_config  # the paper environment; only analysis scripts write

    if not _KEY.match(namespace):
        raise ValueError(f"namespace {namespace!r} is not a word of letters, digits and _")
    script = Path(sys.argv[0]).name
    path = paper_config.OUT / "stats" / f"{namespace}.json"
    if path.exists():
        owner = json.loads(path.read_text())["script"]
        if owner != script:
            raise ValueError(f"{path} belongs to {owner}, not {script}")
    record = {
        "namespace": namespace,
        "script": script,
        "detections": paper_config.DETECTIONS,
        "commit": _git("rev-parse", "HEAD"),
        "code_modified": bool(_git("status", "--porcelain", "--", ".")),
        "values": _clean(values, namespace),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"stats: {path}")
    return path


# ------------------------------------------------------------------ collecting


def build(run: str) -> dict:
    """The content of ``stats.json`` for the run ``paper/output/<run>``."""
    stats_dir = PAPER / "output" / run / "stats"
    files = sorted(stats_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no stats files in {stats_dir}")
    inputs, values, detections = {}, {}, set()
    for f in files:
        record = json.loads(f.read_text())
        if record["namespace"] != f.stem:
            raise ValueError(f"{f}: namespace {record['namespace']!r} differs from the file name")
        if record["code_modified"]:
            raise ValueError(f"{f}: written by uncommitted code in paper/scripts "
                             f"({record['script']}); commit and rerun it")
        detections.add(record["detections"])
        inputs[f.relative_to(REPO).as_posix()] = {
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "script": record["script"],
            "commit": record["commit"],
        }
        values[record["namespace"]] = _clean(record["values"], f.name)
    if len(detections) != 1:
        raise ValueError(f"{stats_dir}: files from different detections {sorted(detections)}")
    return {"run": run, "detections": detections.pop(), "inputs": inputs, "values": values}


def collect(run: str, out: Path = DEFAULT_OUT) -> Path:
    out.write_text(json.dumps(build(run), indent=1) + "\n")
    return out


def _leaves(tree, prefix: str = "") -> dict:
    if isinstance(tree, dict):
        return {k: v for key, item in tree.items() for k, v in _leaves(item, f"{prefix}{key}.").items()}
    return {prefix[:-1]: tree}


def check(path: Path) -> list[str]:
    """What differs between ``path`` and a fresh collection of its run."""
    saved = json.loads(path.read_text())
    try:
        fresh = build(saved["run"])
    except (FileNotFoundError, ValueError) as err:
        return [str(err)]
    problems = []
    for name in sorted(saved["inputs"].keys() | fresh["inputs"].keys()):
        old, new = saved["inputs"].get(name), fresh["inputs"].get(name)
        if new is None:
            problems.append(f"missing: {name}")
        elif old is None:
            problems.append(f"new: {name}")
        elif old != new:
            problems.append(f"changed: {name}")
    if saved["detections"] != fresh["detections"]:
        problems.append(f"detections: {saved['detections']} -> {fresh['detections']}")
    old, new = _leaves(saved["values"]), _leaves(fresh["values"])
    for key in sorted(old.keys() | new.keys()):
        if old.get(key) != new.get(key):
            problems.append(f"value {key}: {old.get(key)} -> {new.get(key)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="merge the stats files of a run into stats.json")
    p.add_argument("run", help="output directory under paper/output, e.g. rtdetr_gapfill_t080")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p = sub.add_parser("check", help="fail if stats.json differs from its run's stats files")
    p.add_argument("path", type=Path, nargs="?", default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.command == "collect":
        out = collect(args.run, args.out)
        print(f"{out}: {len(json.loads(out.read_text())['inputs'])} stats files of {args.run}")
        return
    problems = check(args.path)
    if problems:
        run = json.loads(args.path.read_text())["run"]
        print(f"{args.path} does not match the analysis outputs:", *problems,
              f"after rerunning what changed: python3 {Path(__file__).resolve().relative_to(REPO)} collect {run}",
              sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print(f"{args.path}: matches the analysis outputs")


if __name__ == "__main__":
    main()
