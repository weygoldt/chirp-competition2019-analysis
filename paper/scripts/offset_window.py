"""The window around chase offsets in which chirps are counted (the "offset window").

Several analyses count chirps from shortly before to some seconds after each
chase offset: the per-chase table of the receiver-effect, coupling and
random-effects models (``summarize_submissive_chirp_effect_on_dominant``), the
per-chase table of the hurdle model and the offset logistic regressions
(``summarize_chirp_event_correlation_around_chasing``), and the models,
figures and summaries built on them. The window is one parameter, taken from
the data by a stated rule; no script has its own numbers.

Rule (the author's decision, 2026-11-19): the window is the stretch around
chase offsets in which the losers' chirp rate is above the null's 95 % band,
i.e. the region of the strongest loser peak around chase offsets
(``event_rates.loser.chase_offset.peak.region_s``, written by
``plot_event_time_rates_winner_loser_chirp_src_paper.py``), with both ends
rounded to the nearest whole second (an end exactly half-way rounds outwards,
widening the window). The window must contain the offset (left < 0 < right):
the analyses split it there into the end of the chase and the time after it.

The window is derived once, from the primary run (``paper_config.PRIMARY``,
the operating threshold 0.8), and the sensitivity runs (thresholds 0.5 and
0.9) use the same window, so that differences between the thresholds reflect
the detections, not the windows. (The same rule on the curve of an earlier
rerun, [-3.1, +13.2] s, gave the -3 to 13 s window, ``m3p13``, that the
scripts had as constants before.)

    paper/run.sh offset_window.py           # chain rates, after the event-rate plot
    paper/run.sh offset_window.py --check   # print the window the analyses will use

The step writes ``$PAPER_OUT/stats/offset_window.json`` (``write_stats``):
the window, its width, the unrounded region and the rule's numbers. In the
primary run it derives the window from that run's event rates; in a
sensitivity run it records the primary's window and fails if the primary's
has not been written yet. Every analysis reads the window with
``offset_window()``, which reads the primary run's file (whatever run is
being analysed) and raises if it is missing or no longer matches the
primary's event rates.

Testing only: ``PAPER_OFFSET_WINDOW=-3,13`` makes ``offset_window()`` return
that window (seconds relative to the offset) without reading any file, e.g.
to rerun the analyses with the old window in a scratch output directory and
compare. ``manuscript_stats`` marks stats written while it is set, and
``collect`` refuses them, so such numbers never reach the manuscript.

Standard library only (and ``paper_config``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import paper_config

NAMESPACE = "offset_window"
# Where the rule reads the region: event_rates.<this path>.
REGION_KEYS = ("loser", "chase_offset", "peak", "region_s")
# The null band of the event-rate plot (2.5-97.5 % of the bootstrapped null)
# and the rounding of the rule; recorded with the window.
NULL_BAND = 0.95
ROUNDING_S = 1
OVERRIDE_ENV = "PAPER_OFFSET_WINDOW"


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _round_half_outward(x: float) -> int:
    # The region lies on the 0.1 s grid of the rates; drop its float noise
    # (10.600000000000577) first, so that a half-way end is a real tie.
    x = round(x, 6)
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def derive_window(region) -> tuple[int, int]:
    """The window (left, right) in whole seconds relative to the chase offset,
    from the unrounded region ``[start, end]`` by the rule."""
    if not (isinstance(region, (list, tuple)) and len(region) == 2
            and all(_is_number(x) for x in region)):
        raise ValueError(f"region {region!r} is not [start, end] in seconds")
    start, end = region
    if not start < end:
        raise ValueError(f"region {region!r} does not end after it starts")
    left, right = _round_half_outward(start), _round_half_outward(end)
    if not left < 0 < right:
        raise ValueError(
            f"region {list(region)} s rounds to ({left}, {right}) s, which does not contain "
            "the chase offset; the analyses split the window at the offset")
    return left, right


def window_stats(region) -> dict:
    """What the step writes (and the reader expects): the window, its width,
    the region it came from and the rule's numbers."""
    left, right = derive_window(region)
    return {"window_s": [left, right], "width_s": right - left,
            "region_s": [float(region[0]), float(region[1])],
            "rule": {"null_band": NULL_BAND, "rounding_s": ROUNDING_S}}


def _read_record(path: Path, what: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing ({what})")
    return json.loads(path.read_text())


def region_from_event_rates(path: Path) -> list:
    """``event_rates.loser.chase_offset.peak.region_s`` of an event_rates stats file."""
    record = _read_record(path, "written by plot_event_time_rates_winner_loser_chirp_src_paper.py, "
                                "chain rates")
    node = record.get("values")
    for key in REGION_KEYS:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"{path} has no event_rates.{'.'.join(REGION_KEYS)} "
                             "(no loser peak above the null around chase offsets?)")
        node = node[key]
    return node


def primary_window_stats(primary_out: Path | None = None) -> dict:
    """The primary run's window file (in ``primary_out``, default
    ``paper_config.PRIMARY_OUT``), checked against the primary's event rates."""
    primary_out = paper_config.PRIMARY_OUT if primary_out is None else primary_out
    path = primary_out / "stats" / f"{NAMESPACE}.json"
    record = _read_record(
        path, f"the window is derived in the primary run: PAPER_DETECTIONS={paper_config.PRIMARY} "
              "paper/run.sh offset_window.py, or its chain rates")
    if record.get("namespace") != NAMESPACE or record.get("detections") != paper_config.PRIMARY:
        raise ValueError(f"{path} is not the window of the primary run {paper_config.PRIMARY} "
                         f"(namespace {record.get('namespace')!r}, detections {record.get('detections')!r})")
    expected = window_stats(region_from_event_rates(primary_out / "stats" / "event_rates.json"))
    if record.get("values") != expected:
        raise ValueError(
            f"{path} ({record.get('values')}) no longer matches the primary's event rates "
            f"({expected}); rerun offset_window.py in the primary run, then everything that uses it")
    return expected


def parse_window(text: str) -> tuple[float, float]:
    """``"-3,13"`` -> (-3.0, 13.0); the window must contain the offset."""
    try:
        left, right = (float(x) for x in text.split(","))
    except ValueError:
        raise ValueError(f"{text!r} is not 'left,right' in seconds, e.g. -3,13") from None
    if not (_is_number(left) and _is_number(right) and left < 0 < right):
        raise ValueError(f"window {text!r} does not contain the chase offset (need left < 0 < right)")
    return left, right


def offset_window(primary_out: Path | None = None) -> tuple[float, float]:
    """The window (left, right) in seconds relative to the chase offset, e.g.
    (-5.0, 11.0): chirps are counted from offset + left to offset + right.

    Read from the primary run's ``stats/offset_window.json``, also when
    another run is being analysed; raises if that file is missing, malformed
    or does not match the primary's event rates. ``PAPER_OFFSET_WINDOW``
    (testing only) replaces it.
    """
    override = os.environ.get(OVERRIDE_ENV)
    if override is not None:
        window = parse_window(override)
        print(f"offset window {window} s from {OVERRIDE_ENV} (testing only; collect refuses "
              "the stats of this run)", file=sys.stderr)
        return window
    left, right = primary_window_stats(primary_out)["window_s"]
    return float(left), float(right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="only print the window the analyses will use (fails if there is none)")
    args = parser.parse_args()

    if args.check:
        print(f"offset window: {offset_window()} s around chase offsets")
        return
    if OVERRIDE_ENV in os.environ:
        raise SystemExit(f"{OVERRIDE_ENV} is set: this step writes the window by the rule "
                         "and takes no override")
    from manuscript_stats import write_stats

    if paper_config.DETECTIONS == paper_config.PRIMARY:
        values = window_stats(region_from_event_rates(paper_config.OUT / "stats" / "event_rates.json"))
        source = "this run's event rates (primary)"
    else:
        values = primary_window_stats()
        source = f"the primary run {paper_config.PRIMARY}"
    left, right = values["window_s"]
    print(f"offset window: {left:+d} to {right:+d} s around chase offsets, from {source}: "
          f"loser peak above the null from {values['region_s'][0]:+.2f} to "
          f"{values['region_s'][1]:+.2f} s")
    write_stats(NAMESPACE, values)


if __name__ == "__main__":
    main()
