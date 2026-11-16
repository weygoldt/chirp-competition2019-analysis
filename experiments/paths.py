"""Where the data of this project live: the one place every script reads.

The scripts in ``experiments/`` were run on one machine (an RTX 4080
workstation, "polarbear"); its paths are the defaults below. To run them
elsewhere, set the environment variables (``env.sh`` does the same for the
shell scripts):

``CHIRP_DATA``
    Bulk data root (libraries, scenes, datasets, runs, bundles, labels,
    detections). Default ``/mnt/data1/chirp_v2``.
``CHIRP_FAST``
    Where the detector datasets are read from (``datasets/sim_v4``). Default
    ``$CHIRP_DATA`` (the data drive, since 2026-11-16). Training and
    evaluation ran from a copy on a fast disk (``~/chirp_v2_fast``, I/O on the
    hard disk starved training); set this to such a copy to retrain fast.
``CHIRP_REC_2019``
    The 2019/2020 tube competition recordings (fishgrid raw, wavetracker
    tracks, hand labels). Default ``/mnt/data2/2019_tube_competition``.
``CHIRP_REC_2022``
    The 2022 tube competition recordings (``raw/`` and ``interrim/`` with
    the tracks). Default ``/mnt/data2/2022_tube_competition``.
``CHIRP_V1_BENCHMARK``
    The frozen v1 detector benchmark of the chirp competition analysis
    (its ``results/boxes`` hold v1's boxes on the labelled regions).
    Default ``~/wrk/analyses/chirp_competition/detector_benchmark``.
``GRIDTOOLS_TEMPLATES``
    The chirp template library gridtools draws from (gridtools has no default
    since it went public). Default ``$CHIRP_DATA/templates/v4``.
``CHIRP_FIELD_FLONA2023``, ``CHIRP_FIELD_FLONA2025``, ``CHIRP_FIELD_IRIRI2026``
    Field recordings for the background libraries (``--root-*`` of
    ``python -m gridtools.background build``). Defaults: the paths on polarbear.

Import from a script with::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import DATA, FAST, REC_2019, REC_2022
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DATA",
    "EXPERIMENTS",
    "FAST",
    "FIELD_FLONA2023",
    "FIELD_FLONA2025",
    "FIELD_IRIRI2026",
    "REC_2019",
    "REC_2022",
    "REPO",
    "TEMPLATES",
    "V1_BENCHMARK",
]


def _env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


EXPERIMENTS = Path(__file__).resolve().parent
REPO = EXPERIMENTS.parent
DATA = _env("CHIRP_DATA", "/mnt/data1/chirp_v2")
FAST = _env("CHIRP_FAST", str(DATA))
REC_2019 = _env("CHIRP_REC_2019", "/mnt/data2/2019_tube_competition")
REC_2022 = _env("CHIRP_REC_2022", "/mnt/data2/2022_tube_competition")
V1_BENCHMARK = _env(
    "CHIRP_V1_BENCHMARK", "~/wrk/analyses/chirp_competition/detector_benchmark"
)
TEMPLATES = _env("GRIDTOOLS_TEMPLATES", str(DATA / "templates" / "v4"))
FIELD_FLONA2023 = _env("CHIRP_FIELD_FLONA2023", "/mnt/data1/efish/fielddata/flona2023")
FIELD_FLONA2025 = _env(
    "CHIRP_FIELD_FLONA2025", "/mnt/data1/efish/fielddata/flona2025_workspace/flona2025"
)
FIELD_IRIRI2026 = _env(
    "CHIRP_FIELD_IRIRI2026",
    "/run/media/weygoldt/data1/efish/fielddata/iriri2026/iriri2026_recordings",
)
