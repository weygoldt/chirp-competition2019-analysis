"""Build template library v4: v3's sources, clean-sample flags, fish-first sampling report."""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from gridtools.templates.merge import build_merged_library
from gridtools.templates.report_merged import write_v4_report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    root = DATA / "templates"
    manifest = build_merged_library(
        root / "v2",
        root / "long_chirps" / "harvest",
        root / "v4",
        review_csv=root / "long_chirps" / "review.csv",
    )
    print(manifest["n_templates"], manifest["n_samples"], manifest["counts"]["cumulative"])
    print(write_v4_report(root / "v4"))
