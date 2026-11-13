
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

import logging, matplotlib
matplotlib.use("Agg")
from gridtools.templates.report_merged import write_merged_report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(write_merged_report(str(DATA / "templates/v3"), str(DATA / "templates/long_chirps")))
