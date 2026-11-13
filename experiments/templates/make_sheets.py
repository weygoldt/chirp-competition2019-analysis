
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA, REC_2019  # noqa: E402

import logging
from pathlib import Path
from gridtools.templates.harvest_plots import write_contact_sheets, write_rejection_examples

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = DATA / "templates/long_chirps"
    pages = write_contact_sheets(out)
    print(len(pages), "pages")
    print(write_rejection_examples(out, str(REC_2019)))
