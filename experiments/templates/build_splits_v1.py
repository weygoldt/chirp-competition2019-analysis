"""Template library v4: fish-disjoint train/val/test splits (splits_v1).

Every unit (competition fish across recordings, else group) goes to exactly
one split; g6f5 (most harvested long chirps) is pinned to train. Writes
splits_v1.json (group -> split, provenance) and splits_v1.md next to the
library. Run once; the detector and the decoder read the same file.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import DATA  # noqa: E402

import logging
import sys
from pathlib import Path

from gridtools.templates import load_library
from gridtools.templates.splits import (
    assign_template_splits,
    load_splits,
    splits_report,
    write_splits,
)

SEED = 0
FRACTIONS = (0.8, 0.1, 0.1)
PINS = {"competition/g6f5": "train"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "templates/v4"
    json_path, md_path = root / "splits_v1.json", root / "splits_v1.md"
    if json_path.exists():
        sys.exit(f"{json_path} exists; the split is written once")
    library = load_library(root, mmap=True)
    splits = assign_template_splits(library, seed=SEED, fractions=FRACTIONS, pins=PINS)
    write_splits(splits, json_path, root)
    loaded = load_splits(json_path)
    md_path.write_text(splits_report(library, loaded, title="Template library v4: splits v1"))
    print(loaded.counts())
