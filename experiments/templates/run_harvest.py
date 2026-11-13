
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: paths.py
from paths import REC_2019  # noqa: E402

import logging, sys, time
from pathlib import Path
from gridtools.templates.harvest import HarvestConfig, run_harvest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    root = REC_2019
    out = Path(sys.argv[1])
    recs = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] == "20" and (p / "fishgrid.cfg").exists())
    t = time.time()
    records = run_harvest(recs, out, HarvestConfig(), individuals_csv=root / "order_meta.csv", workers=6)
    print("done in", time.time() - t, "s;", (records.status == "harvested").sum(), "harvested of", len(records))
    print(records.groupby("status").reason.value_counts())


if __name__ == "__main__":
    main()
