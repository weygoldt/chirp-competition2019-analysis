"""Re-run only the oracle check of eval_postprocess.py (model independent)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/: paths.py
from paths import FAST  # noqa: E402

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from eval_postprocess import _scene_oracle  # noqa: E402
from chirpdetector.data import load_dataset  # noqa: E402

ds = load_dataset(str(FAST / "datasets/sim_v4"), check_files=False)
sources = []
for i in ds.manifest.image_ids("test"):
    s = ds.manifest.image(i).source
    if s not in sources:
        sources.append(s)
result = _scene_oracle(ds, sources)
for run in sys.argv[1:]:
    Path(run, "eval_postprocess", "oracle_true_boxes.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
print(json.dumps(result, indent=1, default=str))
