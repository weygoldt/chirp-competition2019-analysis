# Status (2026-11-13)

The v2 pipeline is finished: the libraries, the training data, the final
detector and decoder, their evaluations, the human benchmark, and the
production run over the 2019/2020 tube competition recordings
(`/mnt/data1/chirp_v2/detections/competition2019_v2_20261113/`: 50
recordings, none failed). The pretrained models are published as the GitHub
release `models-v1` of chirpdetector. Both packages are documented
(`packages/*/docs`, `make docs`); the results with their reports are in the
[README](../README.md#results), every step to reproduce them in
[experiments/README.md](../experiments/README.md).

## What is final

| | final | where |
|---|---|---|
| chirp template library | v4, 13,548 templates (12,692 pass the default QC), fish-disjoint splits `splits_v1` | `$CHIRP_DATA/templates/v4` |
| background library | v2, 1,342 segments: inactive-period windows, v1 and v2 chirp screens, recording gate | `$CHIRP_DATA/background/v2`, `configs/background/v2.toml` |
| training scenes | train_v4: 3,000 one-minute hybrid scenes (seeds 5000-7999) | `$CHIRP_DATA/scenes/train_v4`, `configs/scenes/train_v4_*.toml` |
| detector dataset | sim_v4, 81,000 tiles (content sha256 `d65588ab3d02…`) | `~/chirp_v2_fast/datasets/sim_v4` |
| detector | dual-head RT-DETRv2-r18, weights `9fcdef3c…`, epoch 16 | `$CHIRP_DATA/runs/dualhead_r18_sim_v4_r1/final/bundle`; release `models-v1` |
| box-only reference | YOLO26s, same data | `$CHIRP_DATA/runs/yolo26s_sim_v4_r1/final/bundle` |
| decoder | weights `c0801c56…`, on the dual-head's boxes; calibrated and raw estimates (the manuscript's chirp-parameter figure uses the calibrated ones) | `$CHIRP_DATA/params/runs/v4-dualhead-sim-v4-r2/bundle`; release `models-v1` |
| post-processing | harmonic copies and duplicates flagged; re-attribution automatic (on for the emitter head) | `chirpdetector.postprocess` |
| production config | store 0.05, operate 0.5, assignment 20 Hz at onset, 10-s track gaps | `configs/detect/competition2019_v2.toml` |
| default detector in code | `dualhead` (training config); YOLO26 is `type = "ultralytics-yolo"` | chirpdetector `docs` branch |

## Decisions

The decisions and their reasons are in [plan.md](plan.md) (1-24); in
short:

1. YOLO26 (AGPL) is acceptable as the first baseline.
2. The author hand-labels emitters with a dedicated tool.
3. Re-detection uses the 2020 production tracks.
4. The 2023 Label Studio labels are dropped.
5. Chirps in the simulator are real templates, not parametric models.
6. The long analysis window stays; shorter windows only as extra channels.
7. Boxes are each chirp's footprint, with visibility-based ignore regions.
8. Reproducing the v1 training set is not a goal.
9. v1 is tagged `paper-chirp-competition-v1` in both repositories.
10. Template handling follows the MSc thesis (baseline-relative, equalised, amplitude from the frequency trace).
11. Scenes are multichannel wav with annotated spectrograms.
12. Click removal becomes a preprocessing step if it leaves chirps intact.
13. Template library: keep the 320-440 Hz group, drop templates < 5 ms, harvest long chirps.
14. Scenes use real background segments (hybrid scenes).
15. Electrode spacing and reference are unknown; detection must not rely on positions or channel order.
16. Everything in physical units; any sampling rate.
17. Click removal always uses the electrode-coincidence guard (>= 9 electrodes).
18. Library v4: fish-first sampling, brief events cut, clean templates only.
19. The detector predicts box and emitter EODf (dual-head); no log-mel models.
20. Hybrid scenes only for training, selection and testing.
21. Background library v2: inactive period, two chirp screens, recording gate.
22. Fish-disjoint template splits; train_v4/sim_v4.
23. Final models: the dual-head detector and its decoder; YOLO26s as box-only reference.
24. Harmonic copies and duplicates are flagged and dropped after assignment, never silently.

## Open items

For the author:

- **Merge order.** The chirpdetector `docs` branch documents the
  pretrained-model interface of the `v2-pretrained` branch (`chirpdetector
  models`, `chirpdetector.pretrained`, `"default"` bundles, the
  `CHIRPDETECTOR_*` variables, the plain `uv pip install
  "git+https://github.com/weygoldt/chirpdetector"` with gridtools as a git
  dependency). Merge `v2-pretrained` first or together, and check those
  commands against it; they were not run (the branch had no commits when
  the docs were written). Until then the fallback install is
  `uv pip install --no-sources "gridtools @ git+…" "chirpdetector[dualhead,label] @ git+…"`.
- **Publish the documentation sites.** GitHub Pages of both packages still
  serves the v1 documentation. After merging, run `mkdocs gh-deploy` in each
  package (command in their `docs/development.md`).
- **Host the libraries.** gridtools needs the template library v4 (and the
  background library v2 for hybrid scenes) to simulate anything; both exist
  only on this machine. A stranger cannot run `gridtools sim run` until they
  are published (e.g. Zenodo), and the shipped scene config should then
  point to the published location.
- **The v1 -> v2 diff** per trial and fish (plan, goal 3 and phase P6) is not
  written yet; the production run it needs is complete.
- **Real emitter ground truth** does not exist; assignment accuracy and
  confidence are simulation numbers.
- **The released bundle's `metrics.json`** holds the evaluation on the 335
  original labels (precision 0.868); the 365-label numbers are in
  `$CHIRP_DATA/eval/real_labels_365`. Re-attaching would change the release
  zip and its checksum.
- **Front-end ablation.** Decision 6 asked whether the extra windows help;
  no ablation report was found among the run directories.
- **Release tags** of both packages (plan P7) and a test of reproducing P4-P6
  from a fresh clone are not done.

Found while documenting (not fixed; code behaviour):

- `chirpdetector run` can write a `candidates.csv` row with a negative
  `t_start_s` (a chirp in the first 0.1 s), which `label-emitters` refuses.
- `label-benchmark build` stores `--scenes` as given, so a relative path
  breaks the candidates file (the labeller resolves it against the file's
  directory). Absolute paths work.
- `chirpdetector render` prints the PNG name with `.00s.png` while the file
  is written without it (`Path.with_suffix` on a name containing `.00s`).
- `python -m gridtools.background review` fails on a library with no
  segments (no `review/index.html`).
- The `[optim]` defaults of the detector training config are tuned for
  YOLO26, now that the dual-head model is the default; the docs and the
  docstring point to `examples/train_dualhead.toml`.
- gridtools' shipped `default.toml` still uses background library v1 at
  `prob = 0.6`; the final data used v2 at `prob = 1.0`.
- Defaults that are paths of this machine: chirpdetector `[run] out`,
  `label-emitters --out`/`--labeler`, `label-summary`, `evaluate params
  --library`/`--recordings`, `data build-real` (the four recordings);
  gridtools `[chirps] library`, `[background] library`, `sim run --out`,
  `sim compare --calibration`, the background source roots.
