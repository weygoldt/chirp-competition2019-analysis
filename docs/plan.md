# Plan: chirp pipeline v2 (gridtools + chirpdetector)

Status: draft, 2026-11-12. Findings behind each decision: `docs/audit-2026-11.md`.

## Goals and success criteria

1. **Detector.** Beats v1 on held-out real recordings. v1 baseline (335 hand
   labels, threshold 0.5): precision 1.00, recall 0.764, F1 0.866. Compared on
   the same matching rule and on an enlarged real test set.
2. **Assignment.** Metrics measured on the exact weights that are deployed,
   on held-out simulations stratified by difficulty (ΔF, distance, SNR,
   near-simultaneous chirps) and on real chirps with emitter labels (to be made).
3. **Re-detection** of the 33 trials with full provenance, plus a v1 → v2 diff
   (counts and rates per trial and fish) that shows which paper results move.
4. **Two packages a stranger can use:** install with `uv`, simulate, build a
   dataset, train, evaluate, detect, each in one command, documented and tested.

Reproducing the v1 training set is explicitly not a goal. The new set has to be
more correct (physically sound simulation, exact labels, no split leakage) and
yield a better model on real data.

## Principles

- **One definition per transform.** Spectrogram front end, chirp → box
  geometry and assignment features each exist once and are used by dataset
  building, training and inference (removes D3, N13, N17 by construction).
- **Models ship as bundles.** Weights + front-end parameters + class map +
  thresholds + training-data manifest hash + metrics on the locked test set,
  all keyed by the weights' sha256. Reported metrics can only come from the
  shipped weights (removes D1, N1).
- **Split by source, never by sample.** Groups are simulation seeds and real
  recordings; clean/noisy variants of one scene stay in one group (removes N2,
  D2's leakage).
- **Real data is the arbiter; simulation is the training signal.** A locked
  real test set is never used for model or hyperparameter selection.
- **Every random draw is seeded** from one `numpy.random.SeedSequence` per
  scene; configs, seeds, git SHAs and package versions are written next to
  every output.
- **Tests check physics and geometry, not only shapes.**

## Layout

```
chirp_competition_upgrade/      uv workspace root (not a library)
  packages/gridtools/           I/O + simulator + template curation
  packages/chirpdetector/       front end, datasets, models, training, evaluation, inference
  configs/                      scene, dataset and training configs used for the paper run
  scripts/                      thin orchestration (batch simulation, re-detection of the 33 trials)
  docs/                         audit, plan, decision records
data → /mnt/data1/chirp_v2/     simulations, rendered datasets, run dirs, bundles (not in git)
```

Python: `requires-python >= 3.13` for both libraries (SPEC 0), development and
lock on 3.14. Dependencies rebuilt from what the surviving code imports, at
current releases (numpy 2.5, scipy 1.18, torch 2.15, pydantic 2.14, …), locked
once at the workspace root.

### gridtools v2

- `gridtools.io`: recordings via thunderlab/audioio (raw + `fishgrid.cfg`,
  including electrode geometry), wavetracker tracks (same on-disk format),
  audian label CSVs. Frozen dataclasses with explicit `validate()`; units in
  field names; no pydantic in the data path.
- `gridtools.sim`: scene config → seeded generation:
  - population: baseline EODf from a distribution fitted to the 33 trials'
    tracks; shared temperature drift + individual drift; small ΔF pairs and
    track crossings deliberately included;
  - signals: parametric chirp families (small/type-2, big/type-1) with
    excursion, duration, shape, undershoot and amplitude dip fitted to the
    curated template library and real labels; template mode kept as an
    alternative, sampling without duplicate inflation and logging template IDs;
    rises; chirp timing with bursts and responses so near-simultaneous chirps
    occur;
  - EOD: phase-continuous harmonic series with per-fish harmonic amplitudes
    and phases (no thunderfish dependency);
  - movement: vectorised correlated random walk with rest states, exporting
    heading;
  - field: dipole with orientation and a falloff exponent fitted to real
    per-electrode power profiles (`sign_v`); geometry from real `fishgrid.cfg`
    (3×3 and 15-electrode grids);
  - noise: independent coloured noise per electrode, 50 Hz and harmonics,
    gain mismatch, dead/noisy electrodes, clipping; hybrid mode adds real
    background with geometry preserved and controlled SNR;
  - tracks: ground-truth tracks plus optional realistic tracks obtained by
    running wavetracker on the simulated audio and matching to ground truth;
  - output per scene: audio, `ground_truth.parquet` (one row per chirp/rise:
    time, emitter, baseline, excursion, duration, AM, family, template ID),
    tracks in wavetracker format, electrode positions, fish paths,
    `scene.json` (resolved config, seed, versions).
- `gridtools.templates`: rewrite of chirp extraction; curates the surviving
  9,082 clustered templates (dedup, beat-ripple check, provenance flag
  "source recording lost").
- CLI: `gridtools sim run CONFIG --seeds 0:N --out DIR`, `gridtools sim plot SCENE`.

### chirpdetector v2

- `chirpdetector.spectrogram`: the single front end. GPU STFT; multi-resolution
  channels (e.g. long/medium/short windows on one grid) instead of three
  identical channels; electrode aggregation in power, not dB; fixed dB range
  relative to a robust per-window reference instead of per-tile min–max.
  Parameters live in the bundle. Multi-resolution is a hypothesis to test by
  ablation, not an assumption.
- `chirpdetector.data`: chirp → box geometry (one function, in s and Hz),
  dataset builders from simulated scenes and real audian labels, rendered with
  the front end into a detector dataset with a manifest; grouped splits.
- `chirpdetector.models`: detector backend behind a small protocol (default
  ultralytics YOLO26, NMS-free; see decision 1); assignment model in PyTorch.
- `chirpdetector.training`:
  - `train detector`: wraps the ultralytics trainer with our dataset, a
    physically meaningful augmentation policy (no vertical flips, no HSV;
    domain randomisation happens in signal space in the simulator), grouped
    k-fold CV, final fit on all training groups;
  - `train assignment`: plain PyTorch loop, fresh model and optimizer per fold,
    early stopping on a validation group, AMP;
  - every run writes a run dir: resolved config, git SHAs of both packages,
    `uv.lock` hash, dataset manifest hash, GPU, curves (CSV + TensorBoard),
    weights, bundle.
- `chirpdetector.evaluation`:
  - detection: one-to-one matching in time and frequency (port of
    `detector_benchmark`), P/R/F1 and PR curves across thresholds, exact
    binomial CIs and bootstrap over regions;
  - assignment: accuracy and calibration (ECE), reject curves, stratified by
    ΔF, distance, SNR and chirp overlap;
  - end-to-end: chirp events with correct emitter;
  - writes `metrics.json` into the bundle it evaluated.
- `chirpdetector.inference`: streams recordings in overlapping chunks;
  global deduplication in absolute time across chunks (D4, N11); no tail loss
  (N12); candidates = tracks alive at the chirp time (D5); a real rejection
  threshold (N4); one output table with schema and provenance, plus legacy
  `chirp_times`/`chirp_ids` arrays for the paper analysis code.
- CLI: `chirpdetector build-dataset | train | evaluate | detect`.

### Assignment model

Each detected box is scored against every track alive at its time:

- **frequency evidence:** the spectrogram patch rectified to the candidate's
  baseline (ΔF axis), so the emitter's baseline band losing power during the
  chirp is visible (the cue the v1 power-trough features approximated);
- **spatial evidence:** per-electrode power of the chirp region vs the
  candidate's per-electrode power just before the chirp (needs the dipole
  field, G3);
- **context:** ΔF to the other candidates, track confidence when available
  (`wavetracker-confidence`).

A shared encoder scores each candidate, then a softmax over candidates plus a
"none" option gives a calibrated emitter probability with rejection. Training
oversamples hard cases (small ΔF, crossings, overlapping chirps, fish
equidistant to electrodes).

Considered and deferred: one track-conditioned network that detects and
assigns at once. Two stages stay separately measurable, match the Methods,
and the assignment stage is already track-conditioned.

### Real-data ground truth

- **Detection:** keep the 4 labelled recordings (335 chirps) as the locked test
  set and enlarge it with audian (specdetector workflow), prioritising
  close-ΔF and overlapping-chirp regions. The 2023 Label Studio labels
  (537 images, 7,416 boxes, 17 recordings) become real training/validation
  data after conversion to s/Hz, if the pixel → time/frequency mapping can be
  recovered.
- **Emitter:** none exists (audit summary 1). Options, ranked by
  cost/benefit:
  1. hand-label the emitter on a stratified real subset where a human can
     decide;
  2. hybrid scenes: simulated chirping fish added to real recordings;
  3. superposition of single-fish recordings on the same grid, if such data
     exist (exact by linearity of the field);
  4. 2022 trials with video + SLEAP: compare the chirp's spatial power
     centroid with video positions.

## Phases

Each phase ends with commits on `v2` in the affected submodule, a passing CI,
and a short entry in `docs/`.

| Phase | Deliverable | Done when |
|---|---|---|
| P0 Foundations | Dead code deleted (one commit per repo); PEP 621 + uv_build, src layout; workspace wiring (editable); ruff, pyright, pytest, pre-commit, GitHub Actions (CPU) | `uv sync` at root; CI green on 3.13/3.14 |
| P1 Data layer | `gridtools.io` for recordings, geometry, tracks, labels | loads all 33 trials and the 4 labelled recordings; round-trip tests |
| P2 Simulator | `gridtools.sim` as above | tests: instantaneous frequency of synthesized EOD = f(t); ground-truth chirp time and box enclose the chirp energy; dipole falloff; bit-exact seed reproducibility; chirps at scene edges; sim-vs-real diagnostics report |
| P3 Chirp model | fitted chirp families + curated template library | parameter distributions sim vs real plotted and reviewed |
| P4 Detector | front end, dataset builder, training + evaluation subpackages, first bundle | grouped CV + locked real test; beats v1 (F1 0.866 at P ≈ 1.00) or a documented reason why not; ablation of front-end choices |
| P5 Assignment | features, model, training, evaluation, real emitter labels | stratified sim metrics + real-subset metrics from the shipped bundle |
| P6 Re-detection | streaming inference, output schema, run on 33 trials | v1 → v2 diff report per trial and fish |
| P7 Release | docs site, README quickstart, tagged releases, bundles archived (GitHub release or Zenodo) | a fresh clone reproduces P4–P6 from configs |

Order: P0 → P1 → P2/P3 (interleaved) → P4 → P5 → P6 → P7. P4 can start on
real labels and an early simulator once P1 exists.

## Constraints

- GPU (RTX 4080, 16 GB) is shared with other jobs (eeltracker,
  wavetracker-salience were using ~7.5 GB on 2026-11-12).
- All bulk data on `/mnt/data1`; `/home` and `/mnt/data2` are nearly full.
- `main` of both submodules stays untouched until v2 is ready; v1 stays
  available as tag `paper-chirp-competition-v1` (chirpdetector) and should get
  the same tag on gridtools `df7f504`.
