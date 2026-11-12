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
  - signals: real chirps from the curated template library (decision 5),
    sampled without duplicate inflation, every placed chirp logging its
    template ID; amplitude modulation from real amplitude traces where the
    library has them; rises; chirp timing with bursts and responses so
    near-simultaneous chirps occur;
  - re-rendering: any fish's signal can be re-rendered deterministically for a
    time span, with or without a given chirp, on any electrodes (needed for
    ground-truth boxes, see `chirpdetector.data`);
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
- `gridtools.templates`: a versioned chirp template library built from the
  surviving arrays (9,082 clustered templates; 13,015 extracted chirps):
  dedup, beat-ripple and artefact checks, per-template metadata (excursion,
  duration, family), provenance flag "source recording lost". If the library
  turns out too narrow, it can be extended with chirps extracted from
  well-separated, high-SNR fish in the competition recordings, with
  provenance.
- CLI: `gridtools sim run CONFIG --seeds 0:N --out DIR`, `gridtools sim plot SCENE`.

### chirpdetector v2

- `chirpdetector.spectrogram`: the single front end. GPU STFT; multi-resolution
  channels (e.g. long/medium/short windows on one grid) instead of three
  identical channels; electrode aggregation in power, not dB; fixed dB range
  relative to a robust per-window reference instead of per-tile min–max.
  Parameters live in the bundle. Shorter windows trade frequency resolution
  for time resolution, and frequency resolution is what separates close-ΔF
  fish. The long window (v1: 4096 points, 4.9 Hz) therefore stays as one
  channel; shorter windows are added next to it, not instead of it.
  Assignment features use the long window and the tracks. Whether the extra
  channels help is decided by ablation on the real test set.
- `chirpdetector.data`: ground-truth boxes, dataset builders from simulated
  scenes and real audian labels, rendered with the front end into a detector
  dataset with a manifest; grouped splits.
  - Ground-truth boxes come from the chirp's footprint in the actual
    spectrogram, not from template width/height plus fixed padding
    (v1: ±(width + 0.8·nfft/fs)/2 in time, baseline + trough − 5 bins to
    baseline + height + 5 bins in frequency). For each chirp, the emitter's
    signal is re-rendered over a short span with and without that chirp and
    passed through the same front end as the detector. The footprint is the
    set of time–frequency cells where the two differ by more than a threshold
    relative to the footprint's peak. This includes the power drop at the
    emitter's baseline during the chirp and the smear of the analysis
    window, at whatever window length the bundle uses.
  - Visibility: footprint power is compared with the mixture (other fish +
    noise) in the same cells, giving a per-chirp SNR. Chirps above a
    visibility threshold become boxes; chirps below it become ignore regions
    (neither positive nor negative), so invisible chirps don't train the
    model to hallucinate and missed-but-real chirps don't count as negatives.
  - Edge cases are handled explicitly: chirps cut by window edges (kept if
    enough of the footprint is visible, otherwise ignore), overlapping chirps
    of different fish (one footprint per fish), harmonics (fundamental only by
    default; configurable).
  - Per-chirp SNR, ΔF to the nearest fish and overlap flags are stored with
    every label, for stratified evaluation.
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
- CLI: `chirpdetector build-dataset | train | evaluate | detect | label-emitters`.

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
  close-ΔF and overlapping-chirp regions. The 2023 Label Studio labels are
  not used (labelled on plain PNGs; mapping to time/frequency not
  recoverable).
- **Emitter:** none exists (audit summary 1); no single-fish or empty-tank
  recordings exist either. Plan:
  1. hand labels from a dedicated, keyboard-driven labelling tool
     (`chirpdetector label-emitters`, local browser UI). One chirp per screen:
     multi-resolution spectrogram snippet with candidate tracks overlaid in
     colour, per-electrode power maps of the chirp vs each candidate's
     baseline, keys 1–9 for the emitter, 0 for "can't tell", undo. Queue
     stratified by ΔF and recording; writes audian-compatible CSV plus an
     emitter column. Labelling can start before the new detector exists, on
     v1 detections (`chirpdetector_bboxes.csv`) of the 33 trials; the model
     never trains on these labels, so any trial can be used;
  2. hybrid scenes: simulated chirping fish added to real recordings, with
     geometry preserved and controlled SNR;
  3. optional: 2022 trials with video + SLEAP, comparing the chirp's spatial
     power centroid with video positions.

## Phases

Each phase ends with commits on `v2` in the affected submodule, a passing CI,
and a short entry in `docs/`.

| Phase | Deliverable | Done when |
|---|---|---|
| P0 Foundations | Dead code deleted (one commit per repo); PEP 621 + uv_build, src layout; workspace wiring (editable); ruff, pyright, pytest, pre-commit, GitHub Actions (CPU) | `uv sync` at root; CI green on 3.13/3.14 |
| P1 Data layer | `gridtools.io` for recordings, geometry, tracks, labels | loads all 33 trials and the 4 labelled recordings; round-trip tests |
| P1b Labelling tool | `chirpdetector label-emitters` on v1 detections | the author can label at a sustained pace; labels round-trip to CSV |
| P2 Simulator | `gridtools.sim` as above | tests: instantaneous frequency of synthesized EOD = f(t); chirp re-rendering is exact; dipole falloff; bit-exact seed reproducibility; chirps at scene edges; sim-vs-real diagnostics report |
| P3 Template library | curated, versioned real-chirp library | dedup and quality report; per-template metadata; distributions reviewed |
| P4 Detector | front end, footprint-based ground-truth boxes, dataset builder, training + evaluation subpackages, first bundle | box tests (footprint contains the chirp energy; invisible chirps become ignore regions; edge and overlap cases); grouped CV + locked real test; beats v1 (F1 0.866 at P ≈ 1.00) or a documented reason why not; ablation of front-end choices |
| P5 Assignment | features, model, training, evaluation on real emitter labels | stratified sim metrics + real-subset metrics from the shipped bundle |
| P6 Re-detection | streaming inference, output schema, run on 33 trials | v1 → v2 diff report per trial and fish |
| P7 Release | docs site, README quickstart, tagged releases, bundles archived (GitHub release or Zenodo) | a fresh clone reproduces P4–P6 from configs |

Order: P0 → P1 → P1b → P2/P3 (interleaved) → P4 → P5 → P6 → P7. Emitter
labelling with the P1b tool runs in parallel with P2–P4.

## Decisions (2026-11-12)

1. License: ultralytics (AGPL-3.0) is acceptable; YOLO26 is the default
   detector. RF-DETR stays an optional comparison in P4.
2. Emitter ground truth: the author hand-labels with a fast dedicated tool
   (P1b). No single-fish or empty-tank recordings exist.
3. Re-detection uses the 2020 production tracks; wavetracker is not ready.
4. The 2023 Label Studio labels are dropped.
5. Chirps in the simulator are real chirps (templates), not parametric
   models, which tend to be unrealistically smooth.
6. Time–frequency trade-off: the long analysis window stays; shorter windows
   are only added as extra channels if the ablation shows a gain.
7. Ground-truth boxes are derived from each chirp's footprint in the
   spectrogram (counterfactual re-rendering), with visibility-based ignore
   regions.
8. Reproducing the v1 training set is not a goal.
9. gridtools `df7f504` tagged `paper-chirp-competition-v1` (pushed), matching
   chirpdetector; `v2` branches pushed on both.
10. Template handling follows the author's MSc thesis (2026-11-12): chirps are
    baseline-relative (usable on a fish of any EODf); rare large and long
    chirps (up to ~400 Hz and several hundred ms) are kept and oversampled by
    histogram equalization over excursion x duration (10 x 10 bins); the
    chirp's amplitude envelope is its inverted frequency trace scaled by a
    random factor (EOD amplitude between 100% and 20%); the baseline around a
    chirp is crossfaded smooth.
11. Scenes are written as multichannel wav, each with an annotated
    spectrogram (tracks, chirps, emitter EODf).
12. Click removal (port of audian/wavetracker) becomes a chirpdetector
    preprocessing step if it leaves chirps intact.
13. Template library (2026-11-12): the 320-440 Hz group (maybe Eigenmannia)
    is kept; templates shorter than 5 ms are not real chirps and are
    excluded; long chirps (300-450 ms, plateau and slow decay) are harvested
    from the competition recordings, since the chirp-chamber source has none
    above ~150 ms; equalization may be tempered or log-binned, because
    repeats are acceptable while the chirp's context (noise, other fish,
    position in the spectrogram) varies.
14. Background noise (2026-11-12): simulated noise alone is too Gaussian.
    Scenes add real background segments from the 2019 and 2022 competitions
    (and field recordings if usable), scaled over a range of SNRs, as in the
    thesis' hybrid dataset. Segments avoid detected chirps; real fish in them
    become tracked, non-chirping candidates; a hybrid scene takes its grid
    geometry from the background segment (no random electrode subsets).
15. Recording setup (2026-11-12): electrode spacing of the tube competition
    grids is unknown (not the 0.5 m in fishgrid.cfg; likely different in
    2019 and 2020) and the reference electrode position is unknown. The
    simulator draws spacing per scene from a broad range (field falloff
    calibrated in units of spacing) and simulates an in-tank reference as an
    augmentation. Detection and assignment must not rely on absolute
    electrode positions or channel order.
16. Sample-rate agnostic (2026-11-12): every spectrogram, click-removal,
    plotting and footprint parameter is in physical units (s, Hz); the
    sampling rate comes from the recording. Recordings at any rate get the
    same time and frequency resolution (frequency grid equal to within
    rounding), so a model bundle trained at 20 kHz applies at any rate.
17. Click removal always uses the electrode-coincidence guard and refuses
    recordings with fewer electrodes than it protects chirps on (9 at the
    default) and single-channel data; pipelines skip it with a warning.
    Simulations use 1-16 electrodes and varied spacing, so the detector
    learns to reject clicks where removal is not possible.
18. Template library v4 (2026-11-12): all harvested long chirps kept, but
    sampling picks an individual first and then one of its templates, in
    every equalization bin; events with FWHM <= 2 EOD cycles are cut; only
    clean samples: every QC flag (ripple, multi-peak, line in span,
    baseline drift, low SNR, interrupted decay) excludes a template by
    default.

## Constraints

- GPU (RTX 4080, 16 GB) is shared with other jobs (eeltracker,
  wavetracker-salience were using ~7.5 GB on 2026-11-12).
- All bulk data on `/mnt/data1`; `/home` and `/mnt/data2` are nearly full.
- `main` of both submodules stays untouched until v2 is ready; v1 stays
  available as tag `paper-chirp-competition-v1` on both repos.
