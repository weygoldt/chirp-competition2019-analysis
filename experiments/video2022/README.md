# 2022 competitions: video tracks (milestones B1, B2 of `docs/spec_2022_dataset.md`)

Outputs under `/mnt/data1/chirp_v2/video2022/`. Use the root environment
(`/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python`).

## B1: the SLEAP export (2026-11-19)

Report: `/mnt/data1/chirp_v2/video2022/b1_sleap/report.md`, with its per-recording
table, 17 figures and example frames. Code: `sleap_nix.py` (reader of the NIX/HDF5
exports, tested), `recordings2022.py`, `b1_assess_sleap.py` (stages: tables,
background, locations, examples, figures, report). In short:

- **Video and recording time.** The LED map (`paper/scripts/led_alignment.py`)
  fits all 19 recordings to about one frame. Every video covers the whole
  electric recording.
- **Coverage.** SLEAP finds both fish in 46 % of the dark-phase and 20 % of
  the light-phase frames; some of these are one fish counted twice.
- **Tracks.** The track ids are fragments (median 1 s).
- **Missing fish.** About half of them are hidden (tube, blocks), the rest
  in view and missed.

## B2: training data for a pose model (2026-11-19)

### Frames and SLEAP's frame numbers

`frames.py` reads frames by index, either sequentially (exact by
construction) or by seeking (fast). `b2_frame_check.py` decoded whole videos
and found seeking exact (12 of 12 frames identical per video).

SLEAP's frame numbers agree with `*converted.mp4` only to within one frame.
For confident, fast-moving fish, the keypoints of SLEAP frame f are darkest
(i.e. on the fish) in video frame f for 61 of 108 instances, and in f + 1
for 37 (2022-05-10). This mix runs through the whole video with no drift; I
did not find the cause. Consequences:

- **Mining** keeps only frames where both fish move at most 0.05 body
  lengths (~10 px) per frame. A one-frame offset then moves a mined label by
  less than the fish's body width.
- **Hand labels** are made on the converted frames themselves, so they are
  exact.
- **B3's comparison with SLEAP** must allow for it: score SLEAP's keypoints
  at offset 0 and at the better of 0 and +1, or on slow fish only.

### Mined frames (`b2_mine.py`)

`/mnt/data1/chirp_v2/video2022/datasets/mined_v1/` holds the Ultralytics YOLO
pose layout, `manifest.csv`, `summary.csv`, `dataset.yaml`, and the check
figures `check_crops.png` and `check_coverage.png` (`b2_check_dataset.py`).

A frame is mined when SLEAP found exactly the two fish:

- each fish with instance, head, centre and tail scores >= 0.5 and head-tail
  >= 0.5 body lengths;
- the two more than 2 body lengths apart, with no shared keypoint;
- both slow (see above);
- inside the electric recording.

Selection: at most 150 frames per recording, at least 2 s apart, spread
round-robin over light/dark x 15-min block x the arena cells of the two
fish. Labels: SLEAP's three keypoints (visible), and a box made of the
keypoints padded by 0.1 body lengths. The same box rule applies to
hand-labelled frames.

Splits by recording (`b2_mine.SPLITS`):

| split | recordings | use |
|---|---|---|
| test | 2022-05-11 (pair 1b), 2022-06-14 (6c), 2022-07-01 (4c) | never mined; hand-labelled evaluation (blind) |
| val | 2022-06-13 (5c), 2022-06-22 (3b) | mined + hand-labelled hard frames, for model selection |
| train | the other 14 | mined + hand-labelled hard frames |

The test recordings share their fish with other bouts of the same pairs, so
the test measures new recordings and days, not new individuals. That is the
use case: the model will run on these 19 recordings.

### The labelling session (`b2_session.py`, `pose_labeller/`)

Session: `/mnt/data1/chirp_v2/video2022/labels/b2_v1/`. It holds
`items.json` (the queue), `images/`, and `truth.csv` (recording, frame,
stratum and SLEAP's instances per item; never served). Your labels go to
`labels.jsonl`.

- **Test items, blind (no points shown):** per test recording, 20 frames
  drawn uniformly over the electric recording (an unbiased sample, about a
  quarter of them in the light) and up to 18 hard frames.
- **Hard items, with SLEAP's points shown to correct:** per train / val
  recording, 2 frames from each hard stratum. The strata are: a dropout of
  >= 1 s, 0.5 s in, by where the lost fish was last seen (tube, block, edge,
  open water); two instances less than 1 body length apart; a random
  light-phase frame.

**Rule (yours, 2026-11-19):** label every visible fish. A partly hidden fish
is labelled by its visible part (e.g. only the head out of the tube; skip
the hidden points with `s`). A fish you cannot see is not labelled.

**The "centre" point is SLEAP's, not the middle of the body.** In the mined
frames it lies 18 % of the way from snout to tail (median of 3,098 fish;
10th-90th percentile 15-21 %): just behind the head, at the pectoral fins.
The hand labels use the same point, so mined and hand labels agree
(`check_crops.png` shows it).

Start, label, stop:

```sh
experiments/video2022/serve_b2.sh                  # user service pose-labeller-b2, port 8766
ssh -N -L 8766:localhost:8766 polarbear            # on the laptop, then open http://localhost:8766/
systemctl --user stop pose-labeller-b2             # stop (labels are saved at every Enter)
```

Keys are in the page's help panel (`h`):

- click places the active fish's next point (head, then centre, then tail);
- `s` skips a point that is not visible, `n` starts a new fish;
- drag moves a point, right click deletes one, `u` undoes;
- Enter marks the frame done, `x` marks it unusable;
- the arrows move to the previous / next frame, the wheel zooms;
- `c` turns the contrast on or off.

Tests: `test_pose_labels.py`, `test_labeller_server.py`, `test_labeller_page.py`
(runs `pose_labeller/test_labeller.mjs` under node: clicks, skips, drags,
deletes, undo and saves of the real page against a fake DOM), `test_frames.py`,
`test_sleap_nix.py`.
