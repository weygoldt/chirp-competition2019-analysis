# new_chirpextractions (inputs of Fig. 1)

The arrays live in `paper/inputs/new_chirpextractions/` (git-ignored, 167 MB;
GitHub refuses files over 100 MB). This README and their sha256 below are tracked.

Verbatim copies of three arrays of the old analysis that cannot be regenerated
(the raw recordings of their upstream chain are gone). They hold chirp
frequency excursions extracted from recordings that are not part of the
competition data set; Fig. 1 (`plot_what_is_a_chirp_and_why_use_spetrograms`)
picks two templates from them. They do not depend on any competition chirp
detections.

Source: `~/wrk/analyses/chirp_competition/data/interrim/new_chirpextractions/`
(repository at commit d040123), copied 2026-11-13.

Read by `paper/scripts/collect_chirp_overview_plot_data.py`.

sha256:

```
ac6ac98e49da9542ae18499eaf7e7cc8d7cecf96c279ad1790dc640a4fe602e4  flattened_chirps.npy
f4d9f5d0fe9ee9349cf74aba47a9357389e6b58daf287aa8557fd989dc122dac  flattened_heights.npy
e54c06b958fbeadbd90204464bfa915ba686d2a16c6a867838c24fa2fbf69da7  flattened_widths.npy
```

# chase_validation (the author's re-labelled chases)

`paper/inputs/chase_validation/` (git-ignored): a frozen copy of the chase review session `offsets_v1`
(`/mnt/data1/chirp_v2/labels/chase_review_2019/offsets_v1/`): `session.json` (the BORIS chases, LED maps
and video facts the review tool was built from) and `revisions.jsonl` (every save of the author's review;
the latest save of a chase counts). Read by `paper/scripts/chase_validation.py` (chain `validation`).

Frozen 2026-11-24 (last save 10:15:53 UTC; 116 saves of 100 reviewed chases, 85 of them whole-window
reviews). Read-only snapshot beside the labels: `offsets_v1/frozen_2026-11-24/`.

sha256:

```
eacf1f8129cf0d8d24b3230f61e04035530494f454eb528e3b0bcaffef8b74cf  session.json
1c000006918e5d97538fcc7a2f509fbc45351d1765fc06eb740204644ac6fbab  revisions.jsonl
```
