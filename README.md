# chirp_competition_upgrade

Working repository for version 2 of the chirp pipeline: simulating grid
recordings of wave-type electric fish (gridtools), training and evaluating a
chirp detector and an emitter-assignment model on them (chirpdetector), and
re-detecting chirps in the 33 staged competitions of the chirp competition
paper.

- `packages/gridtools`, `packages/chirpdetector`: the two packages, as git
  submodules on their `v2` branches, installed editable into this
  environment.
- `docs/plan.md`: the plan, decisions and phases.
- `docs/audit-2026-11.md`: what was wrong with v1 and which assets survive.

Bulk data (simulations, rendered datasets, training runs, model bundles)
lives on `/mnt/data1/chirp_v2/`, not in git.

## Setup

```sh
git clone --recurse-submodules git@github.com:weygoldt/chirp_competition_upgrade.git
cd chirp_competition_upgrade
uv sync
make check   # lint, format check, type check and tests of both packages
```

The submodules track their `v2` branches. To work in one, check the branch
out first (`git -C packages/gridtools switch v2`); `git submodule update
--remote` moves both to the latest `v2`.
