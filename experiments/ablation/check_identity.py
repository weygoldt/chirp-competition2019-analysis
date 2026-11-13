"""Check that arm B's dataset differs from sim_v4 only in image channels 1 and 2.

    uv run python experiments/ablation/check_identity.py A_DATASET B_DATASET [--sample N] [--out JSON]

For every image of B (or a seeded sample of N): the image exists in A with
the same id, source, split, geometry and label sha256; channel 0 of B equals
channel 0 of A bit for bit; channels 1 and 2 of B equal its channel 0. Also
compares the box tables (all columns) of the shared images. Prints a
summary and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from chirpdetector.data import load_dataset
from chirpdetector.data.contract import read_image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    a = load_dataset(args.a, check_files=False)
    b = load_dataset(args.b, check_files=False)
    ma, mb = a.manifest, b.manifest
    res: dict = {"a": str(args.a), "b": str(args.b)}
    res["spectrogram_a"] = ma.spectrogram.to_dict()
    res["spectrogram_b"] = mb.spectrogram.to_dict()
    res["tiling_a"] = ma.tiling.to_dict()
    res["tiling_b"] = mb.tiling.to_dict()
    sa, sb = ma.spectrogram.to_dict(), mb.spectrogram.to_dict()
    res["spectrogram_fields_differing"] = sorted(k for k in sa if sa[k] != sb.get(k))
    ta, tb = ma.tiling.to_dict(), mb.tiling.to_dict()
    res["tiling_fields_differing"] = sorted(k for k in ta if ta[k] != tb.get(k))
    res["preprocessing_equal"] = ma.preprocessing == mb.preprocessing
    ids_b = [r.id for r in mb.images]
    ids_a = {r.id for r in ma.images}
    res["n_images_a"], res["n_images_b"] = len(ma.images), len(ids_b)
    res["b_ids_missing_in_a"] = sum(i not in ids_a for i in ids_b)
    if len(ids_b) == len(ma.images):
        res["same_id_set"] = set(ids_b) == ids_a
    check = ids_b
    if args.sample is not None and args.sample < len(ids_b):
        rng = np.random.default_rng(0)
        check = sorted(rng.choice(ids_b, size=args.sample, replace=False).tolist())
    meta_bad, ch0_bad, rep_bad, ch12_same_in_a = [], [], [], 0
    for image_id in check:
        ra, rb = ma.image(image_id), mb.image(image_id)
        if (
            ra.source != rb.source
            or ra.group != rb.group
            or ra.label_sha256 != rb.label_sha256
            or ra.geometry != rb.geometry
            or ra.n_chirps != rb.n_chirps
            or ra.n_ignore != rb.n_ignore
            or ma.split_of_group(ra.group) != mb.split_of_group(rb.group)
            or ra.extra != rb.extra
        ):
            meta_bad.append(image_id)
        ia = read_image(a.image_path(image_id))
        ib = read_image(b.image_path(image_id))
        if ia.shape != ib.shape or not np.array_equal(ia[..., 0], ib[..., 0]):
            ch0_bad.append(image_id)
        if not (np.array_equal(ib[..., 1], ib[..., 0]) and np.array_equal(ib[..., 2], ib[..., 0])):
            rep_bad.append(image_id)
        if np.array_equal(ia[..., 1], ia[..., 0]) and np.array_equal(ia[..., 2], ia[..., 0]):
            ch12_same_in_a += 1
    res["n_checked"] = len(check)
    res["metadata_mismatch"] = len(meta_bad)
    res["channel0_mismatch"] = len(ch0_bad)
    res["b_not_replicated"] = len(rep_bad)
    res["a_images_with_identical_channels"] = ch12_same_in_a
    res["examples"] = {"meta": meta_bad[:5], "ch0": ch0_bad[:5], "rep": rep_bad[:5]}
    # box tables of the checked images
    ba = a.boxes[a.boxes["image_id"].isin(set(check))].sort_values(["image_id", "kind", "box_index"]).reset_index(drop=True)
    bb = b.boxes[b.boxes["image_id"].isin(set(check))].sort_values(["image_id", "kind", "box_index"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(ba, bb, check_exact=True)
        res["boxes_equal"] = True
    except AssertionError as err:
        res["boxes_equal"] = False
        res["boxes_diff"] = str(err)[:2000]
    res["n_boxes_checked"] = len(bb)
    text = json.dumps(res, indent=1, default=str)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    ok = (
        res["b_ids_missing_in_a"] == 0
        and not meta_bad
        and not ch0_bad
        and not rep_bad
        and res["boxes_equal"]
        and res["spectrogram_fields_differing"] == ["extra_windows_s"]
        and res["tiling_fields_differing"] == ["channel_fill"]
        and res["preprocessing_equal"]
    )
    print("IDENTICAL EXCEPT CHANNELS 1-2" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
