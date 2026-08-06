#!/usr/bin/env python3
"""
Write the official CUB-200-2011 split to committed manifest files.

    python3 scripts/make_split.py
    python3 scripts/make_split.py --val-frac 0.1 --seed 42

Produces (all relative paths into CUB_200_2011/images/):

    data/splits/official_train.txt   ~5395  official train minus val
    data/splits/official_val.txt     ~599   stratified carve-out of official train
    data/splits/official_test.txt     5794  official test, untouched
    data/splits/split_meta.json      seed, counts, sha256 of each manifest

Why this file exists
--------------------
The checkpoint currently in this repo cannot be honestly evaluated. Measured
top-1 with it: official train 94.6%, official test 94.2%, and official test
images that were never in birds_split/train 95.2% -- images it supposedly
never saw scoring *higher* than ones it did, against a realistic ~86-88%
ceiling for this architecture on CUB. The split it was trained on was not
recorded and the directories on disk were reshuffled afterwards, so there is
no way to reconstruct which images were held out.

Committing the manifest makes that failure impossible to repeat: the exact
image list is in version control, hashed, and every later accuracy claim can
name the manifest it was measured against.

The val set is carved out of official *train* only. Official test is never
touched by training or model selection -- that is the whole point of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUB_DIR = os.path.join(PROJECT_DIR, "CUB_200_2011")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "splits")


def read_cub():
    """Return (images, labels, is_train) keyed by CUB image id."""
    need = ["images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"]
    for f in need:
        if not os.path.exists(os.path.join(CUB_DIR, f)):
            sys.exit(f"Missing {os.path.join(CUB_DIR, f)}")

    images, labels, is_train = {}, {}, {}
    with open(os.path.join(CUB_DIR, "images.txt")) as f:
        for line in f:
            i, rel = line.strip().split(" ", 1)
            images[i] = rel
    with open(os.path.join(CUB_DIR, "image_class_labels.txt")) as f:
        for line in f:
            i, c = line.split()
            labels[i] = int(c)
    with open(os.path.join(CUB_DIR, "train_test_split.txt")) as f:
        for line in f:
            i, t = line.split()
            is_train[i] = t == "1"
    classes = [l.split(" ", 1)[1].strip() for l in open(os.path.join(CUB_DIR, "classes.txt"))]
    return images, labels, is_train, classes


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path, rows):
    """One 'relative/path.jpg<TAB>class_index_0based' per line, sorted."""
    with open(path, "w") as f:
        for rel, cls in sorted(rows):
            f.write(f"{rel}\t{cls}\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images, labels, is_train, classes = read_cub()
    n_official_train = sum(is_train.values())
    n_official_test = len(is_train) - n_official_train
    print(f"CUB-200-2011: {len(images)} images, {len(classes)} classes")
    print(f"official split: train {n_official_train}, test {n_official_test}")
    if (n_official_train, n_official_test) != (5994, 5794):
        print("WARNING: official split counts differ from the published 5994/5794")

    # Stratified val carve-out, from official train only.
    by_class = defaultdict(list)
    for i, rel in images.items():
        if is_train[i]:
            by_class[labels[i]].append(rel)

    rng = random.Random(args.seed)
    train_rows, val_rows = [], []
    for cls in sorted(by_class):
        rels = sorted(by_class[cls])          # sort first so the shuffle is reproducible
        rng.shuffle(rels)
        n_val = max(1, round(len(rels) * args.val_frac))
        idx = cls - 1                          # CUB labels are 1-based; torch wants 0-based
        val_rows += [(r, idx) for r in rels[:n_val]]
        train_rows += [(r, idx) for r in rels[n_val:]]

    test_rows = [(rel, labels[i] - 1) for i, rel in images.items() if not is_train[i]]

    # Nothing may appear in two splits, and val must not leak into train.
    s_train = {r for r, _ in train_rows}
    s_val = {r for r, _ in val_rows}
    s_test = {r for r, _ in test_rows}
    assert not (s_train & s_val), "train/val overlap"
    assert not (s_train & s_test), "train/test overlap"
    assert not (s_val & s_test), "val/test overlap"
    assert len(s_train) + len(s_val) + len(s_test) == len(images), "images lost or duplicated"

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {}
    for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        p = os.path.join(OUT_DIR, f"official_{name}.txt")
        write_manifest(p, rows)
        paths[name] = p
        n_cls = len({c for _, c in rows})
        print(f"  official_{name}.txt  {len(rows):>5} images, {n_cls} classes")

    meta = {
        "dataset": "CUB-200-2011",
        "source": "CUB_200_2011/train_test_split.txt (official)",
        "val_carved_from": "official train only; official test never touched",
        "val_frac": args.val_frac,
        "seed": args.seed,
        "counts": {k: sum(1 for _ in open(v)) for k, v in paths.items()},
        "sha256": {k: sha256(v) for k, v in paths.items()},
        "line_format": "relative_path_under_CUB_200_2011/images<TAB>class_index_0based",
    }
    meta_path = os.path.join(OUT_DIR, "split_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {meta_path}")
    print("Commit data/splits/ -- every accuracy claim should cite these hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
