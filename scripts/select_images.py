"""Choose the images to annotate (boxes for phone, cup, hand, face).

Rules:
  1. split by driver: each image stays in its driver's split (splits.json), so the detector
     is evaluated on people never seen, like the classifier;
  2. classes balanced by object: c6 (the only source of cups) is oversampled;
  3. spaced frames: within each group, rows are taken at regular intervals, because nearby
     rows in the CSV are near-identical frames.

Usage:
    python scripts/select_images.py                        # first batch (150)
    python scripts/select_images.py --batch 2 --scale 1.5  # next batch, without repeats

Writes annotations/selection.csv: file names only, no images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dms.data.splits import load_driver_list, load_splits

# Images per State Farm class in the first batch (150 in total)
PER_CLASS = {
    "c6": 30,                                     # drinking: the only source of cups
    "c1": 15, "c2": 15, "c3": 15, "c4": 15,       # phone: texting and calls, both hands
    "c0": 12, "c5": 12, "c7": 12, "c8": 12, "c9": 12,  # hands and face, no objects
}
SPLIT_SHARE = {"train": 4 / 6, "val": 1 / 6, "test": 1 / 6}  # about 100 / 25 / 25
OUT = Path("annotations/selection.csv")


def pick_spaced(group: pd.DataFrame, n: int) -> pd.DataFrame:
    """n rows of the group at regular intervals (not random), in CSV order."""
    if len(group) <= n:
        return group
    positions = np.linspace(0, len(group) - 1, n).round().astype(int)  # evenly spaced, first to last
    return group.iloc[positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/driver_imgs_list.csv")
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0, help="multiplies PER_CLASS")
    args = parser.parse_args()

    df = load_driver_list(args.csv)  # columns: subject, classname, img (CSV order)
    splits = load_splits(args.splits)

    # Later batches: skip images already chosen
    previous = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["img"])
    df = df[~df["img"].isin(previous["img"])]

    chosen = []
    for split, subjects in splits.items():
        part = df[df["subject"].isin(subjects)]
        for classname, n_class in PER_CLASS.items():
            n = round(n_class * args.scale * SPLIT_SHARE[split])
            # Sorted by driver, the evenly spaced positions fall on all the split's drivers
            # in proportion to their images
            group = part[part["classname"] == classname].sort_values("subject", kind="stable")
            chosen.append(pick_spaced(group, n).assign(split=split))

    new = pd.concat(chosen).assign(batch=args.batch)
    out = pd.concat([previous, new], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"batch {args.batch}: {len(new)} images, total {len(out)}")
    print(new.pivot_table(index="classname", columns="split", values="img", aggfunc="count",
                          fill_value=0))


if __name__ == "__main__":
    main()
