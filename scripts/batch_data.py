"""Dataset that trains on the annotation batches up to N and tests on the full test split.

Used to retrain an older model (e.g. batch 1 only, with a new seed) on today's clean test.
No images are copied: Ultralytics accepts .txt lists of image paths, and finds each label
by replacing "images" with "labels" in the path.

    data/yolo_upto_b<N>/train.txt, val.txt, data.yaml     (test: data/yolo/images/test)

Usage:
    python scripts/batch_data.py --upto 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

SRC = Path("data/yolo")  # built by python -m dms.detection.prepare_yolo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upto", type=int, required=True, help="last batch to train on")
    parser.add_argument("--selection", default="annotations/selection.csv")
    args = parser.parse_args()

    sel = pd.read_csv(args.selection)
    sel = sel[sel["batch"] <= args.upto]
    out = Path(f"data/yolo_upto_b{args.upto}")
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "val"):
        paths = [(SRC / "images" / split / img).resolve() for img in sel[sel["split"] == split]["img"]]
        paths = [str(p) for p in paths if p.exists()]  # selected but not annotated: skip
        (out / f"{split}.txt").write_text("\n".join(paths) + "\n")
        counts[split] = len(paths)

    data = {
        "path": str(out.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": str((SRC / "images" / "test").resolve()),  # always the full clean test
        "names": yaml.safe_load((SRC / "data.yaml").read_text())["names"],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"batches <= {args.upto}: {counts} -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
