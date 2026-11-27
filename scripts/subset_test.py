"""Test subset with only the images of one annotation batch.

Why: batch 2 was pre-labelled by the batch-1 detector, and most of its boxes were accepted
unchanged. On those images the old model is scored partly against its own predictions,
which inflates its mAP50-95. Batch-1 labels come from YOLO-COCO + MediaPipe, independent of
both detectors, so they give a fair comparison.

    data/yolo_test_b<batch>/images/test, labels/test, data.yaml

Usage:
    python scripts/subset_test.py --batch 1
    yolo val model=<best.pt> data=data/yolo_test_b1/data.yaml split=test
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

SRC = Path("data/yolo")  # built by python -m dms.detection.prepare_yolo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--selection", default="annotations/selection.csv")
    args = parser.parse_args()

    sel = pd.read_csv(args.selection)
    imgs = sel[(sel["batch"] == args.batch) & (sel["split"] == args.split)]["img"]

    out = Path(f"data/yolo_{args.split}_b{args.batch}")
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / args.split).mkdir(parents=True)
    (out / "labels" / args.split).mkdir(parents=True)

    n = 0
    for img in imgs:
        src_img = SRC / "images" / args.split / img
        if not src_img.exists():  # selected but not annotated
            continue
        label = f"{Path(img).stem}.txt"
        shutil.copy2(src_img, out / "images" / args.split / img)
        shutil.copy2(SRC / "labels" / args.split / label, out / "labels" / args.split / label)
        n += 1

    split_dir = f"images/{args.split}"
    data = {
        "path": str(out.resolve()),
        "train": split_dir,  # unused, but Ultralytics requires train and val keys
        "val": split_dir,
        args.split: split_dir,
        "names": yaml.safe_load((SRC / "data.yaml").read_text())["names"],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"batch {args.batch}, {args.split}: {n} images -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
