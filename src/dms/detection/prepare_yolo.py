"""Prepare the detection dataset in the layout Ultralytics expects.

    data/yolo/images/<split>/<img>.jpg     copies of the annotated images (data/ is git-ignored)
    data/yolo/labels/<split>/<img>.txt     copies of the YOLO labels
    data/yolo/data.yaml                    paths and classes, read by YOLO

Ultralytics finds an image's label by replacing "images" with "labels" in its path, so
images and labels must live in parallel folders.

Usage:
    python -m dms.detection.prepare_yolo
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import yaml

CLASSES = ["phone", "cup", "hand", "face"]  # same order as scripts/ls_to_yolo.py
SELECTION = Path("annotations/selection.csv")
LABELS = Path("annotations/labels")
IMG_ROOT = Path("data/raw/imgs/train")
OUT = Path("data/yolo")


def main() -> None:
    sel = pd.read_csv(SELECTION).set_index("img")  # img -> classname, subject, split
    if OUT.exists():
        shutil.rmtree(OUT)  # rebuilt from scratch: no stale files from earlier batches

    counts = {}
    for split_dir in sorted(LABELS.iterdir()):          # train, val, test
        split = split_dir.name
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        n = 0
        for label in sorted(split_dir.glob("*.txt")):
            img = f"{label.stem}.jpg"
            src = IMG_ROOT / sel.loc[img, "classname"] / img
            dst_img = OUT / "images" / split / img
            dst_label = OUT / "labels" / split / label.name
            shutil.copy2(src, dst_img)
            shutil.copy2(label, dst_label)
            n += 1
        counts[split] = n

    data = {
        "path": str(OUT.resolve()),  # absolute: YOLO resolves relative paths in its own way
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"images per split: {counts}")
    print(f"wrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
