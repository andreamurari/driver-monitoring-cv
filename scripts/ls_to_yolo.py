"""Label Studio JSON export -> YOLO labels, one folder per split.

Usage:
    python scripts/ls_to_yolo.py scratch/annotations_batch1.json

Writes annotations/labels/<split>/<img without extension>.txt, one line per box:
    <class id> <x_center> <y_center> <width> <height>     (all as fractions 0-1)
Text only: no image is copied.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CLASSES = ["phone", "cup", "hand", "face"]  # the order defines the ids: same in data.yaml
OUT = Path("annotations/labels")


def ls_to_yolo_line(value: dict) -> str:
    """One Label Studio box -> one YOLO line.

    Label Studio: top-left corner (x, y) and size, in PERCENT (0-100).
    YOLO: box CENTER and size, as FRACTIONS (0-1).
    """
    class_id = CLASSES.index(value["rectanglelabels"][0])
    w, h = value["width"] / 100, value["height"] / 100  # percent -> fraction
    xc, yc = value["x"] / 100 + w / 2, value["y"] / 100 + h / 2  # corner -> center
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", help="JSON file exported from Label Studio")
    args = parser.parse_args()

    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    counts, per_split = Counter(), Counter()
    for task in tasks:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            continue  # image not annotated yet
        boxes = [r for r in done[-1]["result"] if r.get("type") == "rectanglelabels"]
        lines = [ls_to_yolo_line(r["value"]) for r in boxes]

        split, img = task["data"]["split"], task["data"]["img"]
        path = OUT / split / f"{Path(img).stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""))  # empty file = no objects

        per_split[split] += 1
        counts.update(r["value"]["rectanglelabels"][0] for r in boxes)

    print(f"images per split: {dict(per_split)}")
    print(f"boxes per class: {dict(counts)}")


if __name__ == "__main__":
    main()
