"""Automatic pre-labels for Label Studio: correcting them is faster than drawing from scratch.

Sources:
    phone, cup  YOLO trained on COCO (class "cell phone"; "cup" and "bottle" -> cup)
    hand        MediaPipe HandLandmarker: box around the 21 hand landmarks
    face        MediaPipe FaceDetector

Usage:
    python scripts/prelabel.py                       # all images in annotations/selection.csv
    python scripts/prelabel.py --limit 5             # quick test

Writes scratch/prelabels.json (git-ignored: it holds image paths; it is imported into
Label Studio). The images stay where they are: Label Studio reads them from data/raw.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from ultralytics import YOLO

DATA_ROOT = Path("data/raw")  # the same folder enabled in Label Studio (DOCUMENT_ROOT)
IMG_DIR = DATA_ROOT / "imgs/train"
YOLO_CLASSES = {67: "phone", 41: "cup", 39: "cup"}  # COCO: cell phone, cup, bottle
MODEL_VERSION = "prelabel-v1"


def to_ls_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int,
              label: str, score: float) -> dict:
    """Box in pixels (x0, y0) - (x1, y1) -> Label Studio result.

    Label Studio wants the top-left corner and the size as a PERCENTAGE of the image
    width and height (0-100), not in pixels.
    """
    # inside the image (enlarged hand boxes can cross the border)
    x0 = max(0, x0)
    x1 = min(w, x1)
    y0 = max(0, y0)
    y1 = min(h, y1)
    value = {
        "x": x0 / w * 100,
        "y": y0 / h * 100,
        "width": (x1 - x0) / w * 100,
        "height": (y1 - y0) / h * 100,
        "rotation": 0,
        "rectanglelabels": [label],
    }
    return {"from_name": "label", "to_name": "image", "type": "rectanglelabels",
            "original_width": w, "original_height": h, "value": value, "score": round(score, 3)}


def hand_box(landmarks, w: int, h: int, margin: float = 0.15) -> tuple[float, float, float, float]:
    """Box in pixels around the 21 landmarks of a hand (normalised 0-1 coordinates).

    The landmarks sit on joints and fingertips: a tight box cuts the edge of the hand,
    so it is enlarged by `margin` (15% of the side) on each side.
    """
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) * margin
    dy = (y1 - y0) * margin
    return (x0 - dx) * w, (y0 - dy) * h, (x1 + dx) * w, (y1 + dy) * h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="annotations/selection.csv")
    parser.add_argument("--out", default="scratch/prelabels.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.15, help="low threshold: better an extra box"
                        " to delete than a missing one to draw")
    parser.add_argument("--batch", type=int, default=None, help="pre-label only this batch")
    parser.add_argument("--model", default=None,
                        help="our trained detector (best.pt): it predicts all 4 classes itself,"
                             " instead of YOLO-COCO + MediaPipe")
    args = parser.parse_args()

    if args.model:
        own = YOLO(args.model)
    yolo = YOLO("models/yolo11s.pt")
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2, min_hand_detection_confidence=0.3))
    faces = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path="models/blaze_face_short_range.tflite"),
        min_detection_confidence=0.3))

    sel = pd.read_csv(args.selection)
    # TODO 2: with --batch, keep only the rows of that batch (column "batch" of selection.csv)
    if args.batch:
        sel = sel[sel["batch"] == args.batch]
    if args.limit:
        sel = sel.head(args.limit)

    tasks, counts = [], {"phone": 0, "cup": 0, "hand": 0, "face": 0}
    for row in sel.itertuples():
        path = IMG_DIR / row.classname / row.img
        image = mp.Image.create_from_file(str(path))
        w, h = image.width, image.height
        result = []

        if args.model:
            # Our detector already knows the 4 classes, with our names (own.names: id -> name)
            det = own(str(path), conf=args.conf, verbose=False)[0]
            # TODO 3: one Label Studio box per detection, like the YOLO-COCO loop below,
            #   but the label is own.names[int(cls)] (no YOLO_CLASSES mapping needed)
            for (x0, y0, x1, y1), cls, conf in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist(),
                                                  det.boxes.conf.tolist()):
                result.append(to_ls_box(x0, y0, x1, y1, w, h, own.names[int(cls)], conf))
            # TODO 4: skip MediaPipe and the COCO model for this image: jump to the bookkeeping
            #   below (hint: move the counting/tasks.append part into a small helper, or use
            #   an if/else around the three blocks that follow)
            if args.model:
                for r in result:
                    counts[r["value"]["rectanglelabels"][0]] += 1
                tasks.append({
                    # Path relative to DATA_ROOT, served by Label Studio as a local file
                    "data": {"image": f"/data/local-files/?d=imgs/train/{row.classname}/{row.img}",
                             "img": row.img, "split": row.split, "classname": row.classname,
                             "info": f"{row.img} | {row.subject} | {row.classname} | {row.split}"},
                    "predictions": [{"model_version": MODEL_VERSION, "result": result}],
                })
                continue  # skip the rest of the loop

        # Phone and cup: YOLO, only the 3 COCO classes we need
        det = yolo(str(path), conf=args.conf, classes=list(YOLO_CLASSES), verbose=False)[0]
        for (x0, y0, x1, y1), cls, conf in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist(),
                                              det.boxes.conf.tolist()):
            result.append(to_ls_box(x0, y0, x1, y1, w, h, YOLO_CLASSES[int(cls)], conf))

        # Hands: one box per detected hand
        for landmarks in hands.detect(image).hand_landmarks:
            # HandLandmarker gives no per-box score: 1.0 as a placeholder
            result.append(to_ls_box(*hand_box(landmarks, w, h), w, h, "hand", 1.0))

        # Faces: the detector already returns a box in pixels
        for d in faces.detect(image).detections:
            b = d.bounding_box
            result.append(to_ls_box(b.origin_x, b.origin_y, b.origin_x + b.width,
                                    b.origin_y + b.height, w, h, "face", d.categories[0].score))

        for r in result:
            counts[r["value"]["rectanglelabels"][0]] += 1
        tasks.append({
            # Path relative to DATA_ROOT, served by Label Studio as a local file
            "data": {"image": f"/data/local-files/?d=imgs/train/{row.classname}/{row.img}",
                     "img": row.img, "split": row.split, "classname": row.classname,
                     "info": f"{row.img} | {row.subject} | {row.classname} | {row.split}"},
            "predictions": [{"model_version": MODEL_VERSION, "result": result}],
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks)} images, proposed boxes: {counts}")
    print(f"saved to {args.out}: import it into Label Studio")


if __name__ == "__main__":
    main()
