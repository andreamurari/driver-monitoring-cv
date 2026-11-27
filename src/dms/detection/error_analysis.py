"""Detector error analysis on the test split: error types, per class and per driver.

Each ground-truth box and each prediction (conf >= --conf) gets one type:
    tp            prediction of the same class with IoU >= 0.5
    localization  same class, but IoU between 0.1 and 0.5 (found, box misplaced)
    wrong_class   IoU >= 0.5 with a true box of another class
    missed        true box with no prediction near it (false negative)
    background    prediction that touches no true box (false positive)

Usage:
    python -m dms.detection.error_analysis --run yolo11n_batch2_s0

Writes reports/metrics/<run>_test_errors.csv (one row per box: file names, no images) and
prints the tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

DATA = Path("data/yolo")
NAMES = ["phone", "cup", "hand", "face"]


def load_labels(path: Path) -> np.ndarray:
    """YOLO label file -> array of (class, x0, y0, x1, y1), normalised 0-1."""
    rows = [list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 5))
    a = np.array(rows)
    c, xc, yc, w, h = a.T
    return np.stack([c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box of a (n, 4) and of b (m, 4), xyxy -> (n, m)."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def classify(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> list[dict]:
    """Error type of every true box and every prediction of one image."""
    rows = []
    ious = iou_matrix(pred[:, 1:], gt[:, 1:]) if len(gt) and len(pred) else np.zeros((len(pred), len(gt)))
    same = pred[:, None, 0] == gt[None, :, 0] if len(gt) and len(pred) else np.zeros_like(ious, bool)
    gt_type = [None] * len(gt)
    pred_type = [None] * len(pred)
    pred_iou = [0.0] * len(pred)

    order = np.argsort(-conf)  # most confident predictions claim their box first
    # 1) true positives: same class, IoU >= 0.5, each true box matched once
    for i in order:
        cand = np.where(same[i] & (ious[i] >= 0.5) & np.array([t is None for t in gt_type]))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            gt_type[j], pred_type[i], pred_iou[i] = "tp", "tp", float(ious[i, j])
    # 2) remaining predictions: wrong class, localization or background
    for i in order:
        if pred_type[i]:
            continue
        free = np.array([t is None for t in gt_type], bool)
        other = np.where(~same[i] & (ious[i] >= 0.5) & free)[0] if len(gt) else []
        loose = np.where(same[i] & (ious[i] >= 0.1) & free)[0] if len(gt) else []
        if len(other):
            j = other[np.argmax(ious[i, other])]
            gt_type[j] = pred_type[i] = "wrong_class"
        elif len(loose):
            j = loose[np.argmax(ious[i, loose])]
            gt_type[j] = pred_type[i] = "localization"
        else:
            pred_type[i] = "background"
            continue
        pred_iou[i] = float(ious[i, j])

    for j, g in enumerate(gt):
        rows.append({"source": "gt", "cls": NAMES[int(g[0])], "type": gt_type[j] or "missed",
                     "conf": np.nan, "iou": np.nan, "area": (g[3] - g[1]) * (g[4] - g[2])})
    for i, p in enumerate(pred):
        rows.append({"source": "pred", "cls": NAMES[int(p[0])], "type": pred_type[i],
                     "conf": float(conf[i]), "iou": pred_iou[i],
                     "area": (p[3] - p[1]) * (p[4] - p[2])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder in runs/yolo with weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(Path("runs/yolo") / args.run / "weights" / "best.pt")
    subject = pd.read_csv("annotations/selection.csv").set_index("img")["subject"]

    rows = []
    for img in sorted((DATA / "images" / "test").glob("*.jpg")):
        gt = load_labels(DATA / "labels" / "test" / f"{img.stem}.txt")
        r = model(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pred = np.concatenate([r.boxes.cls.numpy()[:, None], r.boxes.xyxyn.numpy()], axis=1)
        for row in classify(gt, pred, r.boxes.conf.numpy()):
            rows.append({"img": img.name, "subject": subject[img.name], **row})
    df = pd.DataFrame(rows)

    out = Path("reports/metrics") / f"{args.run}_test_errors.csv"
    df.to_csv(out, index=False)

    gt = df[df["source"] == "gt"]
    print(f"\n{args.run}: true boxes by type (conf >= {args.conf})")
    print(pd.crosstab(gt["cls"], gt["type"], margins=True).reindex(NAMES + ["All"]))
    fp = df[(df["source"] == "pred") & (df["type"] == "background")]
    print("\nbackground false positives per class:", fp["cls"].value_counts().to_dict())
    print("\nrecall (tp / true boxes) per driver and class")
    gt = gt.assign(tp=gt["type"] == "tp")
    rec = gt.pivot_table(index="subject", columns="cls", values="tp", aggfunc="mean")
    n = gt.pivot_table(index="subject", columns="cls", values="tp", aggfunc="size")
    print(rec[NAMES].round(2).astype(str) + " (" + n[NAMES].fillna(0).astype(int).astype(str) + ")")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
