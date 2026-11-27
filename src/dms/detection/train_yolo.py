"""Fine-tune YOLO on the annotations and evaluate on the test split (unseen drivers).

Usage:
    python -m dms.detection.train_yolo --config configs/yolo11n.yaml
    python -m dms.detection.train_yolo --config configs/yolo11n.yaml --eval-only   # test set changed
    python -m dms.detection.train_yolo --config configs/yolo11n.yaml --eval-only --tag _test41

Writes:
    runs/yolo/<name>/                   Ultralytics weights, curves, plots (git-ignored)
    reports/metrics/<name>_test.json    test mAP and per-class AP (publishable)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training: re-evaluate the existing best.pt on the test split")
    parser.add_argument("--tag", default="_test", help="suffix of the metrics file name")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    project = Path("runs/yolo").resolve()

    if not args.eval_only:
        model = YOLO(cfg["model"])
        model.train(data=cfg["data"], project=str(project), name=cfg["name"], exist_ok=True,
                    **cfg["train"])

    # Evaluate the best checkpoint (chosen on validation) on the test split
    best = YOLO(project / cfg["name"] / "weights" / "best.pt")
    res = best.val(data=cfg["data"], split="test", imgsz=cfg["train"]["imgsz"],
                   device=cfg["train"]["device"], project=str(project),
                   name=f"{cfg['name']}{args.tag}", exist_ok=True)

    names = res.names
    metrics = {
        "config": cfg,
        "test": {
            "mAP50": float(res.box.map50),
            "mAP50_95": float(res.box.map),
            "precision": float(res.box.mp),
            "recall": float(res.box.mr),
            "per_class_AP50": {names[int(c)]: float(ap)
                               for c, ap in zip(res.box.ap_class_index, res.box.ap50)},
            "per_class_AP50_95": {names[int(c)]: float(ap)
                                  for c, ap in zip(res.box.ap_class_index, res.box.ap)},
        },
    }
    out = Path("reports/metrics") / f"{cfg['name']}{args.tag}.json"
    out.write_text(json.dumps(metrics, indent=2))
    t = metrics["test"]
    print(f"test: mAP50 {t['mAP50']:.3f} | mAP50-95 {t['mAP50_95']:.3f}")
    print("AP50 per class:", {k: round(v, 3) for k, v in t["per_class_AP50"].items()})
    print(f"saved {out}")


if __name__ == "__main__":  # required on Windows with workers > 0
    main()
