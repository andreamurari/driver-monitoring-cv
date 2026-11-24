"""Fine-tuning di YOLO sulle annotazioni e valutazione sul test (conducenti mai visti).

Uso:
    python -m dms.detection.train_yolo --config configs/yolo11n.yaml

Scrive:
    runs/yolo/<name>/                   pesi, curve, grafici di Ultralytics (gitignored)
    reports/metrics/<name>_test.json    mAP e AP per classe sul test (si può pubblicare)
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
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    project = Path("runs/yolo").resolve()

    model = YOLO(cfg["model"])
    model.train(data=cfg["data"], project=str(project), name=cfg["name"], exist_ok=True,
                **cfg["train"])

    # Valutazione del checkpoint migliore (scelto sulla validazione) sul test
    best = YOLO(project / cfg["name"] / "weights" / "best.pt")
    res = best.val(data=cfg["data"], split="test", imgsz=cfg["train"]["imgsz"],
                   device=cfg["train"]["device"], project=str(project),
                   name=f"{cfg['name']}_test", exist_ok=True)

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
    out = Path("reports/metrics") / f"{cfg['name']}_test.json"
    out.write_text(json.dumps(metrics, indent=2))
    t = metrics["test"]
    print(f"test: mAP50 {t['mAP50']:.3f} | mAP50-95 {t['mAP50_95']:.3f}")
    print("AP50 per classe:", {k: round(v, 3) for k, v in t["per_class_AP50"].items()})
    print(f"salvato {out}")


if __name__ == "__main__":  # necessario su Windows con workers > 0
    main()
