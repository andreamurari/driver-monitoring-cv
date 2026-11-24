"""Prepara il dataset di detection nella struttura che Ultralytics si aspetta.

    data/yolo/images/<split>/<img>.jpg     copie delle immagini annotate (data/ è gitignored)
    data/yolo/labels/<split>/<img>.txt     copie delle etichette YOLO
    data/yolo/data.yaml                    percorsi e classi, letto da YOLO

Ultralytics trova l'etichetta di un'immagine sostituendo "images" con "labels" nel percorso:
per questo immagini ed etichette devono stare in cartelle parallele.

Uso:
    python -m dms.detection.prepare_yolo
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import yaml

CLASSES = ["phone", "cup", "hand", "face"]  # stesso ordine di scripts/ls_to_yolo.py
SELECTION = Path("annotations/selection.csv")
LABELS = Path("annotations/labels")
IMG_ROOT = Path("data/raw/imgs/train")
OUT = Path("data/yolo")


def main() -> None:
    sel = pd.read_csv(SELECTION).set_index("img")  # img -> classname, subject, split
    if OUT.exists():
        shutil.rmtree(OUT)  # si rigenera da zero: niente file vecchi di lotti precedenti

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
        "path": str(OUT.resolve()),  # assoluto: YOLO risolve i percorsi relativi a modo suo
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"immagini per split: {counts}")
    print(f"scritto {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
