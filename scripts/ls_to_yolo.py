"""Export JSON di Label Studio -> etichette YOLO, una cartella per split.

Uso:
    python scripts/ls_to_yolo.py scratch/annotations_batch1.json

Scrive annotations/labels/<split>/<img senza estensione>.txt, una riga per riquadro:
    <id classe> <x_centro> <y_centro> <larghezza> <altezza>     (tutto in frazione 0-1)
Solo testo: nessuna immagine viene copiata.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CLASSES = ["phone", "cup", "hand", "face"]  # l'ordine decide gli id: lo stesso in data.yaml
OUT = Path("annotations/labels")


def ls_to_yolo_line(value: dict) -> str:
    """Un riquadro di Label Studio -> una riga YOLO.

    Label Studio: angolo in alto a sinistra (x, y) e dimensioni, in PERCENTUALE (0-100).
    YOLO: CENTRO del riquadro e dimensioni, in FRAZIONE (0-1).
    """
    class_id = CLASSES.index(value["rectanglelabels"][0])
    w, h = value["width"] / 100, value["height"] / 100  # percentuale -> frazione
    xc, yc = value["x"] / 100 + w / 2, value["y"] / 100 + h / 2  # angolo -> centro
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", help="file JSON esportato da Label Studio")
    args = parser.parse_args()

    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    counts, per_split = Counter(), Counter()
    for task in tasks:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            continue  # immagine non ancora annotata
        boxes = [r for r in done[-1]["result"] if r.get("type") == "rectanglelabels"]
        lines = [ls_to_yolo_line(r["value"]) for r in boxes]

        split, img = task["data"]["split"], task["data"]["img"]
        path = OUT / split / f"{Path(img).stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""))  # file vuoto = nessun oggetto

        per_split[split] += 1
        counts.update(r["value"]["rectanglelabels"][0] for r in boxes)

    print(f"immagini per split: {dict(per_split)}")
    print(f"riquadri per classe: {dict(counts)}")


if __name__ == "__main__":
    main()
