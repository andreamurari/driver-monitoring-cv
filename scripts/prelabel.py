"""Pre-etichette automatiche per Label Studio: si correggono invece di disegnare da zero.

Fonti:
    phone, cup  YOLO addestrato su COCO (classi "cell phone"; "cup" e "bottle" -> cup)
    hand        MediaPipe HandLandmarker: riquadro attorno ai 21 punti della mano
    face        MediaPipe FaceDetector

Uso:
    python scripts/prelabel.py                       # tutte le immagini di annotations/selection.csv
    python scripts/prelabel.py --limit 5             # prova veloce

Scrive scratch/prelabels.json (gitignored: contiene i percorsi delle immagini, si importa
in Label Studio). Le immagini restano dove sono: Label Studio le legge da data/raw.
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

DATA_ROOT = Path("data/raw")  # la stessa cartella abilitata in Label Studio (DOCUMENT_ROOT)
IMG_DIR = DATA_ROOT / "imgs/train"
YOLO_CLASSES = {67: "phone", 41: "cup", 39: "cup"}  # COCO: cell phone, cup, bottle
MODEL_VERSION = "prelabel-v1"


def to_ls_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int,
              label: str, score: float) -> dict:
    """Riquadro in pixel (x0, y0) - (x1, y1) -> risultato di Label Studio.

    Label Studio vuole l'angolo in alto a sinistra e le dimensioni in PERCENTUALE
    della larghezza e dell'altezza dell'immagine (0-100), non in pixel.
    """
    # dentro l'immagine (i riquadri allargati delle mani possono uscire dal bordo)
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
    """Riquadro in pixel attorno ai 21 punti di una mano (coordinate normalizzate 0-1).

    I punti sono su articolazioni e polpastrelli: il riquadro stretto taglia il bordo della
    mano, per questo si allarga di `margin` (15% del lato) per parte.
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
    parser.add_argument("--conf", type=float, default=0.15, help="soglia bassa: meglio un riquadro"
                        " in più da cancellare che uno in meno da disegnare")
    args = parser.parse_args()

    yolo = YOLO("models/yolo11s.pt")
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2, min_hand_detection_confidence=0.3))
    faces = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path="models/blaze_face_short_range.tflite"),
        min_detection_confidence=0.3))

    sel = pd.read_csv(args.selection)
    if args.limit:
        sel = sel.head(args.limit)

    tasks, counts = [], {"phone": 0, "cup": 0, "hand": 0, "face": 0}
    for row in sel.itertuples():
        path = IMG_DIR / row.classname / row.img
        image = mp.Image.create_from_file(str(path))
        w, h = image.width, image.height
        result = []

        # Telefono e bicchiere: YOLO, solo le 3 classi COCO che ci interessano
        det = yolo(str(path), conf=args.conf, classes=list(YOLO_CLASSES), verbose=False)[0]
        for (x0, y0, x1, y1), cls, conf in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist(),
                                              det.boxes.conf.tolist()):
            result.append(to_ls_box(x0, y0, x1, y1, w, h, YOLO_CLASSES[int(cls)], conf))

        # Mani: un riquadro per ogni mano trovata
        for landmarks in hands.detect(image).hand_landmarks:
            # HandLandmarker non dà un punteggio per riquadro: 1.0 come segnaposto
            result.append(to_ls_box(*hand_box(landmarks, w, h), w, h, "hand", 1.0))

        # Volti: il detector restituisce già un riquadro in pixel
        for d in faces.detect(image).detections:
            b = d.bounding_box
            result.append(to_ls_box(b.origin_x, b.origin_y, b.origin_x + b.width,
                                    b.origin_y + b.height, w, h, "face", d.categories[0].score))

        for r in result:
            counts[r["value"]["rectanglelabels"][0]] += 1
        tasks.append({
            # Percorso relativo a DATA_ROOT, servito da Label Studio come file locale
            "data": {"image": f"/data/local-files/?d=imgs/train/{row.classname}/{row.img}",
                     "img": row.img, "split": row.split, "classname": row.classname,
                     "info": f"{row.img} | {row.subject} | {row.classname} | {row.split}"},
            "predictions": [{"model_version": MODEL_VERSION, "result": result}],
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks)} immagini, riquadri proposti: {counts}")
    print(f"salvato in {args.out}: da importare in Label Studio")


if __name__ == "__main__":
    main()
