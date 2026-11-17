"""Trasformazioni per State Farm (immagini 640x480, RGB)."""
from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_ops(size: int, mode: str) -> list:
    """Operazioni che portano l'immagine a size x size.

    mode="letterbox": mantiene il 4:3 e aggiunge bande nere sopra e sotto.
    mode="stretch":   deforma a size x size.
    """
    if mode == "stretch":
        return [v2.Resize((size, size))]

    if mode == "letterbox":
        h = size * 3 // 4  # il lato lungo (larghezza) diventa size, l'altezza i suoi 3/4
        total = size - h  # padding verticale totale
        top = total // 2
        bottom = total - top  # con size dispari sotto c'è un pixel in più
        return [
            v2.Resize((h, size)),  # (altezza, larghezza)
            v2.Pad(padding=(0, top, 0, bottom)),  # (sinistra, sopra, destra, sotto)
        ]

    raise ValueError(f"mode sconosciuto: {mode}")


def get_transforms(train: bool, size: int = 224, mode: str = "letterbox") -> v2.Compose:
    ops = _resize_ops(size, mode)

    if train:
        # Augmentation su pixel veri, prima della normalizzazione. Niente flip orizzontale:
        # scambierebbe destra e sinistra (c1<->c3, c2<->c4).
        ops.append(v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        ops.append(v2.RandomAffine(degrees=5, translate=(0.1, 0.1)))

    ops.append(v2.ToImage())
    ops.append(v2.ToDtype(torch.float32, scale=True))  # 0-255 -> 0-1
    ops.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose(ops)
