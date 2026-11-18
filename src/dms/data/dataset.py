"""Dataset PyTorch per State Farm. Atteso: data/raw/imgs/train/<classe>/<img>.jpg"""
from __future__ import annotations

from pathlib import Path
import torch
import pandas as pd

CLASSES = [f"c{i}" for i in range(10)]
CLASS_NAMES = {
    "c0": "guida sicura",
    "c1": "messaggi, mano destra",
    "c2": "telefono, mano destra",
    "c3": "messaggi, mano sinistra",
    "c4": "telefono, mano sinistra",
    "c5": "radio",
    "c6": "bere",
    "c7": "raggiungere dietro",
    "c8": "capelli e trucco",
    "c9": "parlare col passeggero",
}
FLIP_LABEL = {1: 3, 3: 1, 2: 4, 4: 2}


class DriverDataset:
    """Restituisce (immagine, indice_classe). `transform` riceve un'immagine PIL."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform=None, hflip: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.hflip = hflip

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image  # import locale: i test degli split non richiedono Pillow
        row = self.df.iloc[idx]
        path = self.root / row["classname"] / row["img"]
        image = Image.open(path).convert("RGB")
        label = CLASSES.index(row["classname"])
        if self.hflip and torch.rand(1).item() < 0.5:  # flip orizzontale con probabilità 0.5
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = FLIP_LABEL.get(label, label)

        if self.transform is not None:
            image = self.transform(image)
        return image, label