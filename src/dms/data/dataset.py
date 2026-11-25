"""PyTorch dataset for State Farm. Expected layout: data/raw/imgs/train/<class>/<img>.jpg"""
from __future__ import annotations

from pathlib import Path
import torch
import pandas as pd

CLASSES = [f"c{i}" for i in range(10)]
CLASS_NAMES = {
    "c0": "safe driving",
    "c1": "texting, right hand",
    "c2": "phone call, right hand",
    "c3": "texting, left hand",
    "c4": "phone call, left hand",
    "c5": "radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to the passenger",
}
FLIP_LABEL = {1: 3, 3: 1, 2: 4, 4: 2}


class DriverDataset:
    """Return (image, class index). `transform` receives a PIL image.

    hflip=True mirrors half of the images and swaps the left/right labels (c1<->c3, c2<->c4).
    """

    def __init__(self, df: pd.DataFrame, root: str | Path, transform=None, hflip: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.hflip = hflip

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image  # local import: the split tests do not need Pillow
        row = self.df.iloc[idx]
        path = self.root / row["classname"] / row["img"]
        image = Image.open(path).convert("RGB")
        label = CLASSES.index(row["classname"])
        if self.hflip and torch.rand(1).item() < 0.5:  # horizontal flip with probability 0.5
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = FLIP_LABEL.get(label, label)

        if self.transform is not None:
            image = self.transform(image)
        return image, label