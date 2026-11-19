"""DataLoader per train/val/test, con split per conducente o casuale (esperimento sul leakage)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dms.data.dataset import DriverDataset
from dms.data.splits import load_driver_list, load_splits, random_image_split
from dms.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_transforms


def split_dataframes(
    df: pd.DataFrame, split: str, splits_path: str | Path, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Restituisce {"train": df, "val": df, "test": df}.

    split="driver": conducenti interi da splits.json (il protocollo del progetto).
    split="random": split casuale per immagine, SOLO per l'esperimento sul leakage.
    """
    if split == "driver":
        splits = load_splits(splits_path)
        return {name: df[df["subject"].isin(subs)] for name, subs in splits.items()}

    if split == "random":
        return random_image_split(df, frac_val=0.15, frac_test=0.15, seed=seed)

    raise ValueError(f"split sconosciuto: {split}")


def build_dataloaders(
    csv_path: str | Path = "data/raw/driver_imgs_list.csv",
    img_root: str | Path = "data/raw/imgs/train",
    split: str = "driver",
    splits_path: str | Path = "splits.json",
    batch_size: int = 64,
    num_workers: int = 2,
    size: int = 224,
    mode: str = "letterbox",
    seed: int = 0,
    aug: str = "standard",
    # TODO (passo 2): aggiungi mean e std (default: IMAGENET_MEAN, IMAGENET_STD da dms.data.transforms)
    #   e passali a get_transforms qui sotto, per nome (mean=..., std=...)
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    ) -> dict[str, DataLoader]:
    if aug not in ["standard", "none", "flip"]:
        raise ValueError(f"augmentation sconosciuta: {aug}")
    df = load_driver_list(csv_path)
    parts = split_dataframes(df, split, splits_path, seed)

    loaders = {}
    for name, part in parts.items():
        is_train = name == "train"
        augment = aug != "none" and is_train
        hflip = aug == "flip" and is_train

        dataset = DriverDataset(part, img_root, get_transforms(augment, size, mode, mean=mean, std=std), hflip=hflip)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,  # shuffle solo in train per mescolare le immagini
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # utile se si usa la GPU per velocizzare il trasferimento
            persistent_workers=num_workers > 0,  # utile per non ricaricare i worker ad ogni epoca
            drop_last=is_train,  # drop l'ultimo batch se è più piccolo di batch_size in train
        )
    return loaders
