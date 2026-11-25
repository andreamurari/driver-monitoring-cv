"""DataLoaders for train/val/test, with a split by driver or a random split (leakage experiment)."""
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
    """Return {"train": df, "val": df, "test": df}.

    split="driver": whole drivers from splits.json (the project protocol).
    split="random": random split by image, ONLY for the leakage experiment.
    """
    if split == "driver":
        splits = load_splits(splits_path)
        return {name: df[df["subject"].isin(subs)] for name, subs in splits.items()}

    if split == "random":
        return random_image_split(df, frac_val=0.15, frac_test=0.15, seed=seed)

    raise ValueError(f"unknown split: {split}")


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
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    ) -> dict[str, DataLoader]:
    if aug not in ["standard", "none", "flip"]:
        raise ValueError(f"unknown augmentation: {aug}")
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
            shuffle=is_train,  # shuffle only in training
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # faster host-to-GPU copies
            persistent_workers=num_workers > 0,  # keep workers alive between epochs
            drop_last=is_train,  # drop the last, smaller batch in training
        )
    return loaders
