"""Transforms for State Farm (640x480 RGB images)."""
from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_ops(size: int, mode: str) -> list:
    """Operations that bring the image to size x size.

    mode="letterbox": keeps the 4:3 ratio and adds black bands above and below.
    mode="stretch":   distorts to size x size.
    """
    if mode == "stretch":
        return [v2.Resize((size, size))]

    if mode == "letterbox":
        h = size * 3 // 4  # the long side (width) becomes size, the height 3/4 of it
        total = size - h  # total vertical padding
        top = total // 2
        bottom = total - top  # with an odd size, one extra pixel goes at the bottom
        return [
            v2.Resize((h, size)),  # (height, width)
            v2.Pad(padding=(0, top, 0, bottom)),  # (left, top, right, bottom)
        ]

    raise ValueError(f"unknown mode: {mode}")


def get_transforms(
    train: bool,
    size: int = 224,
    mode: str = "letterbox",
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> v2.Compose:
    """Resize, (train only) augmentation, tensor conversion and normalisation."""
    ops = _resize_ops(size, mode)

    if train:
        # Augmentation on real pixel values, before normalisation. No horizontal flip here:
        # it would swap right and left (c1<->c3, c2<->c4); see DriverDataset(hflip=True).
        ops.append(v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        ops.append(v2.RandomAffine(degrees=5, translate=(0.1, 0.1)))

    ops.append(v2.ToImage())
    ops.append(v2.ToDtype(torch.float32, scale=True))  # 0-255 -> 0-1
    ops.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(ops)
