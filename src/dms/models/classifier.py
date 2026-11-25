"""Classifiers with a pretrained backbone (transfer learning): torchvision ResNets, timm ViT."""
from __future__ import annotations

import timm
from torch import nn
from torchvision import models

NUM_CLASSES = 10
VIT_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"  # ViT-S/16, pretrained on ImageNet-21k, then 1k


def build_model(
    name: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True
) -> nn.Module:
    """Pretrained backbone with the head replaced by one with `num_classes` outputs."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "vit_s16":
        model = timm.create_model(VIT_NAME, pretrained=pretrained, num_classes=num_classes)
        return model
    else:
        raise ValueError(f"modello sconosciuto: {name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_head(model: nn.Module) -> nn.Module:
    """The head module, for any model built here."""
    head = model.get_classifier() if hasattr(model, "get_classifier") else model.fc
    return head


def get_normalization(model: nn.Module) -> tuple[tuple, tuple]:
    """(mean, std) used in the model's pretraining."""
    if hasattr(model, "get_classifier"):
        import timm

        cfg = timm.data.resolve_model_data_config(model)
        return cfg["mean"], cfg["std"]
    else:
        from dms.data.transforms import IMAGENET_MEAN, IMAGENET_STD

        return IMAGENET_MEAN, IMAGENET_STD


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (trainable=False) or unfreeze the backbone; the head always stays trainable."""
    for param in model.parameters():
        param.requires_grad = trainable
    for param in get_head(model).parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(total parameters, trainable parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
