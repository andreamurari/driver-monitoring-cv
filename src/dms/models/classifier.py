"""Classificatori con backbone pre-addestrato (transfer learning): ResNet di torchvision, ViT di timm."""
from __future__ import annotations

import timm
from torch import nn
from torchvision import models

NUM_CLASSES = 10
VIT_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"  # ViT-S/16, preaddestrato su ImageNet-21k, poi 1k


def build_model(
    name: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True
) -> nn.Module:
    """Backbone pre-addestrato con la testa sostituita da una a `num_classes` uscite."""
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
    """Il modulo della testa, per qualsiasi modello che costruiamo."""
    head = model.get_classifier() if hasattr(model, "get_classifier") else model.fc
    return head


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Congela (trainable=False) o scongela il backbone; la testa resta sempre addestrabile."""
    for param in model.parameters():
        param.requires_grad = trainable
    for param in get_head(model).parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(parametri totali, parametri addestrabili)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
