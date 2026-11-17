"""Classificatori con backbone pre-addestrato su ImageNet (transfer learning)."""
from __future__ import annotations

from torch import nn
from torchvision import models

NUM_CLASSES = 10


def build_model(
    name: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True
) -> nn.Module:
    """ResNet di torchvision con la testa sostituita da una a `num_classes` uscite."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"modello sconosciuto: {name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Congela (trainable=False) o scongela tutto tranne la testa `fc`."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(parametri totali, parametri addestrabili)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
