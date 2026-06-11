"""CNN model factory for transfer learning (ResNet50, DenseNet121, EfficientNet-B0).

All models take 3x224x224 inputs, output `num_classes` logits. Backbone is loaded
with ImageNet pretrained weights; the classification head is replaced.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES, SUPPORTED_MODELS


def _build_resnet50(num_classes: int) -> tuple[nn.Module, str]:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m, "layer4"  # target layer name for Grad-CAM


def _build_densenet121(num_classes: int) -> tuple[nn.Module, str]:
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m, "features.denseblock4"


def _build_efficientnet_b0(num_classes: int) -> tuple[nn.Module, str]:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m, "features.8"  # last conv block


_BUILDERS = {
    "resnet50": _build_resnet50,
    "densenet121": _build_densenet121,
    "efficientnet_b0": _build_efficientnet_b0,
}


def build_model(name: str, num_classes: int = NUM_CLASSES) -> tuple[nn.Module, str]:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}")
    return _BUILDERS[name](num_classes)


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    head_param_ids: set[int] = set()
    if name == "resnet50":
        head_params = model.fc.parameters()
    elif name == "densenet121":
        head_params = model.classifier.parameters()
    elif name == "efficientnet_b0":
        head_params = model.classifier[-1].parameters()
    else:
        raise ValueError(name)
    for p in head_params:
        head_param_ids.add(id(p))
    for p in model.parameters():
        if id(p) in head_param_ids:
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def param_groups(model: nn.Module, name: str, lr_head: float, lr_backbone: float):
    """Separate optimizer param groups for backbone vs head."""
    if name == "resnet50":
        head = list(model.fc.parameters())
    elif name == "densenet121":
        head = list(model.classifier.parameters())
    elif name == "efficientnet_b0":
        head = list(model.classifier[-1].parameters())
    else:
        raise ValueError(name)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": head, "lr": lr_head},
    ]


def get_gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Return the conv layer to hook for Grad-CAM."""
    if name == "resnet50":
        return model.layer4[-1]
    if name == "densenet121":
        return model.features.denseblock4
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(name)
