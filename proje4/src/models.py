"""Model factory. Wraps timm.create_model for the 3 architectures we compare."""
import torch.nn as nn

from .config import MODEL_CONFIGS, NUM_CLASSES


def create_model(name: str, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(MODEL_CONFIGS)}")

    import timm  # local import so the project still imports without timm installed yet

    cfg = MODEL_CONFIGS[name]
    model = timm.create_model(
        cfg["timm_name"],
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
