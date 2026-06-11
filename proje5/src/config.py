"""Central config: paths, hyperparameters, class labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "FracAtlas"
IMAGES_ROOT = DATA_ROOT / "images"
FRACTURED_DIR = IMAGES_ROOT / "Fractured"
NON_FRACTURED_DIR = IMAGES_ROOT / "Non_fractured"

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_ROOT / "splits"
CHECKPOINTS_DIR = OUTPUTS_ROOT / "checkpoints"
LOGS_DIR = OUTPUTS_ROOT / "logs"
FIGURES_DIR = OUTPUTS_ROOT / "figures"
RESULTS_DIR = OUTPUTS_ROOT / "results"

SEED = 42
NUM_CLASSES = 2
CLASS_NAMES = ["non_fractured", "fractured"]  # 0, 1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainConfig:
    model_name: str = "resnet50"
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    warmup_frozen_epochs: int = 2  # train head only for N epochs, then unfreeze
    early_stopping_patience: int = 5
    use_weighted_sampler: bool = True
    pin_memory: bool = True
    mixed_precision: bool = True
    val_split: float = 0.15
    test_split: float = 0.15


SEEDS: list[int] = [42, 123, 2024]


SUPPORTED_MODELS: list[str] = ["resnet50", "densenet121", "efficientnet_b0"]
