"""Project-wide configuration: paths, class names, hyperparameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "Dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_CSV = RESULTS_DIR / "metrics.csv"

CLASS_NAMES = [
    "Among Us",
    "Apex Legends",
    "Fortnite",
    "Forza Horizon",
    "Free Fire",
    "Genshin Impact",
    "God of War",
    "Minecraft",
    "Roblox",
    "Terraria",
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42

MODEL_CONFIGS = {
    "resnet50": {
        "timm_name": "resnet50",
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 20,
        "pretrained": True,
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 20,
        "pretrained": True,
    },
    "vit_base": {
        "timm_name": "vit_base_patch16_224",
        "batch_size": 16,
        "lr": 1e-4,
        "epochs": 20,
        "pretrained": True,
    },
}

WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 4
