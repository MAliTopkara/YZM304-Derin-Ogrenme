"""Dataset loaders. Uses torchvision ImageFolder over data/processed/{train,val,test}."""
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .config import (
    NUM_WORKERS,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from .transforms import build_eval_transform, build_train_transform


def build_datasets():
    train_ds = ImageFolder(TRAIN_DIR, transform=build_train_transform())
    val_ds = ImageFolder(VAL_DIR, transform=build_eval_transform())
    test_ds = ImageFolder(TEST_DIR, transform=build_eval_transform())
    return train_ds, val_ds, test_ds


def build_loaders(batch_size: int, num_workers: int = NUM_WORKERS):
    train_ds, val_ds, test_ds = build_datasets()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds.classes
