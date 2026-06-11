"""Dataset, transforms and dataloader builders for FracAtlas binary classification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few FracAtlas JPEGs are truncated

from .config import IMAGENET_MEAN, IMAGENET_STD, SPLITS_DIR, TrainConfig


class FracAtlasDataset(Dataset):
    def __init__(self, csv_path: Path, transform: transforms.Compose | None = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, path


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Light medical-imaging-friendly augmentation."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def class_weights_tensor(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels)
    weights = counts.sum() / (len(counts) * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32)


def build_dataloaders(cfg: TrainConfig) -> dict[str, DataLoader]:
    train_tf = build_transforms(cfg.image_size, train=True)
    eval_tf = build_transforms(cfg.image_size, train=False)

    train_ds = FracAtlasDataset(SPLITS_DIR / "train.csv", train_tf)
    val_ds = FracAtlasDataset(SPLITS_DIR / "val.csv", eval_tf)
    test_ds = FracAtlasDataset(SPLITS_DIR / "test.csv", eval_tf)

    train_labels = train_ds.df["label"].to_numpy()
    if cfg.use_weighted_sampler:
        sampler = make_weighted_sampler(train_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
