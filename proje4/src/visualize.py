"""Eğitim eğrileri ve confusion matrix görselleştirme yardımcıları."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import CLASS_NAMES


def plot_training_curves(history: dict, out_path: Path, title: str = "") -> None:
    """history: {'train_loss','val_loss','train_acc','val_acc','lr'} listeleri."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", color="#4C72B0")
    axes[0].plot(epochs, history["val_loss"], label="val", color="#DD8452")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train", color="#4C72B0")
    axes[1].plot(epochs, history["val_acc"], label="val", color="#DD8452")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
    title: str = "",
    normalize: bool = True,
) -> None:
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        data = cm_norm
        fmt = ".2f"
        cbar_label = "Oran"
    else:
        data = cm
        fmt = "d"
        cbar_label = "Adet"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": cbar_label},
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
