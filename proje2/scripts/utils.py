"""
Ortak yardımcı fonksiyonlar
============================

Tüm training/evaluate scriptlerinin paylaşacağı fonksiyonlar:
    - set_seed                  : Reproducibility (KURAL 6)
    - get_device                : CUDA şart (KURAL 1)
    - train_one_epoch           : tek epoch eğitim (opsiyonel AMP)
    - evaluate                  : test/validation değerlendirme
    - plot_loss_curves          : train/test loss eğrisi
    - plot_accuracy_curves      : train/test accuracy eğrisi
    - plot_confusion_matrix     : sklearn + seaborn ısı haritası
    - save_metrics              : metrics.json
    - save_classification_report: sklearn classification_report → .txt
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.cuda.amp import autocast, GradScaler  # noqa: F401
except Exception:  # pragma: no cover
    autocast = None
    GradScaler = None


# ---------------------------------------------------------------------------
# Reproducibility & Device
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Tüm rastgelelik kaynakları için sabit seed (KURAL 6)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """CUDA zorunlu (KURAL 1). Yoksa assert ile projeyi durdur."""
    assert torch.cuda.is_available(), (
        "HATA: CUDA bulunamadı. Eğitim GPU üzerinde yapılacak şekilde planlandı. "
        "nvidia-smi ile GPU'yu kontrol et, PyTorch'un CUDA versiyonunun doğru "
        "kurulu olduğunu doğrula."
    )
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional["GradScaler"] = None,
) -> Tuple[float, float]:
    """
    Tek epoch eğitim.

    use_amp=True ise torch.cuda.amp.autocast + GradScaler kullanılır
    (KURAL 5 - Model 3 için).

    Returns
    -------
    (avg_loss, accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            assert scaler is not None, "use_amp=True ise scaler zorunludur."
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.detach(), dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Modelin loader üzerindeki performansını ölçer.

    Returns
    -------
    (avg_loss, accuracy_percent, all_preds, all_labels)
    """
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    preds_arr = np.concatenate(all_preds, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return avg_loss, acc, preds_arr, labels_arr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _ensure_parent(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_loss_curves(
    train_losses: Sequence[float],
    test_losses: Sequence[float],
    save_path: str,
    title: str = "Loss Curves",
) -> None:
    import matplotlib.pyplot as plt

    _ensure_parent(save_path)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, test_losses, marker="s", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_curves(
    train_accs: Sequence[float],
    test_accs: Sequence[float],
    save_path: str,
    title: str = "Accuracy Curves",
) -> None:
    import matplotlib.pyplot as plt

    _ensure_parent(save_path)
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker="o", label="Train Acc (%)")
    plt.plot(epochs, test_accs, marker="s", label="Test Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    _ensure_parent(save_path)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_metrics(metrics_dict: dict, save_path: str) -> None:
    """metrics.json dosyası olarak kaydet."""
    _ensure_parent(save_path)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False, default=_default)


def save_classification_report(
    labels: np.ndarray,
    preds: np.ndarray,
    target_names: Sequence[str],
    save_path: str,
) -> None:
    """sklearn classification_report çıktısını .txt'e yaz."""
    from sklearn.metrics import classification_report

    _ensure_parent(save_path)
    report = classification_report(labels, preds, target_names=list(target_names), digits=4)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
