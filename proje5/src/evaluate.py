"""Evaluation metrics, confusion matrix and ROC curve utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    y_prob: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion": self.confusion,
        }


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    for batch in loader:
        imgs, labels, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    y_true, y_pred, y_prob = run_inference(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return EvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        confusion=cm.tolist(),
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        y_prob=y_prob.tolist(),
    )


def compute_metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
) -> dict:
    """Recompute full metric set at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion": cm.tolist(),
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, criterion: str = "f1",
) -> tuple[float, float]:
    """Scan thresholds 0.01..0.99 and return (best_threshold, best_score)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if criterion == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif criterion == "youden":
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:
            raise ValueError(criterion)
        if score > best_s:
            best_s, best_t = float(score), float(t)
    return best_t, best_s


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True,
    )


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, str(cm_arr[i, j]),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(
    results: dict[str, EvalResult],
    out_path: Path,
    title: str = "ROC Curves",
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res.y_true, res.y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={res.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
