"""Plot learning curves (train loss + val F1) per model, averaged over seeds.

Writes:
  outputs/figures/learning_curves_{model}.png
  outputs/figures/learning_curves_combined.png  (all models on one panel)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, LOGS_DIR, SEEDS, SUPPORTED_MODELS
from src.utils import load_json


def collect_histories(model_name: str) -> list[list[dict]]:
    hists = []
    for seed in SEEDS:
        p = LOGS_DIR / f"{model_name}_seed{seed}_summary.json"
        if p.exists():
            hists.append(load_json(p)["history"])
    return hists


def pad_to_max(arr: list[list[float]]) -> np.ndarray:
    """Pad shorter runs with NaN (for early-stopped seeds) then stack."""
    max_len = max(len(a) for a in arr)
    padded = np.full((len(arr), max_len), np.nan)
    for i, a in enumerate(arr):
        padded[i, : len(a)] = a
    return padded


def plot_single(model_name: str, histories: list[list[dict]], out_path: Path) -> None:
    if not histories:
        return
    train_loss = pad_to_max([[h["train_loss"] for h in hist] for hist in histories])
    val_f1 = pad_to_max([[h["val_f1"] for h in hist] for hist in histories])
    epochs = np.arange(1, train_loss.shape[1] + 1)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    mean_loss = np.nanmean(train_loss, axis=0)
    std_loss = np.nanstd(train_loss, axis=0)
    ax1.plot(epochs, mean_loss, color="tab:blue", label="Train loss (mean)")
    ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    mean_f1 = np.nanmean(val_f1, axis=0)
    std_f1 = np.nanstd(val_f1, axis=0)
    ax2.plot(epochs, mean_f1, color="tab:red", label="Val F1 (mean)")
    ax2.fill_between(epochs, mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.2, color="tab:red")
    ax2.set_ylabel("Val F1", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Learning curves — {model_name} (n={len(histories)} seeds)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined(histories_by_model: dict[str, list[list[dict]]], out_path: Path) -> None:
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"resnet50": "tab:blue", "densenet121": "tab:green", "efficientnet_b0": "tab:orange"}
    for model_name, hists in histories_by_model.items():
        if not hists:
            continue
        train_loss = pad_to_max([[h["train_loss"] for h in hist] for hist in hists])
        val_f1 = pad_to_max([[h["val_f1"] for h in hist] for hist in hists])
        epochs = np.arange(1, train_loss.shape[1] + 1)
        c = colors.get(model_name, None)
        ax_loss.plot(epochs, np.nanmean(train_loss, axis=0), label=model_name, color=c)
        ax_loss.fill_between(
            epochs,
            np.nanmean(train_loss, axis=0) - np.nanstd(train_loss, axis=0),
            np.nanmean(train_loss, axis=0) + np.nanstd(train_loss, axis=0),
            alpha=0.15, color=c,
        )
        ax_f1.plot(epochs, np.nanmean(val_f1, axis=0), label=model_name, color=c)
        ax_f1.fill_between(
            epochs,
            np.nanmean(val_f1, axis=0) - np.nanstd(val_f1, axis=0),
            np.nanmean(val_f1, axis=0) + np.nanstd(val_f1, axis=0),
            alpha=0.15, color=c,
        )
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Train loss"); ax_loss.legend()
    ax_loss.set_title("Training loss (mean \u00b1 std)")
    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Val F1"); ax_f1.legend()
    ax_f1.set_title("Validation F1 (mean \u00b1 std)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    histories_by_model: dict[str, list[list[dict]]] = {}
    for model_name in SUPPORTED_MODELS:
        hists = collect_histories(model_name)
        histories_by_model[model_name] = hists
        if hists:
            plot_single(model_name, hists, FIGURES_DIR / f"learning_curves_{model_name}.png")
            print(f"Saved learning_curves_{model_name}.png (n={len(hists)} seeds)")
    plot_combined(histories_by_model, FIGURES_DIR / "learning_curves_combined.png")
    print("Saved learning_curves_combined.png")


if __name__ == "__main__":
    main()
