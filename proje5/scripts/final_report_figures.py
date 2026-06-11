"""Generate final report-ready figures from aggregated multi-seed results.

Produces:
  outputs/figures/roc_comparison_all_seeds.png    (one ROC per (model, seed), + mean per model)
  outputs/figures/confusion_{model}_seed{seed}.png  (per seed, at F1-optimal threshold)
  outputs/figures/metric_boxplot.png              (F1, recall, AUC boxplot across seeds per model)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES, FIGURES_DIR, LOGS_DIR, SEEDS, SUPPORTED_MODELS
from src.evaluate import plot_confusion_matrix
from src.utils import load_json


MODEL_COLORS = {
    "resnet50": "tab:blue",
    "densenet121": "tab:green",
    "efficientnet_b0": "tab:orange",
}


def plot_roc_all_seeds(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    # FPR grid for interpolating means.
    fpr_grid = np.linspace(0, 1, 101)
    for model_name in SUPPORTED_MODELS:
        tprs_interp = []
        aucs = []
        color = MODEL_COLORS[model_name]
        for seed in SEEDS:
            preds_path = LOGS_DIR / f"{model_name}_seed{seed}_test_preds.json"
            ext_path = LOGS_DIR / f"{model_name}_seed{seed}_extended.json"
            if not preds_path.exists():
                continue
            preds = load_json(preds_path)
            y_true = np.array(preds["y_true"])
            y_prob = np.array(preds["y_prob"])
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            tprs_interp.append(np.interp(fpr_grid, fpr, tpr))
            if ext_path.exists():
                aucs.append(load_json(ext_path)["test_metrics_t0.5"]["roc_auc"])
            ax.plot(fpr, tpr, color=color, alpha=0.25, linewidth=1)
        if tprs_interp:
            mean_tpr = np.mean(tprs_interp, axis=0)
            std_tpr = np.std(tprs_interp, axis=0)
            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            std_auc = float(np.std(aucs)) if aucs else 0.0
            ax.plot(
                fpr_grid, mean_tpr, color=color, linewidth=2.2,
                label=f"{model_name} (AUC={mean_auc:.3f}\u00b1{std_auc:.3f})",
            )
            ax.fill_between(
                fpr_grid, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                color=color, alpha=0.15,
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set (mean \u00b1 std across seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_box(out_path: Path) -> None:
    metrics = ["f1", "recall", "roc_auc"]
    data = {m: {met: [] for met in metrics} for m in SUPPORTED_MODELS}
    for model_name in SUPPORTED_MODELS:
        for seed in SEEDS:
            path = LOGS_DIR / f"{model_name}_seed{seed}_extended.json"
            if not path.exists():
                continue
            ext = load_json(path)["test_metrics_tF1opt"]
            for met in metrics:
                data[model_name][met].append(ext[met])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    met_labels = {"f1": "F1 (at t*(F1))", "recall": "Recall (at t*(F1))", "roc_auc": "ROC-AUC"}
    for ax, met in zip(axes, metrics):
        vals = [data[m][met] for m in SUPPORTED_MODELS]
        bp = ax.boxplot(vals, tick_labels=SUPPORTED_MODELS, showmeans=True, patch_artist=True)
        for patch, model_name in zip(bp["boxes"], SUPPORTED_MODELS):
            patch.set_facecolor(MODEL_COLORS[model_name])
            patch.set_alpha(0.5)
        ax.set_ylabel(met_labels[met])
        ax.set_title(met_labels[met])
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15)
    fig.suptitle("Per-model metric distribution across seeds")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusions_all_seeds() -> None:
    for model_name in SUPPORTED_MODELS:
        for seed in SEEDS:
            path = LOGS_DIR / f"{model_name}_seed{seed}_extended.json"
            if not path.exists():
                continue
            ext = load_json(path)
            cm = ext["test_metrics_tF1opt"]["confusion"]
            t = ext["test_metrics_tF1opt"]["threshold"]
            plot_confusion_matrix(
                cm, CLASS_NAMES,
                FIGURES_DIR / f"confusion_{model_name}_seed{seed}.png",
                title=f"{model_name} seed={seed} (t={t:.2f})",
            )


def main() -> None:
    plot_roc_all_seeds(FIGURES_DIR / "roc_comparison_all_seeds.png")
    print("Saved roc_comparison_all_seeds.png")
    plot_metric_box(FIGURES_DIR / "metric_boxplot.png")
    print("Saved metric_boxplot.png")
    plot_confusions_all_seeds()
    print("Saved per-seed confusion matrices")


if __name__ == "__main__":
    main()
