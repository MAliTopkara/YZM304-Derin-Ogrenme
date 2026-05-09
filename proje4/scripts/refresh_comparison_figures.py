"""5-model karşılaştırması için sunum figürlerini yeniden üretir.

Üretir:
    results/figures/comparison_curves_overlay.png   (eğitim eğrileri overlay)
    results/figures/comparison_pareto.png           (acc vs boyut/hız)
    results/figures/comparison_confusion_grid.png   (5 confusion matrix grid)
    results/figures/comparison_macro_f1_bar.png     (yatay macro-F1 barchart)
    results/figures/comparison_class_f1.png         (sınıf-bazlı F1 dağılımı)

Çalıştırma:
    python scripts/refresh_comparison_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, FIGURES_DIR, LOGS_DIR, METRICS_CSV, RESULTS_DIR  # noqa: E402

# 5 model — pedagojik sıra (zayıftan güçlüye)
MODEL_NAMES = ["mlp", "cnn_scratch", "resnet50", "efficientnet_b0", "vit_base"]
DISPLAY = {
    "mlp": "MLP",
    "cnn_scratch": "CNN (Scratch)",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNetB0",
    "vit_base": "ViT-Base/16",
}
PARADIGM = {
    "mlp": "Tam Bağlı Baseline",
    "cnn_scratch": "CNN — Sıfırdan",
    "resnet50": "Klasik CNN — Transfer",
    "efficientnet_b0": "Modern CNN — Transfer",
    "vit_base": "Transformer — Transfer",
}
COLORS = {
    "mlp": "#64748b",          # slate-500
    "cnn_scratch": "#d97706",  # amber-600
    "resnet50": "#2563eb",     # blue-600
    "efficientnet_b0": "#16a34a",  # green-600
    "vit_base": "#dc2626",     # red-600
}


def fig1_curves_overlay() -> Path:
    histories = {}
    for name in MODEL_NAMES:
        hp = LOGS_DIR / f"{name}_history.csv"
        if hp.exists():
            histories[name] = pd.read_csv(hp)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_specs = [
        ("train_loss", "Train Loss", axes[0, 0]),
        ("val_loss",   "Val Loss",   axes[0, 1]),
        ("train_acc",  "Train Acc",  axes[1, 0]),
        ("val_acc",    "Val Acc",    axes[1, 1]),
    ]
    for col, title, ax in metric_specs:
        for name, h in histories.items():
            ax.plot(h["epoch"], h[col], label=DISPLAY[name],
                    color=COLORS[name], linewidth=2, marker="o", markersize=3)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    plt.suptitle("Eğitim Eğrileri — 5 Model Overlay", fontsize=14, y=1.00)
    plt.tight_layout()
    out = FIGURES_DIR / "comparison_curves_overlay.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig2_pareto(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # acc vs size
    for _, row in df.iterrows():
        name = row["model"]
        axes[0].scatter(
            row["size_mb"], row["macro_f1"],
            s=320, color=COLORS[name], edgecolor="black", linewidth=1.5,
            label=DISPLAY[name], zorder=3,
        )
        # label biraz offset, MLP altta kalmasın diye dikey ayarla
        dy = -10 if row["macro_f1"] > 0.9 else 14
        axes[0].annotate(
            DISPLAY[name],
            (row["size_mb"], row["macro_f1"]),
            xytext=(8, dy), textcoords="offset points", fontsize=10,
        )
    axes[0].set_xlabel("Model Boyutu (MB)", fontsize=11)
    axes[0].set_ylabel("Macro-F1", fontsize=11)
    axes[0].set_title("Macro-F1 vs Boyut", fontsize=13)
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale("log")

    # acc vs inference
    for _, row in df.iterrows():
        name = row["model"]
        axes[1].scatter(
            row["inference_ms_per_sample"], row["macro_f1"],
            s=320, color=COLORS[name], edgecolor="black", linewidth=1.5,
            label=DISPLAY[name], zorder=3,
        )
        dy = -10 if row["macro_f1"] > 0.9 else 14
        axes[1].annotate(
            DISPLAY[name],
            (row["inference_ms_per_sample"], row["macro_f1"]),
            xytext=(8, dy), textcoords="offset points", fontsize=10,
        )
    axes[1].set_xlabel("Inference Süresi (ms/örnek)", fontsize=11)
    axes[1].set_ylabel("Macro-F1", fontsize=11)
    axes[1].set_title("Macro-F1 vs Hız", fontsize=13)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Doğruluk vs Maliyet — Pareto (5 Model)", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "comparison_pareto.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_confusion_grid() -> Path:
    # 1 satır 5 sütun (geniş ekran sunum için ideal)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, name in zip(axes, MODEL_NAMES):
        path = FIGURES_DIR / f"{name}_confusion_matrix_norm.png"
        if not path.exists():
            ax.text(0.5, 0.5, f"missing: {name}", ha="center", va="center")
            ax.axis("off")
            continue
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(DISPLAY[name], fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    out = FIGURES_DIR / "comparison_confusion_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig4_macro_f1_bar(df: pd.DataFrame) -> Path:
    df = df.set_index("model").loc[MODEL_NAMES].reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    y = np.arange(len(df))
    bars = ax.barh(
        y, df["macro_f1"],
        color=[COLORS[n] for n in df["model"]],
        edgecolor="black", linewidth=1.0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[n] for n in df["model"]], fontsize=11)
    ax.invert_yaxis()  # en güçlü modeller üstte değil → MLP üstte (pedagojik sıra)
    ax.set_xlabel("Macro-F1", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    # değerleri bar üstüne yaz
    for i, (bar, val) in enumerate(zip(bars, df["macro_f1"])):
        ax.text(
            min(val + 0.015, 0.97), bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10, fontweight="bold",
        )
    # baseline ↔ transfer ayırıcı çizgi
    ax.axhline(y=1.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.02, 0.5, "Baseline", fontsize=9, color="gray", style="italic")
    ax.text(0.02, 3.5, "Transfer Learning", fontsize=9, color="gray", style="italic")
    ax.set_title("5 Modelin Macro-F1 Karşılaştırması", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "comparison_macro_f1_bar.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def fig5_class_f1() -> Path:
    """Her sınıf için 5 modelin F1 değerlerini grouped bar chart."""
    # Per-class JSON'ları oku
    per_class = {}
    for name in MODEL_NAMES:
        path = RESULTS_DIR / f"{name}_per_class_metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                per_class[name] = json.load(f)
    if not per_class:
        return None  # hiç veri yok

    classes = CLASS_NAMES
    n_models = len(per_class)
    n_classes = len(classes)
    bar_width = 0.15
    x = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, name in enumerate(MODEL_NAMES):
        if name not in per_class:
            continue
        f1s = [per_class[name][c]["f1"] for c in classes]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, f1s, bar_width,
            label=DISPLAY[name], color=COLORS[name],
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Sınıf-Bazlı F1 Karşılaştırması (5 Model × 10 Sınıf)", fontsize=13)
    ax.legend(loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.30))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "comparison_class_f1.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    df = pd.read_csv(METRICS_CSV)

    print("Üretilen figürler:")
    out1 = fig1_curves_overlay()
    print(f"  {out1.name}")
    out2 = fig2_pareto(df)
    print(f"  {out2.name}")
    out3 = fig3_confusion_grid()
    print(f"  {out3.name}")
    out4 = fig4_macro_f1_bar(df)
    print(f"  {out4.name}")
    out5 = fig5_class_f1()
    if out5:
        print(f"  {out5.name}")


if __name__ == "__main__":
    main()
