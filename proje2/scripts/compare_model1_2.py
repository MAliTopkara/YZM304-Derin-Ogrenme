"""
Model 1 vs Model 2 Karşılaştırma Grafiği
=========================================

results/metrics/model1_metrics.json ve results/metrics/model2_metrics.json
dosyalarından epoch-bazında metrikleri okur, tek figürde 2x2 subplot ile
karşılaştırır:

    ┌──────────────────┬──────────────────┐
    │  Train Loss      │  Test Loss       │
    ├──────────────────┼──────────────────┤
    │  Train Accuracy  │  Test Accuracy   │
    └──────────────────┴──────────────────┘

Çıktı:
    results/figures/model1_vs_model2_comparison.png

Bu script, train_model1_mnist.py ve train_model2_mnist.py tamamlandıktan
sonra çalıştırılmalıdır.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

M1_PATH = METRICS_DIR / "model1_metrics.json"
M2_PATH = METRICS_DIR / "model2_metrics.json"
OUT_PATH = FIGURES_DIR / "model1_vs_model2_comparison.png"


def _load(path: Path) -> dict:
    assert path.exists(), f"Eksik metrics dosyası: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    m1 = _load(M1_PATH)
    m2 = _load(M2_PATH)

    assert m1["epochs"] == m2["epochs"], (
        "Epoch sayıları farklı — karşılaştırma için eşit olmalı."
    )
    epochs = range(1, m1["epochs"] + 1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Model 1 (LeNet-5 Base) vs Model 2 (LeNet-5 + BN + Dropout)",
        fontsize=14, fontweight="bold",
    )

    # --- Sol üst: Train Loss -----------------------------------------------
    ax = axes[0, 0]
    ax.plot(epochs, m1["train_loss_per_epoch"], marker="o", label="Model 1")
    ax.plot(epochs, m2["train_loss_per_epoch"], marker="s", label="Model 2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Sağ üst: Test Loss ------------------------------------------------
    ax = axes[0, 1]
    ax.plot(epochs, m1["test_loss_per_epoch"], marker="o", label="Model 1")
    ax.plot(epochs, m2["test_loss_per_epoch"], marker="s", label="Model 2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_title("Test Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Sol alt: Train Accuracy -------------------------------------------
    ax = axes[1, 0]
    ax.plot(epochs, m1["train_acc_per_epoch"], marker="o", label="Model 1")
    ax.plot(epochs, m2["train_acc_per_epoch"], marker="s", label="Model 2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy (%)")
    ax.set_title("Training Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Sağ alt: Test Accuracy --------------------------------------------
    ax = axes[1, 1]
    ax.plot(epochs, m1["test_acc_per_epoch"], marker="o", label="Model 1")
    ax.plot(epochs, m2["test_acc_per_epoch"], marker="s", label="Model 2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"Karşılaştırma grafiği kaydedildi: {OUT_PATH}")

    # --- Özet tablo --------------------------------------------------------
    print("\n=== Karşılaştırma Özeti ===")
    print(f"{'Metric':<28s}{'Model 1':>14s}{'Model 2':>14s}")
    print("-" * 56)
    rows = [
        ("Parametre sayısı",        m1["num_parameters"],       m2["num_parameters"]),
        ("Final test acc (%)",      m1["final_test_accuracy"],  m2["final_test_accuracy"]),
        ("Best  test acc (%)",      m1["best_test_accuracy"],   m2["best_test_accuracy"]),
        ("Final test loss",         m1["final_test_loss"],      m2["final_test_loss"]),
        ("Training time (s)",       m1["training_time_seconds"],m2["training_time_seconds"]),
    ]
    for name, v1, v2 in rows:
        if isinstance(v1, float):
            print(f"{name:<28s}{v1:>14.4f}{v2:>14.4f}")
        else:
            print(f"{name:<28s}{v1:>14}{v2:>14}")


if __name__ == "__main__":
    main()
