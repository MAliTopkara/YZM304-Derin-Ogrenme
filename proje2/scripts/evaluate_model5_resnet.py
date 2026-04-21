"""
Model 5 — Uçtan Uca ResNet18 Değerlendirme
===========================================

PDF'den kritik cümle:
    "Eğer ilk 3 modelden biri seçilecekse aynı veri setleri kullanılacak
    olup 5. model kullanımına gerek yoktur."

Projede Model 3 (ResNet18) zaten CIFAR-10 üzerinde uçtan uca eğitildi.
Dolayısıyla Model 5 için YENİ BİR EĞİTİM YAPILMAZ; Model 3'ün ağırlıkları
tekrar kullanılır ve Model 4 (hibrit) ile APAÇIK karşılaştırılabilmesi
için aynı test pipeline'ı üzerinde eval edilir.

Oturum kuralları
----------------
    KURAL 1 : CUDA zorunlu
    KURAL 3 : DataLoader (num_workers=4, pin_memory=True, persistent_workers=True)
    KURAL 6 : seed=42
    KURAL 7 : torch.cuda.empty_cache() sonunda

Çıktılar
--------
    results/metrics/model5_metrics.json
    results/metrics/model5_classification_report.txt
    results/figures/model5_confusion_matrix.png

    results/metrics/model4_vs_model5_comparison.json
    results/figures/model4_vs_model5_bar.png
"""

# ---------------------------------------------------------------------------
# KURAL 1 — GPU doğrulaması
# ---------------------------------------------------------------------------
import torch

assert torch.cuda.is_available(), (
    "HATA: CUDA bulunamadi. Eval GPU uzerinde yapilacak sekilde planlandi."
)
device = torch.device("cuda")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")

# ---------------------------------------------------------------------------
# Importlar
# ---------------------------------------------------------------------------
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet_adapter import get_resnet18_cifar  # noqa: E402
from scripts.utils import (  # noqa: E402
    plot_confusion_matrix,
    save_classification_report,
    save_metrics,
    set_seed,
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 256          # Model 4 feature extraction ile tutarlı ölçek
NUM_WORKERS = 4           # KURAL 3

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

WEIGHT_PATH = RESULTS_DIR / "model3_weights.pth"

M5_METRICS_PATH = METRICS_DIR / "model5_metrics.json"
M5_REPORT_PATH = METRICS_DIR / "model5_classification_report.txt"
M5_CM_PATH = FIGURES_DIR / "model5_confusion_matrix.png"

M4_SVM_METRICS_PATH = METRICS_DIR / "model4_svm_metrics.json"
M4_RF_METRICS_PATH = METRICS_DIR / "model4_rf_metrics.json"

COMPARE_JSON_PATH = METRICS_DIR / "model4_vs_model5_comparison.json"
COMPARE_BAR_PATH = FIGURES_DIR / "model4_vs_model5_bar.png"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Test loader üzerinde predict toplar → (preds, labels) döner."""
    model.eval()
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        preds_list.append(preds.cpu().numpy())
        labels_list.append(labels.numpy())
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return preds, labels


def plot_comparison_bar(
    model_names: list[str],
    accs: list[float],
    save_path: Path,
    title: str = "Model 4 vs Model 5 - Test Accuracy",
) -> None:
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, [a * 100 for a in accs], color=colors[: len(model_names)])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    # Barların üstüne yüzde yaz
    for bar, acc in zip(bars, accs):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{acc*100:.2f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def main() -> None:
    t_start = time.time()

    set_seed(SEED)  # KURAL 6

    # -----------------------------------------------------------------------
    # Test set — Model 4 feature extraction ile AYNI transform/pipeline
    # -----------------------------------------------------------------------
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_data = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=eval_transform
    )
    print(f"Test ornek sayisi : {len(test_data)}")

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )

    # -----------------------------------------------------------------------
    # Model 3'ün ağırlıkları → tam ResNet18 (fc 10 sınıflı)
    # -----------------------------------------------------------------------
    assert WEIGHT_PATH.exists(), (
        f"HATA: {WEIGHT_PATH} bulunamadi. Once Model 3 egitimini calistir."
    )
    model = get_resnet18_cifar(num_classes=10, pretrained=False)
    state_dict = torch.load(str(WEIGHT_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model 5 parametre sayisi: {n_params:,}")

    # -----------------------------------------------------------------------
    # Eval
    # -----------------------------------------------------------------------
    t0 = time.time()
    preds, labels = evaluate_model(model, test_loader)
    eval_time = time.time() - t0

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"\nModel 5 test accuracy : {acc * 100:.2f}%")
    print(f"Model 5 macro F1      : {macro_f1:.4f}")
    print(f"Eval suresi           : {eval_time:.2f} sn")

    # -----------------------------------------------------------------------
    # Kaydet — Model 5
    # -----------------------------------------------------------------------
    m5_metrics = {
        "model_name": "Model 5 - End-to-End ResNet18 (CIFAR-10)",
        "note": (
            "Model 3 agirliklari kullanildi, odev izni uyarinca yeni egitim "
            "yapilmadi (PDF: 'ilk 3 modelden biri secilecekse ayni veri "
            "setleri kullanilacak olup 5. model kullanimina gerek yoktur')."
        ),
        "weights_source": str(WEIGHT_PATH.name),
        "test_size": int(len(labels)),
        "num_params": int(n_params),
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "eval_time_seconds": float(eval_time),
    }
    save_metrics(m5_metrics, str(M5_METRICS_PATH))
    save_classification_report(labels, preds, CLASS_NAMES, str(M5_REPORT_PATH))
    plot_confusion_matrix(
        labels, preds, CLASS_NAMES, str(M5_CM_PATH),
        title="Model 5 (End-to-End ResNet18) - Confusion Matrix",
    )
    print(f"\nKaydedildi: {M5_METRICS_PATH.name}, {M5_REPORT_PATH.name}, "
          f"{M5_CM_PATH.name}")

    # -----------------------------------------------------------------------
    # Model 4 vs Model 5 karsilastirma
    # -----------------------------------------------------------------------
    assert M4_SVM_METRICS_PATH.exists() and M4_RF_METRICS_PATH.exists(), (
        "HATA: Model 4 metrik dosyalari bulunamadi. Once train_model4_hybrid "
        "scriptini calistir."
    )
    with open(M4_SVM_METRICS_PATH, "r", encoding="utf-8") as f:
        m4_svm = json.load(f)
    with open(M4_RF_METRICS_PATH, "r", encoding="utf-8") as f:
        m4_rf = json.load(f)

    compare = {
        "model4_svm": {
            "accuracy": float(m4_svm["accuracy"]),
            "macro_f1": float(m4_svm["macro_f1"]),
        },
        "model4_rf": {
            "accuracy": float(m4_rf["accuracy"]),
            "macro_f1": float(m4_rf["macro_f1"]),
        },
        "model5_cnn": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
        },
    }
    save_metrics(compare, str(COMPARE_JSON_PATH))

    plot_comparison_bar(
        model_names=["SVM", "Random Forest", "End-to-End CNN"],
        accs=[
            compare["model4_svm"]["accuracy"],
            compare["model4_rf"]["accuracy"],
            compare["model5_cnn"]["accuracy"],
        ],
        save_path=COMPARE_BAR_PATH,
        title="Model 4 vs Model 5 - Test Accuracy (CIFAR-10)",
    )
    print(f"Karsilastirma kaydedildi: {COMPARE_JSON_PATH.name}, "
          f"{COMPARE_BAR_PATH.name}")

    # -----------------------------------------------------------------------
    # Ozet tablo
    # -----------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Model 4 vs Model 5 — Test Seti Performansi (CIFAR-10, 10k ornek)")
    print("=" * 64)
    print(f"{'Model':<32}{'Accuracy':>12}{'Macro F1':>12}")
    print("-" * 64)
    print(f"{'4a) SVM (RBF, 10k subset)':<32}"
          f"{compare['model4_svm']['accuracy']*100:>11.2f}%"
          f"{compare['model4_svm']['macro_f1']:>12.4f}")
    print(f"{'4b) Random Forest (50k)':<32}"
          f"{compare['model4_rf']['accuracy']*100:>11.2f}%"
          f"{compare['model4_rf']['macro_f1']:>12.4f}")
    print(f"{'5 ) End-to-End ResNet18':<32}"
          f"{compare['model5_cnn']['accuracy']*100:>11.2f}%"
          f"{compare['model5_cnn']['macro_f1']:>12.4f}")
    print("=" * 64)

    total = time.time() - t_start
    print(f"\nToplam sure: {total:.2f} sn")

    # KURAL 7 — bellek temizligi
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
