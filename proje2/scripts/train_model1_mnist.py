"""
Model 1 Eğitim Scripti — LeNet-5 Tabanlı CNN (MNIST)
=====================================================

Oturum kurallarına (KURAL 1-8) tam uyumlu.

Çıktılar
--------
    results/model1_weights.pth
    results/metrics/model1_metrics.json
    results/metrics/model1_classification_report.txt
    results/figures/model1_loss.png
    results/figures/model1_accuracy.png
    results/figures/model1_confusion_matrix.png
"""

# ---------------------------------------------------------------------------
# KURAL 1 — GPU doğrulaması (her şeyden önce)
# ---------------------------------------------------------------------------
import torch

assert torch.cuda.is_available(), (
    "HATA: CUDA bulunamadı. Eğitim GPU üzerinde yapılacak şekilde planlandı. "
    "nvidia-smi ile GPU'yu kontrol et, PyTorch'un CUDA versiyonunun doğru "
    "kurulu olduğunu doğrula."
)
device = torch.device("cuda")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")

# ---------------------------------------------------------------------------
# Importlar
# ---------------------------------------------------------------------------
import sys
import time
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# proje kökünü PYTHONPATH'e ekle (models/ ve scripts/ import'ları için)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.lenet5_base import LeNet5Base  # noqa: E402
from scripts.utils import (  # noqa: E402
    evaluate,
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_loss_curves,
    save_classification_report,
    save_metrics,
    set_seed,
    train_one_epoch,
)

# ---------------------------------------------------------------------------
# Sabitler (KURAL 2, 4, 6)
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 10           # KURAL 2
BATCH_SIZE = 128      # KURAL 4
LR = 1e-3
NUM_WORKERS = 4       # KURAL 3
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

WEIGHT_PATH = RESULTS_DIR / "model1_weights.pth"
METRICS_PATH = METRICS_DIR / "model1_metrics.json"
REPORT_PATH = METRICS_DIR / "model1_classification_report.txt"
LOSS_FIG = FIGURES_DIR / "model1_loss.png"
ACC_FIG = FIGURES_DIR / "model1_accuracy.png"
CM_FIG = FIGURES_DIR / "model1_confusion_matrix.png"

CLASS_NAMES = [str(i) for i in range(10)]


def main() -> None:
    t_start = time.time()

    # KURAL 6 — seed
    set_seed(SEED)

    # -----------------------------------------------------------------------
    # Veri seti + transformlar
    # -----------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Pad(2),                      # 28x28 → 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_data = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )

    print(f"Train örnek sayısı : {len(train_data)}")
    print(f"Test  örnek sayısı : {len(test_data)}")

    # KURAL 3 — DataLoader optimizasyonları
    common_loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = DataLoader(train_data, shuffle=True, **common_loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **common_loader_kwargs)

    # -----------------------------------------------------------------------
    # Model, loss, optimizer
    # -----------------------------------------------------------------------
    model = LeNet5Base().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {n_params:,}")

    # -----------------------------------------------------------------------
    # Eğitim döngüsü (KURAL 5: AMP yok — küçük model)
    # -----------------------------------------------------------------------
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_test_acc = 0.0
    final_preds = None
    final_labels = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=False
        )
        te_loss, te_acc, preds, labels = evaluate(
            model, test_loader, criterion, device
        )

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        best_test_acc = max(best_test_acc, te_acc)
        final_preds, final_labels = preds, labels

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.2f}%"
        )

    # -----------------------------------------------------------------------
    # Ağırlıkları, metrikleri ve grafikleri kaydet
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHT_PATH)
    print(f"Ağırlıklar kaydedildi: {WEIGHT_PATH}")

    t_end = time.time()
    training_time = t_end - t_start

    metrics = {
        "model_name": "Model 1 - LeNet5Base",
        "dataset": "MNIST",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "num_parameters": n_params,
        "train_loss_per_epoch": train_losses,
        "train_acc_per_epoch": train_accs,
        "test_loss_per_epoch": test_losses,
        "test_acc_per_epoch": test_accs,
        "final_test_loss": test_losses[-1],
        "final_test_accuracy": test_accs[-1],
        "best_test_accuracy": best_test_acc,
        "training_time_seconds": training_time,
    }
    save_metrics(metrics, str(METRICS_PATH))
    print(f"Metrikler kaydedildi: {METRICS_PATH}")

    save_classification_report(
        final_labels, final_preds, CLASS_NAMES, str(REPORT_PATH)
    )
    print(f"Classification report kaydedildi: {REPORT_PATH}")

    plot_loss_curves(
        train_losses, test_losses, str(LOSS_FIG),
        title="Model 1 (LeNet-5 Base) — Loss Curves",
    )
    plot_accuracy_curves(
        train_accs, test_accs, str(ACC_FIG),
        title="Model 1 (LeNet-5 Base) — Accuracy Curves",
    )
    plot_confusion_matrix(
        final_labels, final_preds, CLASS_NAMES, str(CM_FIG),
        title="Model 1 (LeNet-5 Base) — Confusion Matrix",
    )
    print(f"Grafikler kaydedildi: {LOSS_FIG}, {ACC_FIG}, {CM_FIG}")

    print(f"\nToplam eğitim süresi: {training_time:.2f} saniye")
    print(f"Final test accuracy : {test_accs[-1]:.2f}%")
    print(f"Best  test accuracy : {best_test_acc:.2f}%")

    # KURAL 7 — bellek temizliği
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
