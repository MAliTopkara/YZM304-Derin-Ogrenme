"""
Model 2 Eğitim Scripti — İyileştirilmiş LeNet-5 (MNIST)
========================================================

Model 1 ile TAMAMEN AYNI eğitim kurulumu (seed, lr, optimizer, epoch,
batch_size, transform, train/test split). Tek fark:
    - Mimari: LeNet5Improved (BatchNorm2d + Dropout eklenmiş)

Böylece iki model ADİL biçimde karşılaştırılabilir.

Çıktılar
--------
    results/model2_weights.pth
    results/metrics/model2_metrics.json
    results/metrics/model2_classification_report.txt
    results/figures/model2_loss.png
    results/figures/model2_accuracy.png
    results/figures/model2_confusion_matrix.png
"""

# ---------------------------------------------------------------------------
# KURAL 1 — GPU doğrulaması
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.lenet5_improved import LeNet5Improved  # noqa: E402
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
# Sabitler (Model 1 ile birebir AYNI — karşılaştırma adaleti için)
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 10            # KURAL 2
BATCH_SIZE = 128       # KURAL 4
LR = 1e-3
NUM_WORKERS = 4        # KURAL 3
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

WEIGHT_PATH = RESULTS_DIR / "model2_weights.pth"
METRICS_PATH = METRICS_DIR / "model2_metrics.json"
REPORT_PATH = METRICS_DIR / "model2_classification_report.txt"
LOSS_FIG = FIGURES_DIR / "model2_loss.png"
ACC_FIG = FIGURES_DIR / "model2_accuracy.png"
CM_FIG = FIGURES_DIR / "model2_confusion_matrix.png"

CLASS_NAMES = [str(i) for i in range(10)]


def main() -> None:
    t_start = time.time()

    set_seed(SEED)

    # -----------------------------------------------------------------------
    # Veri seti (Model 1 ile aynı transform)
    # -----------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Pad(2),
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
    model = LeNet5Improved().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {n_params:,}")

    # -----------------------------------------------------------------------
    # Eğitim döngüsü
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
    # Kaydetme
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHT_PATH)
    print(f"Ağırlıklar kaydedildi: {WEIGHT_PATH}")

    t_end = time.time()
    training_time = t_end - t_start

    metrics = {
        "model_name": "Model 2 - LeNet5Improved (BN+Dropout)",
        "dataset": "MNIST",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "num_parameters": n_params,
        "additions_over_model1": ["BatchNorm2d after each Conv2d", "Dropout(p=0.5) before each FC"],
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
        title="Model 2 (LeNet-5 + BN + Dropout) — Loss Curves",
    )
    plot_accuracy_curves(
        train_accs, test_accs, str(ACC_FIG),
        title="Model 2 (LeNet-5 + BN + Dropout) — Accuracy Curves",
    )
    plot_confusion_matrix(
        final_labels, final_preds, CLASS_NAMES, str(CM_FIG),
        title="Model 2 (LeNet-5 + BN + Dropout) — Confusion Matrix",
    )
    print(f"Grafikler kaydedildi: {LOSS_FIG}, {ACC_FIG}, {CM_FIG}")

    print(f"\nToplam eğitim süresi: {training_time:.2f} saniye")
    print(f"Final test accuracy : {test_accs[-1]:.2f}%")
    print(f"Best  test accuracy : {best_test_acc:.2f}%")

    torch.cuda.empty_cache()  # KURAL 7


if __name__ == "__main__":
    main()
