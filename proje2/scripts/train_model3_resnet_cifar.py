"""
Model 3 Eğitim Scripti — ResNet18 (CIFAR-10)
=============================================

Oturum kurallarına tam uyumlu:
    KURAL 1 : CUDA zorunlu
    KURAL 2 : epochs = 50 SABİT, CosineAnnealingLR T_max=50
    KURAL 3 : DataLoader (num_workers=4, pin_memory=True, persistent_workers=True)
    KURAL 4 : batch_size = 256
    KURAL 5 : Mixed precision (torch.cuda.amp autocast + GradScaler) — ZORUNLU
    KURAL 6 : seed = 42 (tam kurulum)
    KURAL 7 : torch.cuda.empty_cache() sonunda — Model 4 öncesi kritik
    KURAL 8 : time.time() ile süre ölçümü, metrics.json'a yazılıyor

Hiperparametre tercih nedenleri (README'de de yer alacak):
    - Optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)
      → CIFAR-10 ResNet eğitiminin literatürdeki standart optimizer'ı
        (He et al. 2015, orijinal ResNet paper). Adam genellikle daha
        düşük final accuracy verir küçük görsellerde.
    - Scheduler: CosineAnnealingLR(T_max=50)
      → Oturum kuralı gereği T_max=50. LR'yi epoch boyunca 0'a yumuşak
        indirerek stabil yakınsama sağlar.
    - Loss: CrossEntropyLoss (ödev gereksinimi)
    - Veri artırma: RandomCrop(32, padding=4) + HorizontalFlip → CIFAR-10
      için de facto augmentasyon seti.

Çıktılar
--------
    results/model3_weights.pth
    results/metrics/model3_metrics.json
    results/metrics/model3_classification_report.txt
    results/figures/model3_loss.png
    results/figures/model3_accuracy.png
    results/figures/model3_confusion_matrix.png
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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet_adapter import get_resnet18_cifar  # noqa: E402
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
# Sabitler
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 50             # KURAL 2 — SABİT
BATCH_SIZE = 256        # KURAL 4
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4         # KURAL 3

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

WEIGHT_PATH = RESULTS_DIR / "model3_weights.pth"
METRICS_PATH = METRICS_DIR / "model3_metrics.json"
REPORT_PATH = METRICS_DIR / "model3_classification_report.txt"
LOSS_FIG = FIGURES_DIR / "model3_loss.png"
ACC_FIG = FIGURES_DIR / "model3_accuracy.png"
CM_FIG = FIGURES_DIR / "model3_confusion_matrix.png"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main() -> None:
    t_start = time.time()

    set_seed(SEED)  # KURAL 6

    # -----------------------------------------------------------------------
    # Veri seti + transformlar
    # -----------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_data = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=train_transform
    )
    test_data = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=test_transform
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
    # Model, loss, optimizer, scheduler, AMP scaler
    # -----------------------------------------------------------------------
    model = get_resnet18_cifar(num_classes=10, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()  # KURAL 5 — AMP

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {n_params:,}")

    # -----------------------------------------------------------------------
    # Eğitim döngüsü (KURAL 5 — AMP aktif)
    # -----------------------------------------------------------------------
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    lrs = []
    best_test_acc = 0.0
    final_preds = None
    final_labels = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=True, scaler=scaler,
        )
        te_loss, te_acc, preds, labels = evaluate(
            model, test_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        scheduler.step()

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            # Best ağırlığı da sakla
            torch.save(model.state_dict(), WEIGHT_PATH)
        final_preds, final_labels = preds, labels

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | lr={current_lr:.5f} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.2f}% | "
            f"best={best_test_acc:.2f}%"
        )

    # -----------------------------------------------------------------------
    # Kaydetme (son epoch ağırlığı üzerine best zaten yazıldı)
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Best ağırlıklar kaydedildi: {WEIGHT_PATH}")

    t_end = time.time()
    training_time = t_end - t_start

    metrics = {
        "model_name": "Model 3 - ResNet18 (CIFAR-10 adapted)",
        "dataset": "CIFAR-10",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "SGD",
        "scheduler": "CosineAnnealingLR",
        "scheduler_T_max": EPOCHS,
        "criterion": "CrossEntropyLoss",
        "mixed_precision": True,
        "pretrained": False,
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "num_parameters": n_params,
        "cifar_adapter": {
            "conv1": "Conv2d(3, 64, k=3, s=1, p=1, bias=False)",
            "maxpool": "Identity()",
            "fc": "Linear(512, 10)",
        },
        "lr_per_epoch": lrs,
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
        title="Model 3 (ResNet18) — Loss Curves",
    )
    plot_accuracy_curves(
        train_accs, test_accs, str(ACC_FIG),
        title="Model 3 (ResNet18) — Accuracy Curves",
    )
    plot_confusion_matrix(
        final_labels, final_preds, CLASS_NAMES, str(CM_FIG),
        title="Model 3 (ResNet18) — Confusion Matrix",
    )
    print(f"Grafikler kaydedildi: {LOSS_FIG}, {ACC_FIG}, {CM_FIG}")

    print(f"\nToplam eğitim süresi: {training_time:.2f} saniye ({training_time/60:.2f} dk)")
    print(f"Final test accuracy : {test_accs[-1]:.2f}%")
    print(f"Best  test accuracy : {best_test_acc:.2f}%")

    # KURAL 7 — bellek temizliği (Model 4 öncesi kritik)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
