"""
Model 4 — Özellik Çıkarma (Feature Extraction)
===============================================

Amaç
-----
Model 3 (eğitilmiş ResNet18) ağının son sınıflandırma katmanını (fc) çıkarıp
512-boyutlu ara temsillerini CIFAR-10 train ve test setlerinin tamamı için
hesaplayıp .npy dosyalarına kaydetmek.

Bu özellikler Adım 2'de klasik makine öğrenmesi modelleri (SVM ve Random
Forest) tarafından kullanılacaktır.

PDF gereksinimi
---------------
"özellik seti ve bunlara karşılık gelen label seti .npy uzantılı dosyalarda
oluşturulacaktır"
"Bu veri setlerinin boyutu ve uzunluğu açıkça projede yazdırılmalıdır"

Oturum kuralları
----------------
    KURAL 1 : CUDA zorunlu
    KURAL 3 : DataLoader (num_workers=4, pin_memory=True, persistent_workers=True)
    KURAL 4 : batch_size = 512 (feature extraction — geri yayılım yok, büyük
              batch güvenli)
    KURAL 6 : seed=42
    KURAL 7 : torch.cuda.empty_cache() sonunda

Çıktılar
--------
    results/features/X_train.npy   (50000, 512) float32
    results/features/y_train.npy   (50000,)     int64
    results/features/X_test.npy    (10000, 512) float32
    results/features/y_test.npy    (10000,)     int64
"""

# ---------------------------------------------------------------------------
# KURAL 1 — GPU doğrulaması
# ---------------------------------------------------------------------------
import torch

assert torch.cuda.is_available(), (
    "HATA: CUDA bulunamadı. Feature extraction GPU üzerinde yapılacak şekilde "
    "planlandı."
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

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.resnet_adapter import get_resnet18_cifar  # noqa: E402
from scripts.utils import set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 512          # KURAL 4 — feature extraction büyük batch güvenli
NUM_WORKERS = 4           # KURAL 3

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
WEIGHT_PATH = RESULTS_DIR / "model3_weights.pth"

X_TRAIN_PATH = FEATURES_DIR / "X_train.npy"
Y_TRAIN_PATH = FEATURES_DIR / "y_train.npy"
X_TEST_PATH = FEATURES_DIR / "X_test.npy"
Y_TEST_PATH = FEATURES_DIR / "y_test.npy"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """
    Verilen loader'daki tüm örnekleri modelden geçirip (N, 512) boyutunda
    feature dizisi ve (N,) boyutunda label dizisi döner.
    """
    model.eval()
    feats_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats = model(images)  # fc=Identity olduğu için (B, 512) döner
        feats_list.append(feats.cpu().numpy().astype(np.float32))
        labels_list.append(labels.numpy().astype(np.int64))

    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y


def main() -> None:
    t_start = time.time()

    set_seed(SEED)  # KURAL 6

    # -----------------------------------------------------------------------
    # Veri seti — AUGMENTATION YOK (deterministik feature üretimi)
    # -----------------------------------------------------------------------
    # Model 3 eğitiminde train-time'da augmentation (RandomCrop + Flip) vardı;
    # ancak feature extraction aşamasında hem train hem test için aynı
    # deterministik transform kullanıyoruz. Böylece klasik ML modeline giden
    # özellikler rastgelelik içermiyor, tekrar üretilebilir oluyor.
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_data = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=eval_transform
    )
    test_data = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=eval_transform
    )

    print(f"Train örnek sayısı : {len(train_data)}")
    print(f"Test  örnek sayısı : {len(test_data)}")

    # KURAL 3 — DataLoader optimizasyonları, shuffle=False (label sırası sabit)
    common_loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )
    train_loader = DataLoader(train_data, **common_loader_kwargs)
    test_loader = DataLoader(test_data, **common_loader_kwargs)

    # -----------------------------------------------------------------------
    # Model — Model 3 ağırlıklarını yükle, fc → Identity
    # -----------------------------------------------------------------------
    assert WEIGHT_PATH.exists(), (
        f"HATA: {WEIGHT_PATH} bulunamadı. Önce Model 3 eğitimini çalıştır."
    )

    model = get_resnet18_cifar(num_classes=10, pretrained=False)
    state_dict = torch.load(str(WEIGHT_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # Son fc katmanını Identity yap → forward 512-d vektör döner
    model.fc = nn.Identity()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Feature extractor parametre sayısı: {n_params:,} (fc kaldırıldı)")

    # -----------------------------------------------------------------------
    # Train / Test feature extraction
    # -----------------------------------------------------------------------
    print("\n[1/2] Train set feature extraction...")
    t0 = time.time()
    X_train, y_train = extract_features(model, train_loader)
    print(f"      süre: {time.time() - t0:.2f} sn")

    print("[2/2] Test set feature extraction...")
    t0 = time.time()
    X_test, y_test = extract_features(model, test_loader)
    print(f"      süre: {time.time() - t0:.2f} sn")

    # -----------------------------------------------------------------------
    # Kaydet
    # -----------------------------------------------------------------------
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(X_TRAIN_PATH), X_train)
    np.save(str(Y_TRAIN_PATH), y_train)
    np.save(str(X_TEST_PATH), X_test)
    np.save(str(Y_TEST_PATH), y_test)

    # -----------------------------------------------------------------------
    # PDF gereksinimi — shape, dtype, len TERMINAL'e yazdır
    # -----------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Kaydedilen özellik / label dosyalari (shape, dtype, len)")
    print("=" * 64)
    print(f"X_train : shape={X_train.shape}, dtype={X_train.dtype}, len={len(X_train)}")
    print(f"y_train : shape={y_train.shape}, dtype={y_train.dtype}, len={len(y_train)}")
    print(f"X_test  : shape={X_test.shape},  dtype={X_test.dtype}, len={len(X_test)}")
    print(f"y_test  : shape={y_test.shape},   dtype={y_test.dtype}, len={len(y_test)}")
    print("=" * 64)

    print("\nDosya yollari:")
    print(f"  {X_TRAIN_PATH}")
    print(f"  {Y_TRAIN_PATH}")
    print(f"  {X_TEST_PATH}")
    print(f"  {Y_TEST_PATH}")

    total = time.time() - t_start
    print(f"\nToplam sure: {total:.2f} sn ({total/60:.2f} dk)")

    # KURAL 7 — bellek temizliği
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
