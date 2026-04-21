"""
Model 4 — Hibrit Sınıflandırıcı (CNN Features + Klasik ML)
===========================================================

extract_features_model4.py tarafından üretilen 512-boyutlu ResNet18
özellikleri üzerinde İKİ klasik makine öğrenmesi modeli eğitiyoruz:

    1. SVM  (sklearn.svm.SVC, RBF kernel)
    2. Random Forest (sklearn.ensemble.RandomForestClassifier)

PDF'den ilgili cümle:
    "kanonik bir makine öğrenmesi modeli (destek vektör makineleri,
    rastsal ağaçlar vb.) bu veriler ile eğitilip test edilecektir"

SVM ile 50 000 örnekle RBF eğitimi pratik değil (O(N^2)–O(N^3) karmaşıklık).
Bu yüzden SVM için her sınıftan 1 000 örnek seçerek dengeli 10 000'lik bir
alt küme oluşturuyoruz (np.random seed=42). Test her iki modelde de 10 000
örneğin tamamı üzerinde yapılıyor.

Çıktılar
--------
    results/metrics/model4_svm_metrics.json
    results/metrics/model4_svm_classification_report.txt
    results/figures/model4_svm_confusion_matrix.png

    results/metrics/model4_rf_metrics.json
    results/metrics/model4_rf_classification_report.txt
    results/figures/model4_rf_confusion_matrix.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.utils import (  # noqa: E402
    plot_confusion_matrix,
    save_classification_report,
    save_metrics,
)

# sklearn importları (yavaş — dosya başında yapıyoruz ki tekrarlanmasın)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
SEED = 42
SVM_PER_CLASS = 1000  # her sınıftan kaç örnek (dengeli subset)
NUM_CLASSES = 10

RESULTS_DIR = ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

X_TRAIN_PATH = FEATURES_DIR / "X_train.npy"
Y_TRAIN_PATH = FEATURES_DIR / "y_train.npy"
X_TEST_PATH = FEATURES_DIR / "X_test.npy"
Y_TEST_PATH = FEATURES_DIR / "y_test.npy"

SVM_METRICS_PATH = METRICS_DIR / "model4_svm_metrics.json"
SVM_REPORT_PATH = METRICS_DIR / "model4_svm_classification_report.txt"
SVM_CM_PATH = FIGURES_DIR / "model4_svm_confusion_matrix.png"

RF_METRICS_PATH = METRICS_DIR / "model4_rf_metrics.json"
RF_REPORT_PATH = METRICS_DIR / "model4_rf_classification_report.txt"
RF_CM_PATH = FIGURES_DIR / "model4_rf_confusion_matrix.png"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def balanced_subset(
    X: np.ndarray, y: np.ndarray, per_class: int, num_classes: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Her sınıftan `per_class` örnek seçerek dengeli alt küme oluşturur.
    np.random.RandomState(seed) ile deterministic.
    """
    rng = np.random.RandomState(seed)
    idx_list: list[np.ndarray] = []
    for c in range(num_classes):
        cls_idx = np.where(y == c)[0]
        if len(cls_idx) < per_class:
            raise ValueError(
                f"Sınıf {c} için yalnızca {len(cls_idx)} örnek var, "
                f"{per_class} istenmişti."
            )
        chosen = rng.choice(cls_idx, size=per_class, replace=False)
        idx_list.append(chosen)
    idx = np.concatenate(idx_list, axis=0)
    rng.shuffle(idx)  # sınıfların arka arkaya gelmesini önle
    return X[idx], y[idx]


def main() -> None:
    # -----------------------------------------------------------------------
    # .npy dosyalarını yükle
    # -----------------------------------------------------------------------
    for p in (X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH):
        assert p.exists(), (
            f"HATA: {p} bulunamadi. Once extract_features_model4.py calistir."
        )

    X_train = np.load(str(X_TRAIN_PATH))
    y_train = np.load(str(Y_TRAIN_PATH))
    X_test = np.load(str(X_TEST_PATH))
    y_test = np.load(str(Y_TEST_PATH))

    # PDF gereksinimi — shape / dtype / len tekrar yazdır
    print("=" * 64)
    print("Yuklenen ozellik / label dosyalari (shape, dtype, len)")
    print("=" * 64)
    print(f"X_train : shape={X_train.shape}, dtype={X_train.dtype}, len={len(X_train)}")
    print(f"y_train : shape={y_train.shape}, dtype={y_train.dtype}, len={len(y_train)}")
    print(f"X_test  : shape={X_test.shape},  dtype={X_test.dtype}, len={len(X_test)}")
    print(f"y_test  : shape={y_test.shape},   dtype={y_test.dtype}, len={len(y_test)}")
    print("=" * 64)

    # -----------------------------------------------------------------------
    # (a) SVM — dengeli subset + StandardScaler + RBF
    # -----------------------------------------------------------------------
    print("\n" + "#" * 64)
    print("# Model 4a — SVM (RBF)")
    print("#" * 64)

    X_svm, y_svm = balanced_subset(
        X_train, y_train,
        per_class=SVM_PER_CLASS, num_classes=NUM_CLASSES, seed=SEED,
    )
    print(f"SVM train subset : {X_svm.shape} (her siniftan {SVM_PER_CLASS})")

    # StandardScaler — SVM için kritik (RBF kernel ölçeğe duyarlı)
    scaler = StandardScaler()
    X_svm_sc = scaler.fit_transform(X_svm)
    X_test_sc = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)

    t0 = time.time()
    svm.fit(X_svm_sc, y_svm)
    svm_fit_time = time.time() - t0
    print(f"SVM fit suresi   : {svm_fit_time:.2f} sn")

    t0 = time.time()
    svm_preds = svm.predict(X_test_sc)
    svm_pred_time = time.time() - t0

    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average="macro")
    print(f"SVM test accuracy : {svm_acc * 100:.2f}%")
    print(f"SVM macro F1      : {svm_f1:.4f}")
    print(f"SVM predict suresi: {svm_pred_time:.2f} sn")

    svm_metrics = {
        "model_name": "Model 4a - ResNet18 features + SVM (RBF)",
        "classifier": "sklearn.svm.SVC",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "scaler": "StandardScaler (fit on train subset)",
        "train_subset_size": int(len(y_svm)),
        "train_subset_per_class": SVM_PER_CLASS,
        "test_size": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "seed": SEED,
        "accuracy": float(svm_acc),
        "macro_f1": float(svm_f1),
        "fit_time_seconds": float(svm_fit_time),
        "predict_time_seconds": float(svm_pred_time),
    }
    save_metrics(svm_metrics, str(SVM_METRICS_PATH))
    save_classification_report(
        y_test, svm_preds, CLASS_NAMES, str(SVM_REPORT_PATH)
    )
    plot_confusion_matrix(
        y_test, svm_preds, CLASS_NAMES, str(SVM_CM_PATH),
        title="Model 4a (SVM) - Confusion Matrix",
    )
    print(f"Kaydedildi: {SVM_METRICS_PATH.name}, {SVM_REPORT_PATH.name}, {SVM_CM_PATH.name}")

    # -----------------------------------------------------------------------
    # (b) Random Forest — tüm train seti, ölçeksiz
    # -----------------------------------------------------------------------
    print("\n" + "#" * 64)
    print("# Model 4b — Random Forest")
    print("#" * 64)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED,
    )
    print(f"RF train set     : {X_train.shape} (tam set)")

    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_fit_time = time.time() - t0
    print(f"RF fit suresi    : {rf_fit_time:.2f} sn")

    t0 = time.time()
    rf_preds = rf.predict(X_test)
    rf_pred_time = time.time() - t0

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="macro")
    print(f"RF test accuracy : {rf_acc * 100:.2f}%")
    print(f"RF macro F1      : {rf_f1:.4f}")
    print(f"RF predict suresi: {rf_pred_time:.2f} sn")

    rf_metrics = {
        "model_name": "Model 4b - ResNet18 features + Random Forest",
        "classifier": "sklearn.ensemble.RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": None,
        "n_jobs": -1,
        "scaler": None,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "seed": SEED,
        "accuracy": float(rf_acc),
        "macro_f1": float(rf_f1),
        "fit_time_seconds": float(rf_fit_time),
        "predict_time_seconds": float(rf_pred_time),
    }
    save_metrics(rf_metrics, str(RF_METRICS_PATH))
    save_classification_report(
        y_test, rf_preds, CLASS_NAMES, str(RF_REPORT_PATH)
    )
    plot_confusion_matrix(
        y_test, rf_preds, CLASS_NAMES, str(RF_CM_PATH),
        title="Model 4b (Random Forest) - Confusion Matrix",
    )
    print(f"Kaydedildi: {RF_METRICS_PATH.name}, {RF_REPORT_PATH.name}, {RF_CM_PATH.name}")

    # -----------------------------------------------------------------------
    # Ozet tablo
    # -----------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Model 4 — Hibrit Siniflandirici Ozet")
    print("=" * 64)
    print(f"{'Model':<28}{'Accuracy':>12}{'Macro F1':>12}{'Fit time (s)':>14}")
    print("-" * 64)
    print(f"{'4a) SVM (RBF, 10k subset)':<28}{svm_acc*100:>11.2f}%{svm_f1:>12.4f}{svm_fit_time:>14.2f}")
    print(f"{'4b) Random Forest (50k)':<28}{rf_acc*100:>11.2f}%{rf_f1:>12.4f}{rf_fit_time:>14.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
