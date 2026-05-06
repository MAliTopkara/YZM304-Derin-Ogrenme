"""Test seti üzerinde değerlendirme.

Çalıştırma:
    python -m src.evaluate --model resnet50
    python -m src.evaluate --model resnet50 --weights results/models/resnet50.pth

Üretir:
    results/figures/<model>_confusion_matrix.png
    results/figures/<model>_confusion_matrix_norm.png
    results/metrics.csv satırı (append; aynı model için üzerine yazar)
    results/<model>_classification_report.txt
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    METRICS_CSV,
    MODEL_CONFIGS,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.dataset import build_loaders
from src.models import count_parameters, create_model
from src.utils import get_device, model_size_mb, set_seed
from src.visualize import plot_confusion_matrix


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Returns y_true, y_pred, y_probs (np arrays) and inference time (sec/sample)."""
    model.eval()
    ys = []
    preds = []
    probs = []
    n = 0
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        preds.append(p.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
        n += x.size(0)
    elapsed = time.time() - t0
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_probs = np.concatenate(probs)
    inference_ms_per_sample = (elapsed / max(n, 1)) * 1000
    return y_true, y_pred, y_probs, inference_ms_per_sample


def upsert_metrics_row(model_name: str, row: dict) -> None:
    """metrics.csv içinde aynı model varsa güncelle, yoksa ekle."""
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = [
        "model", "accuracy", "macro_f1", "weighted_f1", "top3_accuracy",
        "params_m", "size_mb", "inference_ms_per_sample", "training_time_min",
    ]
    if METRICS_CSV.exists():
        with open(METRICS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("model") != model_name]

    rows.append({k: row.get(k, "") for k in fieldnames})

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--weights", default=None,
                    help="Default: results/models/<model>.pth")
    ap.add_argument("--training-time-min", type=float, default=None,
                    help="metrics.csv'a yazılır")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    cfg = MODEL_CONFIGS[args.model]

    weights_path = Path(args.weights) if args.weights else MODELS_DIR / f"{args.model}.pth"
    if not weights_path.exists():
        print(f"[ERROR] Ağırlık bulunamadı: {weights_path}")
        sys.exit(1)

    print(f"Device: {device}")
    print(f"Model: {args.model}, weights: {weights_path}")

    _, _, test_loader, classes = build_loaders(batch_size=cfg["batch_size"])
    print(f"Test batches: {len(test_loader)}, classes: {classes}")

    model = create_model(args.model, pretrained=False).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    print(f"Trainable params: {count_parameters(model)/1e6:.2f}M  size: {model_size_mb(model):.1f} MB")

    print("Test set üzerinde tahminler...")
    y_true, y_pred, y_probs, inf_ms = collect_predictions(model, test_loader, device)

    accuracy = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    top3 = top_k_accuracy_score(y_true, y_probs, k=3, labels=list(range(len(classes))))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0
    )

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Top-3 Acc:   {top3:.4f}")
    print(f"Inference:   {inf_ms:.2f} ms/sample (batched)")

    print("\nSınıf bazlı:")
    print(f"{'class':22s}  prec    rec     f1      support")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}  {precision[i]:.4f}  {recall[i]:.4f}  {f1[i]:.4f}  {support[i]}")

    # text report
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_path = RESULTS_DIR / f"{args.model}_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nClassification report: {report_path}")

    # confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_path = FIGURES_DIR / f"{args.model}_confusion_matrix.png"
    cm_norm_path = FIGURES_DIR / f"{args.model}_confusion_matrix_norm.png"
    plot_confusion_matrix(cm, cm_path, title=f"{args.model} — Confusion Matrix", normalize=False)
    plot_confusion_matrix(cm, cm_norm_path, title=f"{args.model} — Confusion Matrix (normalized)", normalize=True)
    print(f"Confusion matrices: {cm_path}, {cm_norm_path}")

    # save full per-class metrics as JSON
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    (RESULTS_DIR / f"{args.model}_per_class_metrics.json").write_text(
        json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # update metrics.csv
    row = {
        "model": args.model,
        "accuracy": f"{accuracy:.4f}",
        "macro_f1": f"{macro_f1:.4f}",
        "weighted_f1": f"{weighted_f1:.4f}",
        "top3_accuracy": f"{top3:.4f}",
        "params_m": f"{count_parameters(model)/1e6:.2f}",
        "size_mb": f"{model_size_mb(model):.1f}",
        "inference_ms_per_sample": f"{inf_ms:.2f}",
        "training_time_min": f"{args.training_time_min:.1f}" if args.training_time_min else "",
    }
    upsert_metrics_row(args.model, row)
    print(f"Metrics CSV: {METRICS_CSV}")


if __name__ == "__main__":
    main()
