"""Post-process all trained runs:
  - Re-infer on val set to get y_prob for threshold selection
  - Pick optimal threshold by F1 and by Youden's J (both reported)
  - Apply threshold to existing test predictions; recompute metrics
  - Save extended per-run metrics to outputs/logs/{run}_extended.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    LOGS_DIR,
    SEEDS,
    SUPPORTED_MODELS,
    TrainConfig,
)
from src.data import build_dataloaders
from src.evaluate import (
    compute_metrics_at_threshold,
    find_optimal_threshold,
    per_class_report,
    run_inference,
)
from src.models import build_model
from src.utils import get_device, load_json, save_json


def process_run(model_name: str, seed: int, device: torch.device) -> dict | None:
    run = f"{model_name}_seed{seed}"
    ckpt_path = CHECKPOINTS_DIR / f"{run}_best.pt"
    preds_path = LOGS_DIR / f"{run}_test_preds.json"
    if not ckpt_path.exists() or not preds_path.exists():
        return None

    cfg = TrainConfig(model_name=model_name, seed=seed)
    loaders = build_dataloaders(cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _ = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # Val inference for threshold selection.
    y_val_true, _, y_val_prob = run_inference(model, loaders["val"], device)
    t_f1, s_f1 = find_optimal_threshold(y_val_true, y_val_prob, "f1")
    t_yj, s_yj = find_optimal_threshold(y_val_true, y_val_prob, "youden")

    # Load existing test predictions.
    test = load_json(preds_path)
    y_test_true = np.array(test["y_true"])
    y_test_prob = np.array(test["y_prob"])

    metrics_05 = compute_metrics_at_threshold(y_test_true, y_test_prob, 0.5)
    metrics_f1 = compute_metrics_at_threshold(y_test_true, y_test_prob, t_f1)
    metrics_yj = compute_metrics_at_threshold(y_test_true, y_test_prob, t_yj)

    # Per-class report at F1-optimal threshold.
    y_test_pred_f1 = (y_test_prob >= t_f1).astype(int)
    report = per_class_report(y_test_true, y_test_pred_f1, CLASS_NAMES)

    extended = {
        "run": run,
        "model_name": model_name,
        "seed": seed,
        "val_best_threshold_f1": {"threshold": t_f1, "val_score": s_f1},
        "val_best_threshold_youden": {"threshold": t_yj, "val_score": s_yj},
        "test_metrics_t0.5": metrics_05,
        "test_metrics_tF1opt": metrics_f1,
        "test_metrics_tYouden": metrics_yj,
        "per_class_report_tF1opt": report,
    }
    out = LOGS_DIR / f"{run}_extended.json"
    save_json(extended, out)
    print(
        f"{run}: t*(F1)={t_f1:.2f} | "
        f"test@0.5 F1={metrics_05['f1']:.3f} | "
        f"test@F1opt F1={metrics_f1['f1']:.3f} rec={metrics_f1['recall']:.3f} | "
        f"test@Youden rec={metrics_yj['recall']:.3f}"
    )
    return extended


def main() -> None:
    device = get_device()
    processed = 0
    for model_name in SUPPORTED_MODELS:
        for seed in SEEDS:
            res = process_run(model_name, seed, device)
            if res is not None:
                processed += 1
    print(f"\nProcessed {processed} runs.")


if __name__ == "__main__":
    main()
