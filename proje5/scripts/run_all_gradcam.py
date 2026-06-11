"""Grad-CAM panel: same test samples visualized across all 3 models.

Picks a fixed set of test images (some correctly classified as fractured,
some misclassified) and shows each model's Grad-CAM side-by-side.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHECKPOINTS_DIR, CLASS_NAMES, FIGURES_DIR, LOGS_DIR, SUPPORTED_MODELS, TrainConfig
from src.gradcam import GradCAM, load_and_preprocess, overlay_cam
from src.models import build_model, get_gradcam_target_layer
from src.utils import get_device, load_json


def load_model(model_name: str, seed: int, device: torch.device):
    path = CHECKPOINTS_DIR / f"{model_name}_seed{seed}_best.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, _ = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def pick_samples(model_name: str, seed: int, n_correct: int, n_errors: int) -> pd.DataFrame:
    test_df = pd.read_csv(Path(__file__).resolve().parent.parent / "outputs" / "splits" / "test.csv")
    preds = load_json(LOGS_DIR / f"{model_name}_seed{seed}_test_preds.json")
    test_df = test_df.copy()
    test_df["pred"] = preds["y_pred"]
    test_df["prob"] = preds["y_prob"]

    true_fractured = test_df[test_df["label"] == 1]
    tp = true_fractured[true_fractured["pred"] == 1].sort_values("prob", ascending=False)
    fn = true_fractured[true_fractured["pred"] == 0].sort_values("prob")  # low-conf misses
    fp = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)].sort_values("prob", ascending=False)

    n_fn = (n_errors + 1) // 2
    n_fp = n_errors // 2
    tp_rows = tp.head(n_correct)
    fn_rows = fn.head(n_fn)
    fp_rows = fp.head(n_fp)
    picks = pd.concat([tp_rows, fn_rows, fp_rows])
    picks = picks.reset_index(drop=True)
    picks["kind"] = (
        ["TP"] * len(tp_rows) + ["FN"] * len(fn_rows) + ["FP"] * len(fp_rows)
    )
    return picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pivot-model", default="resnet50", choices=SUPPORTED_MODELS,
                        help="Pick samples based on this model's predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-correct", type=int, default=3)
    parser.add_argument("--num-errors", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    picks = pick_samples(args.pivot_model, args.seed, args.num_correct, args.num_errors)
    image_paths = [Path(p) for p in picks["image_path"].tolist()]
    true_labels = [int(v) for v in picks["label"].tolist()]
    kinds = picks["kind"].tolist()

    # Load all three models once.
    models: dict[str, torch.nn.Module] = {}
    cams: dict[str, GradCAM] = {}
    for model_name in SUPPORTED_MODELS:
        try:
            m = load_model(model_name, args.seed, device)
            models[model_name] = m
            cams[model_name] = GradCAM(m, get_gradcam_target_layer(m, model_name))
        except FileNotFoundError:
            print(f"Checkpoint missing for {model_name}; skipping")

    # For each sample, compute CAM for each model.
    n = len(image_paths)
    n_rows = 1 + len(models)  # original + one row per model
    fig, axes = plt.subplots(n_rows, n, figsize=(3.0 * n, 2.8 * n_rows))
    if n == 1:
        axes = axes[:, None]

    cfg = TrainConfig()
    for j, (p, true_label, kind) in enumerate(zip(image_paths, true_labels, kinds)):
        x, img01 = load_and_preprocess(p, cfg.image_size)
        x_gpu = x.to(device)
        axes[0, j].imshow(img01)
        axes[0, j].set_title(f"{p.name}\n[{kind}] true={CLASS_NAMES[true_label]}", fontsize=8)
        axes[0, j].axis("off")
        for i, (name, m) in enumerate(models.items(), start=1):
            cam_map, pred, prob = cams[name](x_gpu)
            axes[i, j].imshow(overlay_cam(img01, cam_map))
            axes[i, j].set_title(
                f"{name} pred={CLASS_NAMES[pred]} ({prob:.2f})",
                fontsize=8,
            )
            axes[i, j].axis("off")

    fig.suptitle(f"Grad-CAM across models (pivot={args.pivot_model}, seed={args.seed})")
    fig.tight_layout()
    out = FIGURES_DIR / f"gradcam_all_models_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    for c in cams.values():
        c.remove_hooks()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
