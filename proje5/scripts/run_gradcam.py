"""Generate Grad-CAM visualizations for a trained model on selected test images.

Picks N true-positive (correctly predicted fractured) and N misclassified samples
and saves a panel with original + heatmap overlay.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    FIGURES_DIR,
    LOGS_DIR,
    SUPPORTED_MODELS,
    TrainConfig,
)
from src.gradcam import GradCAM, load_and_preprocess, save_gradcam_panel
from src.models import build_model, get_gradcam_target_layer
from src.utils import get_device, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--num-correct", type=int, default=3)
    parser.add_argument("--num-errors", type=int, default=3)
    args = parser.parse_args()

    cfg = TrainConfig(model_name=args.model)
    device = get_device()

    ckpt = torch.load(CHECKPOINTS_DIR / f"{args.model}_best.pt", map_location=device, weights_only=False)
    model, _ = build_model(args.model)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    target_layer = get_gradcam_target_layer(model, args.model)
    cam = GradCAM(model, target_layer)

    preds = load_json(LOGS_DIR / f"{args.model}_test_preds.json")
    import pandas as pd
    test_df = pd.read_csv(Path(__file__).resolve().parent.parent / "outputs" / "splits" / "test.csv")
    test_df["pred"] = preds["y_pred"]
    test_df["prob"] = preds["y_prob"]

    # Focus on positive class (fractured) — informative for XAI.
    true_fractured = test_df[test_df["label"] == 1]
    tp = true_fractured[true_fractured["pred"] == 1].sort_values("prob", ascending=False)
    fn = true_fractured[true_fractured["pred"] == 0].sort_values("prob")  # wrongly low prob
    fp = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)].sort_values("prob", ascending=False)

    picks_correct = tp.head(args.num_correct)
    picks_error = pd.concat([fn.head(args.num_errors // 2 + args.num_errors % 2), fp.head(args.num_errors // 2)])

    for tag, picks in [("correct", picks_correct), ("errors", picks_error)]:
        if len(picks) == 0:
            continue
        paths = [Path(p) for p in picks["image_path"].tolist()]
        imgs, cams_list, pred_labels, pred_probs, true_labels = [], [], [], [], []
        for p, true_label in zip(paths, picks["label"].tolist()):
            x, img01 = load_and_preprocess(p, cfg.image_size)
            x = x.to(device)
            cam_map, pred, prob = cam(x)
            imgs.append(img01)
            cams_list.append(cam_map)
            pred_labels.append(pred)
            pred_probs.append(prob)
            true_labels.append(int(true_label))
        save_gradcam_panel(
            paths, imgs, cams_list, pred_labels, pred_probs, true_labels,
            FIGURES_DIR / f"gradcam_{args.model}_{tag}.png",
            title=f"Grad-CAM — {args.model} ({tag})",
        )
        print(f"Saved gradcam_{args.model}_{tag}.png ({len(picks)} images)")

    cam.remove_hooks()


if __name__ == "__main__":
    main()
