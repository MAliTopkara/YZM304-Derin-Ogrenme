"""CLI: train a single model.

Usage:
    python scripts/train_model.py --model resnet50 --epochs 20
    python scripts/train_model.py --model resnet50 --epochs 2 --run-name sanity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SUPPORTED_MODELS, TrainConfig
from src.data import build_dataloaders
from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(model_name=args.model, seed=args.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr_head is not None:
        cfg.lr_head = args.lr_head
    if args.lr_backbone is not None:
        cfg.lr_backbone = args.lr_backbone
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_weighted_sampler:
        cfg.use_weighted_sampler = False

    loaders = build_dataloaders(cfg)
    summary = train(cfg, loaders, run_name=args.run_name)
    print("\n=== TEST METRICS ===")
    for k, v in summary["test_metrics"].items():
        if k != "confusion":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
