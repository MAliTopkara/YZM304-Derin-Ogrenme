"""Training loop with optional warmup-freeze, early stopping on val F1, mixed precision."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CHECKPOINTS_DIR, LOGS_DIR, TrainConfig
from .evaluate import evaluate
from .models import build_model, param_groups, set_backbone_trainable
from .utils import get_device, get_logger, save_json, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def train(
    cfg: TrainConfig,
    loaders: dict[str, DataLoader],
    run_name: str | None = None,
) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    run_name = run_name or f"{cfg.model_name}_seed{cfg.seed}"
    log_file = LOGS_DIR / f"{run_name}.log"
    logger = get_logger(f"train.{run_name}", log_file)
    logger.info(f"Device: {device} | Model: {cfg.model_name}")

    model, _ = build_model(cfg.model_name)
    model.to(device)

    set_backbone_trainable(model, cfg.model_name, trainable=False)
    groups = param_groups(model, cfg.model_name, cfg.lr_head, cfg.lr_backbone)
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if (cfg.mixed_precision and device.type == "cuda") else None

    best_f1 = -1.0
    best_state = None
    patience = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        if epoch == cfg.warmup_frozen_epochs + 1:
            set_backbone_trainable(model, cfg.model_name, trainable=True)
            logger.info("Unfroze backbone for fine-tuning.")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, loss_fn, device, scaler
        )
        val_result = evaluate(model, loaders["val"], device)
        elapsed = time.time() - t0
        logger.info(
            f"ep {epoch:02d} | loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val f1 {val_result.f1:.4f} auc {val_result.roc_auc:.4f} "
            f"acc {val_result.accuracy:.4f} | {elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_result.accuracy,
            "val_f1": val_result.f1,
            "val_precision": val_result.precision,
            "val_recall": val_result.recall,
            "val_roc_auc": val_result.roc_auc,
            "epoch_seconds": elapsed,
        })

        if val_result.f1 > best_f1:
            best_f1 = val_result.f1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (no val F1 improvement).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = CHECKPOINTS_DIR / f"{run_name}_best.pt"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": cfg.model_name, "state_dict": model.state_dict()}, ckpt_path)

    test_result = evaluate(model, loaders["test"], device)
    logger.info(
        f"TEST | acc {test_result.accuracy:.4f} | prec {test_result.precision:.4f} | "
        f"rec {test_result.recall:.4f} | f1 {test_result.f1:.4f} | auc {test_result.roc_auc:.4f}"
    )

    run_summary = {
        "run_name": run_name,
        "model_name": cfg.model_name,
        "best_val_f1": best_f1,
        "test_metrics": test_result.to_dict(),
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    save_json(run_summary, LOGS_DIR / f"{run_name}_summary.json")
    # Also persist raw test predictions for ROC plots later.
    save_json(
        {"y_true": test_result.y_true, "y_pred": test_result.y_pred, "y_prob": test_result.y_prob},
        LOGS_DIR / f"{run_name}_test_preds.json",
    )
    return run_summary
