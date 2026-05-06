"""Eğitim scripti — sertleştirilmiş sürüm.

Özellikler:
    --amp        : Mixed precision (autocast + GradScaler) — ~%40 hız + daha az VRAM
    --resume     : Son checkpoint'ten devam et (kalan epoch'lar için)
    Incremental  : history.csv her epoch sonu yazılır (crash'te kayıp yok)
    empty_cache  : Her epoch sonu CUDA fragmentation'ı azaltır

Çalıştırma:
    python -m src.train --model resnet50
    python -m src.train --model efficientnet_b0 --amp
    python -m src.train --model vit_base --amp
    python -m src.train --model vit_base --amp --resume         # crash sonrası
    python -m src.train --model resnet50 --epochs 1             # smoke test

Çıktılar:
    results/models/<model>.pth            (en iyi val accuracy ağırlıkları — sadece state_dict)
    results/models/<model>_ckpt.pth       (resume için tam checkpoint)
    results/logs/<model>_history.csv      (epoch başına metrikler — incremental)
    results/figures/<model>_curves.png    (loss/accuracy eğrileri)
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    EARLY_STOPPING_PATIENCE,
    FIGURES_DIR,
    LOGS_DIR,
    MODEL_CONFIGS,
    MODELS_DIR,
    WEIGHT_DECAY,
)
from src.dataset import build_loaders
from src.models import count_parameters, create_model
from src.utils import get_device, model_size_mb, set_seed
from src.visualize import plot_training_curves


@torch.no_grad()
def run_validation(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda")
        if use_amp and device.type == "cuda"
        else _NullCtx()
    )
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    use_cuda_amp = use_amp and device.type == "cuda"
    autocast_ctx_factory = (
        (lambda: torch.amp.autocast(device_type="cuda")) if use_cuda_amp else (lambda: _NullCtx())
    )
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx_factory():
            logits = model(x)
            loss = criterion(logits, y)

        if use_cuda_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def write_history_csv(history: dict, log_path: Path) -> None:
    """history.csv'yi mevcut state ile baştan yaz (incremental)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])
        for i in range(len(history["train_loss"])):
            w.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_acc'][i]:.6f}",
                f"{history['lr'][i]:.6e}",
            ])


def save_checkpoint(ckpt_path: Path, *, epoch: int, model, optimizer, scheduler,
                    scaler, best_val_acc: float, bad_epochs: int, history: dict) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_acc": best_val_acc,
        "bad_epochs": bad_epochs,
        "history": history,
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true",
                    help="Mixed precision training (autocast + GradScaler) — yalnızca CUDA")
    ap.add_argument("--resume", action="store_true",
                    help="results/models/<model>_ckpt.pth varsa kaldığı yerden devam et")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    cfg = MODEL_CONFIGS[args.model]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    pretrained = not args.no_pretrained
    use_amp = args.amp and device.type == "cuda"

    print(f"Model: {args.model} (timm: {cfg['timm_name']})")
    print(f"Epochs: {epochs}, batch_size: {batch_size}, lr: {lr}, pretrained: {pretrained}")
    print(f"AMP: {use_amp}  (flag={args.amp}, device={device.type})")

    train_loader, val_loader, _, classes = build_loaders(batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    print(f"Classes ({len(classes)}): {classes}")

    model = create_model(args.model, pretrained=pretrained).to(device)
    print(f"Trainable params: {count_parameters(model)/1e6:.2f}M  "
          f"size: {model_size_mb(model):.1f} MB")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    bad_epochs = 0
    start_epoch = 1

    weights_path = MODELS_DIR / f"{args.model}.pth"
    ckpt_path = MODELS_DIR / f"{args.model}_ckpt.pth"
    log_path = LOGS_DIR / f"{args.model}_history.csv"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and ckpt_path.exists():
        print(f"[resume] Checkpoint yükleniyor: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        best_val_acc = ckpt["best_val_acc"]
        bad_epochs = ckpt["bad_epochs"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] Epoch {start_epoch}'dan devam — best_val_acc={best_val_acc:.4f}, "
              f"bad_epochs={bad_epochs}")
    elif args.resume:
        print(f"[resume] Checkpoint bulunamadı ({ckpt_path}) — sıfırdan başlanıyor")

    total_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_acc = run_validation(model, val_loader, criterion, device, use_amp=use_amp)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            bad_epochs = 0
            marker = "  [best]"
        else:
            bad_epochs += 1
            marker = f"  (no improve {bad_epochs}/{EARLY_STOPPING_PATIENCE})"

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"lr={cur_lr:.2e}  {dt:.1f}s{marker}", flush=True)

        # incremental persistence — crash sonrası kayıp olmasın
        write_history_csv(history, log_path)
        save_checkpoint(
            ckpt_path,
            epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, best_val_acc=best_val_acc, bad_epochs=bad_epochs, history=history,
        )

        # CUDA fragmentation'ı azalt
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping (patience={EARLY_STOPPING_PATIENCE})")
            break

    total_time = time.time() - total_start

    # final persistence (already incrementally saved, but rewrite for safety)
    write_history_csv(history, log_path)

    fig_path = FIGURES_DIR / f"{args.model}_curves.png"
    plot_training_curves(history, fig_path, title=args.model)

    print()
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {total_time/60:.1f} min")
    print(f"Weights:  {weights_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"History:  {log_path}")
    print(f"Curves:   {fig_path}")
    print()
    print(f"--training-time-seconds: {total_time:.2f}")


if __name__ == "__main__":
    main()
