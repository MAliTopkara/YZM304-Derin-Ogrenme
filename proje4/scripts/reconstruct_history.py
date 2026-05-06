"""train_stdout.log dosyasından history.csv ve curves figürünü yeniden üret.

Crash olan eğitimden geriye yalnızca stdout log kalmışsa kullan.

Çalıştırma:
    python scripts/reconstruct_history.py --model efficientnet_b0
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, LOGS_DIR  # noqa: E402
from src.visualize import plot_training_curves  # noqa: E402

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+train_acc=([\d.]+)\s+"
    r"val_loss=([\d.]+)\s+val_acc=([\d.]+)\s+lr=([\d.eE+\-]+)"
)


def parse_log(log_path: Path) -> dict:
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in EPOCH_RE.finditer(text):
        history["train_loss"].append(float(m.group(2)))
        history["train_acc"].append(float(m.group(3)))
        history["val_loss"].append(float(m.group(4)))
        history["val_acc"].append(float(m.group(5)))
        history["lr"].append(float(m.group(6)))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    log_path = LOGS_DIR / f"{args.model}_train_stdout.log"
    if not log_path.exists():
        print(f"[ERROR] Log bulunamadı: {log_path}")
        sys.exit(1)

    history = parse_log(log_path)
    n = len(history["train_loss"])
    if n == 0:
        print("[ERROR] Log içinde epoch satırı bulunamadı")
        sys.exit(1)

    csv_path = LOGS_DIR / f"{args.model}_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])
        for i in range(n):
            w.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_acc'][i]:.6f}",
                f"{history['lr'][i]:.6e}",
            ])

    fig_path = FIGURES_DIR / f"{args.model}_curves.png"
    plot_training_curves(history, fig_path, title=args.model)

    best = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best) + 1
    print(f"Toplam parse edilen epoch: {n}")
    print(f"Best val_acc: {best:.4f} @ epoch {best_epoch}")
    print(f"history CSV: {csv_path}")
    print(f"curves PNG:  {fig_path}")


if __name__ == "__main__":
    main()
