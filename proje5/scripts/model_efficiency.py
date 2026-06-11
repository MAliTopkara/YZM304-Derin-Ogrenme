"""Report per-model efficiency: parameter count, trainable params, avg epoch time.

Reads epoch times from training history JSONs, averaged across seeds.
Writes:
  outputs/results/efficiency.csv
  outputs/results/efficiency.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOGS_DIR, RESULTS_DIR, SEEDS, SUPPORTED_MODELS
from src.models import build_model
from src.utils import load_json


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    rows = []
    for model_name in SUPPORTED_MODELS:
        model, _ = build_model(model_name)
        total, trainable = count_params(model)
        epoch_times, num_epochs = [], []
        for seed in SEEDS:
            path = LOGS_DIR / f"{model_name}_seed{seed}_summary.json"
            if not path.exists():
                continue
            hist = load_json(path)["history"]
            epoch_times.extend(h["epoch_seconds"] for h in hist)
            num_epochs.append(len(hist))
        rows.append({
            "model": model_name,
            "total_params_M": total / 1e6,
            "trainable_params_M": trainable / 1e6,
            "avg_epoch_sec": float(np.mean(epoch_times)) if epoch_times else float("nan"),
            "avg_epochs_trained": float(np.mean(num_epochs)) if num_epochs else float("nan"),
            "avg_total_train_time_sec": (
                float(np.mean(epoch_times) * np.mean(num_epochs))
                if epoch_times and num_epochs else float("nan")
            ),
            "n_runs": len(num_epochs),
        })
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(RESULTS_DIR / "efficiency.csv", index=False)

    md_lines = ["# Model Efficiency", "",
                "| model | params (M) | trainable (M) | avg epoch (s) | avg epochs | total train (s) | runs |",
                "|---|---|---|---|---|---|---|"]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['total_params_M']:.2f} | {r['trainable_params_M']:.2f} | "
            f"{r['avg_epoch_sec']:.1f} | {r['avg_epochs_trained']:.1f} | "
            f"{r['avg_total_train_time_sec']:.0f} | {r['n_runs']} |"
        )
    (RESULTS_DIR / "efficiency.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
