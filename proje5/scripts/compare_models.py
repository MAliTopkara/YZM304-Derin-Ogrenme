"""Aggregate per-model test results into a comparison table and plots.

Reads outputs/logs/{run_name}_summary.json and {run_name}_test_preds.json.
Produces:
  outputs/results/comparison.csv, comparison.md
  outputs/figures/confusion_<model>.png
  outputs/figures/roc_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES, FIGURES_DIR, LOGS_DIR, RESULTS_DIR, SUPPORTED_MODELS
from src.evaluate import EvalResult, plot_confusion_matrix, plot_roc_curves
from src.utils import load_json


def load_run(name: str) -> tuple[dict, EvalResult] | None:
    summary_path = LOGS_DIR / f"{name}_summary.json"
    preds_path = LOGS_DIR / f"{name}_test_preds.json"
    if not summary_path.exists() or not preds_path.exists():
        return None
    summary = load_json(summary_path)
    preds = load_json(preds_path)
    m = summary["test_metrics"]
    res = EvalResult(
        accuracy=m["accuracy"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        roc_auc=m["roc_auc"],
        confusion=m["confusion"],
        y_true=preds["y_true"],
        y_pred=preds["y_pred"],
        y_prob=preds["y_prob"],
    )
    return summary, res


def main() -> None:
    rows = []
    eval_results: dict[str, EvalResult] = {}
    for model_name in SUPPORTED_MODELS:
        loaded = load_run(model_name)
        if loaded is None:
            print(f"Skipping {model_name} (no run found)")
            continue
        summary, res = loaded
        eval_results[model_name] = res
        rows.append({
            "model": model_name,
            "accuracy": res.accuracy,
            "precision": res.precision,
            "recall": res.recall,
            "f1": res.f1,
            "roc_auc": res.roc_auc,
            "best_val_f1": summary.get("best_val_f1"),
        })
        plot_confusion_matrix(
            res.confusion, CLASS_NAMES,
            FIGURES_DIR / f"confusion_{model_name}.png",
            title=f"Confusion Matrix — {model_name}",
        )

    if not rows:
        print("No runs found. Train models first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).set_index("model").round(4)
    csv_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(csv_path)
    md_path = RESULTS_DIR / "comparison.md"
    cols = df.columns.tolist()
    lines = ["# Model Comparison (test set)", ""]
    lines.append("| model | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for name, row in df.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_roc_curves(
        eval_results,
        FIGURES_DIR / "roc_comparison.png",
        title="ROC Curves — Test Set",
    )
    print("\n=== Comparison ===")
    print(df.to_string())
    print(f"\nSaved: {csv_path}\n        {md_path}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
