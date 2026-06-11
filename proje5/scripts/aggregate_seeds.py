"""Aggregate per-seed extended metrics into mean ± std per model.

Reads outputs/logs/{model}_seed{seed}_extended.json for all seeds.
Writes:
  outputs/results/aggregated.csv   (flat mean/std table)
  outputs/results/aggregated.md    (markdown report-ready)
  outputs/results/aggregated.json  (raw numbers)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOGS_DIR, RESULTS_DIR, SEEDS, SUPPORTED_MODELS
from src.utils import load_json, save_json

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]
OPERATING_POINTS = ["test_metrics_t0.5", "test_metrics_tF1opt", "test_metrics_tYouden"]
OP_LABEL = {
    "test_metrics_t0.5": "t=0.5",
    "test_metrics_tF1opt": "t*(F1)",
    "test_metrics_tYouden": "t*(Youden)",
}


def main() -> None:
    rows = []
    raw: dict = {}
    for model_name in SUPPORTED_MODELS:
        model_raw: dict = {}
        for op in OPERATING_POINTS:
            per_seed = []
            thresholds = []
            for seed in SEEDS:
                path = LOGS_DIR / f"{model_name}_seed{seed}_extended.json"
                if not path.exists():
                    continue
                ext = load_json(path)
                per_seed.append(ext[op])
                thresholds.append(ext[op]["threshold"])
            if not per_seed:
                continue
            model_raw[op] = per_seed
            agg = {"model": model_name, "operating_point": OP_LABEL[op], "n_seeds": len(per_seed),
                   "threshold_mean": float(np.mean(thresholds))}
            for k in METRIC_KEYS:
                vals = np.array([m[k] for m in per_seed])
                agg[f"{k}_mean"] = float(vals.mean())
                agg[f"{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            rows.append(agg)
        raw[model_name] = model_raw

    if not rows:
        print("No extended metrics found. Run postprocess_metrics.py first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "aggregated.csv", index=False)
    save_json(raw, RESULTS_DIR / "aggregated.json")

    # Markdown per operating point — readable for report.
    md_lines = ["# Aggregated Test Metrics (mean \u00b1 std over seeds)", ""]
    for op in OPERATING_POINTS:
        op_label = OP_LABEL[op]
        md_lines.append(f"## Operating point: {op_label}")
        md_lines.append("")
        md_lines.append("| model | n | threshold | accuracy | precision | recall | f1 | macro-F1 | roc-auc |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["operating_point"] != op_label:
                continue
            def fmt(k: str) -> str:
                return f"{r[f'{k}_mean']:.3f} \u00b1 {r[f'{k}_std']:.3f}"
            md_lines.append(
                f"| {r['model']} | {r['n_seeds']} | {r['threshold_mean']:.2f} | "
                f"{fmt('accuracy')} | {fmt('precision')} | {fmt('recall')} | "
                f"{fmt('f1')} | {fmt('f1_macro')} | {fmt('roc_auc')} |"
            )
        md_lines.append("")
    (RESULTS_DIR / "aggregated.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {RESULTS_DIR / 'aggregated.csv'}")
    print(f"Wrote {RESULTS_DIR / 'aggregated.md'}")
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
