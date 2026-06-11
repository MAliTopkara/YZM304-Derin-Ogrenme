"""Statistical comparison of the three models.

McNemar's test: paired predictions at threshold 0.5 on test set, per seed.
DeLong's test: ROC-AUC comparison with variance-of-difference.

Uses seed=42 predictions as the representative single run for pairwise tests,
plus aggregates the per-seed F1/AUC via paired t-tests.

Writes outputs/results/stat_tests.md
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOGS_DIR, RESULTS_DIR, SEEDS, SUPPORTED_MODELS
from src.utils import load_json


# ---- DeLong's test (Sun & Xu, 2014) ----
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m, n = label_1_count, predictions_sorted_transposed.shape[1] - label_1_count
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> tuple[float, float, float]:
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack((y_prob_a, y_prob_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    z = (aucs[0] - aucs[1]) / np.sqrt(l @ cov @ l.T)[0, 0]
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


# ---- Analysis ----
def load_all_preds(model: str) -> dict[int, dict]:
    out = {}
    for seed in SEEDS:
        p = LOGS_DIR / f"{model}_seed{seed}_test_preds.json"
        if p.exists():
            out[seed] = load_json(p)
    return out


def main() -> None:
    all_preds: dict[str, dict[int, dict]] = {m: load_all_preds(m) for m in SUPPORTED_MODELS}

    lines = ["# Statistical Comparison", ""]

    # Per-seed McNemar + DeLong.
    lines.append("## Pairwise tests per seed")
    lines.append("")
    for seed in SEEDS:
        seed_has = [m for m in SUPPORTED_MODELS if seed in all_preds[m]]
        if len(seed_has) < 2:
            continue
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append("| model A | model B | McNemar p | DeLong AUC_A | AUC_B | DeLong p |")
        lines.append("|---|---|---|---|---|---|")
        for a, b in combinations(seed_has, 2):
            pa = all_preds[a][seed]
            pb = all_preds[b][seed]
            y_true = np.array(pa["y_true"])
            assert np.array_equal(y_true, np.array(pb["y_true"])), "test labels mismatch"
            pred_a = np.array(pa["y_pred"])
            pred_b = np.array(pb["y_pred"])
            both_correct = ((pred_a == y_true) & (pred_b == y_true)).sum()
            a_only = ((pred_a == y_true) & (pred_b != y_true)).sum()
            b_only = ((pred_a != y_true) & (pred_b == y_true)).sum()
            both_wrong = ((pred_a != y_true) & (pred_b != y_true)).sum()
            table = [[both_correct, a_only], [b_only, both_wrong]]
            mc = mcnemar(table, exact=False, correction=True)
            auc_a, auc_b, dp = delong_test(y_true, np.array(pa["y_prob"]), np.array(pb["y_prob"]))
            lines.append(
                f"| {a} | {b} | {mc.pvalue:.4f} | {auc_a:.3f} | {auc_b:.3f} | {dp:.4f} |"
            )
        lines.append("")

    # Per-seed F1 paired t-test across models.
    lines.append("## Paired t-test on test F1 across seeds")
    lines.append("")
    lines.append("Uses each seed as a paired observation; F1 computed at threshold 0.5.")
    lines.append("")
    lines.append("| model A | model B | mean F1_A | mean F1_B | t | p |")
    lines.append("|---|---|---|---|---|---|")
    f1_by_model: dict[str, list[float]] = {}
    for m in SUPPORTED_MODELS:
        for seed in SEEDS:
            p = LOGS_DIR / f"{m}_seed{seed}_extended.json"
            if not p.exists():
                continue
            ext = load_json(p)
            f1_by_model.setdefault(m, []).append(ext["test_metrics_t0.5"]["f1"])
    for a, b in combinations(SUPPORTED_MODELS, 2):
        fa, fb = f1_by_model.get(a, []), f1_by_model.get(b, [])
        if len(fa) != len(fb) or len(fa) < 2:
            continue
        t, p = stats.ttest_rel(fa, fb)
        lines.append(
            f"| {a} | {b} | {np.mean(fa):.3f} | {np.mean(fb):.3f} | {t:.3f} | {p:.4f} |"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "stat_tests.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
