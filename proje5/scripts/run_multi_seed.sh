#!/usr/bin/env bash
# Run seeds 123 and 2024 for all 3 models sequentially.
set -e
cd "$(dirname "$0")/.."
for SEED in 123 2024; do
  for MODEL in resnet50 densenet121 efficientnet_b0; do
    echo "=== $(date +%H:%M:%S) START ${MODEL} seed=${SEED} ==="
    python scripts/train_model.py --model "${MODEL}" --seed "${SEED}" --num-workers 0
    echo "=== $(date +%H:%M:%S) DONE ${MODEL} seed=${SEED} ==="
  done
done
echo "ALL MULTI-SEED RUNS COMPLETE"
