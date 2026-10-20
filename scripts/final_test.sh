#!/usr/bin/env bash
set -euo pipefail

# usage: ./scripts/final_test.sh <experiment:threshold> [<experiment:threshold> ...]
# e.g.: ./scripts/final_test.sh a1:0.8 r1_event:0.95 a0:0.8

GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-../ev-uav-data}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/logs"
LOG_FILE="$LOG_DIR/final_test_math_${STAMP}.log"

mkdir -p "$LOG_DIR"

for pair in "$@"; do
  experiment="${pair%%:*}"
  thresh="${pair##*:}"
  for seed in 13 37 73; do
    RUN_DIR="$(ls -dt runs/train_${experiment}_s${seed}_fixed50_math_* 2>/dev/null | head -n 1 || true)"
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" ]; then
      echo "=== TEST $experiment seed=$seed thresh=$thresh ckpt=$RUN_DIR $(date -Is) ===" | tee -a "$LOG_FILE"
      uv run python test.py --split test --experiment "$experiment" \
        --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
        --attention_backend math --prediction_thresh "$thresh" \
        --model_path "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" 2>&1 | tee -a "$LOG_FILE" \
        || echo "=== TEST FAILED $experiment seed=$seed ===" | tee -a "$LOG_FILE"
    else
      echo "=== SKIP $experiment seed=$seed (missing $RUN_DIR) ===" | tee -a "$LOG_FILE"
    fi
  done
done

echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG_FILE"
