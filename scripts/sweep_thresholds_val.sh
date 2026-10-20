#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="${2:-../ev-uav-data}"
SUFFIX="${3:-math}"
BACKEND="${4:-math}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/logs"
LOG_FILE="$LOG_DIR/thresh_sweep_val_${SUFFIX}_${STAMP}.log"

mkdir -p "$LOG_DIR"

for experiment in a0 attention_fix a1 r1_event; do
  for seed in 13 37 73; do
    RUN_DIR="$(ls -dt runs/train_${experiment}_s${seed}_fixed50_${SUFFIX}_* 2>/dev/null | head -n 1 || true)"
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" ]; then
      for thresh in 0.7 0.8 0.9 0.95; do
        echo "=== EVAL $experiment seed=$seed thresh=$thresh ckpt=$RUN_DIR $(date -Is) ===" | tee -a "$LOG_FILE"
        uv run python test.py --split val --experiment "$experiment" \
          --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
          --attention_backend "$BACKEND" --prediction_thresh "$thresh" \
          --model_path "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" 2>&1 | tee -a "$LOG_FILE" \
          || echo "=== EVAL FAILED $experiment seed=$seed thresh=$thresh ===" | tee -a "$LOG_FILE"
      done
    else
      echo "=== SKIP $experiment seed=$seed (missing $RUN_DIR) ===" | tee -a "$LOG_FILE"
    fi
  done
done

echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG_FILE"
