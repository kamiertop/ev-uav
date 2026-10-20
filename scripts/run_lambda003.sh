#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="${2:-../ev-uav-data}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/logs"
LOG_FILE="$LOG_DIR/lambda003_math_${STAMP}.log"

mkdir -p "$LOG_DIR"

RUN_DIR=""

for seed in 13 37 73; do
  echo "=== START r1_event lambda=0.03 seed=$seed $(date -Is) ===" | tee -a "$LOG_FILE"
  uv run python train.py --experiment r1_event \
    --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
    --run_name "r1l003_s${seed}_fixed50_math" \
    --seed "$seed" --epochs 50 --batch_size 1 --max_events_num 700000 \
    --optim Adam --lr 0.01 --final_lr 0.001 --scheduler linear \
    --attention_backend math --risk_weight 0.03 --risk_beta 1 --risk_bin_ms 50 \
    --early_stopping_patience 0 --no-test_after_train 2>&1 | tee -a "$LOG_FILE"
  RUN_DIR="$(ls -dt runs/train_r1l003_s${seed}_fixed50_math_* 2>/dev/null | head -n 1 || true)"
  if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" ]; then
    for thresh in 0.85 0.9 0.95; do
      echo "=== VAL r1l003 seed=$seed thresh=$thresh $(date -Is) ===" | tee -a "$LOG_FILE"
      uv run python test.py --split val --experiment r1_event \
        --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
        --attention_backend math --prediction_thresh "$thresh" \
        --model_path "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" 2>&1 | tee -a "$LOG_FILE" \
        || echo "=== VAL FAILED r1l003 seed=$seed thresh=$thresh ===" | tee -a "$LOG_FILE"
    done
  else
    echo "=== VAL SKIPPED r1l003 seed=$seed (missing $RUN_DIR) ===" | tee -a "$LOG_FILE"
  fi
  echo "=== END seed=$seed $(date -Is) ===" | tee -a "$LOG_FILE"
done

echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG_FILE"
