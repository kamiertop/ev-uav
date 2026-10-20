#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="${2:-../ev-uav-data}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/logs"
LOG_FILE="$LOG_DIR/baseline_seeds_${STAMP}.log"
GPU_CSV="$LOG_DIR/gpu${GPU}_memory_${STAMP}.csv"

mkdir -p "$LOG_DIR"

nvidia-smi | tee -a "$LOG_FILE"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU" >> "$GPU_CSV"
    sleep 30
  done
) &
GPU_SAMPLER_PID=$!
trap 'kill $GPU_SAMPLER_PID 2>/dev/null || true' EXIT

RUN_DIR=""

for seed in 13 37 73; do
  for experiment in a0 attention_fix; do
    echo "=== START $experiment seed=$seed $(date -Is) ===" | tee -a "$LOG_FILE"
    uv run python train.py --experiment "$experiment" \
      --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
      --run_name "${experiment}_s${seed}_fixed50_auto" \
      --seed "$seed" --epochs 50 --batch_size 1 --max_events_num 700000 \
      --optim Adam --lr 0.01 --final_lr 0.001 --scheduler linear \
      --attention_backend auto --risk_weight 0.1 --risk_beta 1 --risk_bin_ms 50 \
      --early_stopping_patience 0 --no-test_after_train 2>&1 | tee -a "$LOG_FILE"
    RUN_DIR="$(ls -dt runs/train_${experiment}_s${seed}_fixed50_auto_* 2>/dev/null | head -n 1 || true)"
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" ]; then
      echo "=== VAL $experiment seed=$seed $RUN_DIR $(date -Is) ===" | tee -a "$LOG_FILE"
      uv run python test.py --split val --experiment "$experiment" \
        --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
        --attention_backend auto --prediction_thresh 0.9 \
        --model_path "$RUN_DIR/checkpoints/best_iou_seed${seed}.pt" 2>&1 | tee -a "$LOG_FILE" \
        || echo "=== VAL FAILED $experiment seed=$seed ===" | tee -a "$LOG_FILE"
    else
      echo "=== VAL SKIPPED $experiment seed=$seed (missing $RUN_DIR) ===" | tee -a "$LOG_FILE"
    fi
    echo "=== END $experiment seed=$seed $(date -Is) ===" | tee -a "$LOG_FILE"
  done
done

uv run python scripts/summarize_runs.py | tee -a "$LOG_FILE"
echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG_FILE"
