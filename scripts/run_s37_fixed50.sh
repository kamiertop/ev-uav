#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="${2:-../ev-uav-data}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/logs"
LOG_FILE="$LOG_DIR/four_runs_s37_fixed50_auto_${STAMP}.log"
GPU_CSV="$LOG_DIR/gpu${GPU}_memory_${STAMP}.csv"

mkdir -p "$LOG_DIR"

uv sync
nvidia-smi | tee -a "$LOG_FILE"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU" >> "$GPU_CSV"
    sleep 30
  done
) &
GPU_SAMPLER_PID=$!
trap 'kill $GPU_SAMPLER_PID 2>/dev/null || true' EXIT

for experiment in a1 r1_event r2_balanced r3_term; do
  echo "=== START $experiment $(date -Is) ===" | tee -a "$LOG_FILE"
  uv run python train.py --experiment "$experiment" \
    --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
    --run_name "${experiment}_s37_fixed50_auto" \
    --seed 37 --epochs 50 --batch_size 1 --max_events_num 700000 \
    --optim Adam --lr 0.01 --final_lr 0.001 --scheduler linear \
    --attention_backend auto --risk_weight 0.1 --risk_beta 1 --risk_bin_ms 50 \
    --early_stopping_patience 0 --no-test_after_train 2>&1 | tee -a "$LOG_FILE"
  echo "=== END $experiment $(date -Is) ===" | tee -a "$LOG_FILE"
done

uv run python scripts/summarize_runs.py | tee -a "$LOG_FILE"
echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG_FILE"
