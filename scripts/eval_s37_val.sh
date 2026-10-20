#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="${2:-../ev-uav-data}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/logs"
LOG_FILE="$LOG_DIR/val_eval_s37_${STAMP}.log"

mkdir -p "$LOG_DIR"

eval_run() {
  local experiment="$1" thresh="$2" run_dir="$3"
  echo "=== EVAL $experiment thresh=$thresh ckpt=$run_dir $(date -Is) ===" | tee -a "$LOG_FILE"
  uv run python test.py --split val --experiment "$experiment" \
    --data_dir "$DATA_DIR" --gpu "$GPU" --train_workers 0 \
    --attention_backend auto --prediction_thresh "$thresh" \
    --model_path "$run_dir/checkpoints/best_iou_seed37.pt" 2>&1 | tee -a "$LOG_FILE"
}

for experiment in a1 r1_event r2_balanced r3_term; do
  RUN_DIR="$(ls -dt runs/train_${experiment}_s37_fixed50_auto_* | head -n 1)"
  eval_run "$experiment" 0.9 "$RUN_DIR"
done

for experiment in a1 r1_event; do
  RUN_DIR="$(ls -dt runs/train_${experiment}_s37_fixed50_auto_* | head -n 1)"
  eval_run "$experiment" 0.7 "$RUN_DIR"
  eval_run "$experiment" 0.95 "$RUN_DIR"
done

echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG_FILE"
