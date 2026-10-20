#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

uv run python train.py --experiment a0 \
  --data_dir ../ev-uav-data --gpu "$GPU" --train_workers 0 \
  --run_name a0_s37_fixed50_math_diag --seed 37 --epochs 50 \
  --batch_size 1 --max_events_num 700000 \
  --optim Adam --lr 0.01 --final_lr 0.001 --scheduler linear \
  --attention_backend math --diagnose_numerics \
  --early_stopping_patience 0 --no-test_after_train 2>&1 | tee -a runs/logs/a0_s37_math_diag.log
