# 训练与评估命令

更新：2026-11-10。训练由用户在独立GPU服务器运行，本地只维护代码和检查。

## 安装

```bash
uv sync --locked
uv run python scripts/build_ext.py
```

Linux/NVIDIA、CUDA12.6和GCC11–13；编译器与驱动需在服务器重新核验，`HAIS_OP.so`由目标环境构建。数据根目录示例`../ev-uav-data`，包含train/val/test。`uv sync`会同步删除不再需要的依赖，不改变本轮保留依赖的锁定版本。

## 主方案训练

```bash
uv run python train.py --experiment r1_precision --data_dir ../ev-uav-data --gpu 0 \
  --run_name r1_precision_s37_fixed50_math --seed 37 --epochs 50 \
  --train_workers 0 --batch_size 1 --max_events_num 700000 --attention_backend math \
  --optim Adam --lr 0.01 --final_lr 0.001 --scheduler linear \
  --prediction_thresh 0.9 --val_start_epoch 0 --val_interval 1 \
  --early_stopping_patience 0 --no-test_after_train \
  --risk_weight 0.1 --iou_weight 0.1 --hardneg_weight 0.03 \
  --hardneg_ratio 3 --hardneg_min 256
```

此为从头训练，不保证重现旧seed37数值。每轮用val@.9选择best_iou；最佳已知权重最终test阈值为.85。训练默认预设仍为A1+STC，主方案必须显式`--experiment r1_precision`。固定50轮必须显式关闭早停与自动test；不把epochs=1当正式训练计划。

## 四组消融

```bash
bash scripts/run_r1_precision.sh --gpu 0 --data-dir ../ev-uav-data
```

默认seed37，依次r1_event/r1_jaccard/r1_hardneg/r1_precision；候选先单轮smoke，正式各50轮math、val@.9选权重，再以同一权重评val阈值.8/.85/.9/.95/.97/.98。任何步骤不访问test。保存代码快照/哈希、数据清单/val内容哈希、环境、配置、指标与日志，结果包为`runs/logs/r1_ablation_*.tar.gz`，不含权重；GPU启动前至少22GiB空闲。

只跑某组或补种子（仍是从头训练，不是断点续训）：

```bash
bash scripts/run_r1_precision.sh --gpu 0 --data-dir ../ev-uav-data \
  --experiments r1_precision --seeds 13 73
```

也支持a0/attention_fix/a1/a1_jaccard基线；完整预览：

```bash
bash scripts/run_r1_precision.sh --dry-run --experiments a0 attention_fix a1 a1_jaccard \
  r1_event r1_jaccard r1_hardneg r1_precision --seeds 37
```

## 独立验证或测试

```bash
uv run python test.py --split val --experiment r1_precision --data_dir ../ev-uav-data \
  --gpu 0 --seed 37 --train_workers 0 --batch_size 1 --attention_backend math \
  --model_path runs/实际目录/checkpoints/best_iou_seed37.pt \
  --prediction_thresh 0.85 --output_json runs/logs/my_val.json
```

配置冻结后，将`--split val`改为`--split test`，另存my_test.json。不在test扫阈值或重选epoch。通用test.py不自动恢复训练配置，非默认模型结构/评估参数必须从实际config.json显式传入。模型严格加载，不用strict=False绕过错误权重。

## 重放两份已验证旧最佳权重

```bash
bash scripts/test_r1_best.sh --gpu 0 --data-dir ../ev-uav-data
```

精确绑定10月21日Jaccard_104716及组合_115309，seed37/math/test@.85，先核对两组配置/完成记录/权重SHA256，保存冻结计划、CSV、逐序列JSON与日志。输出`runs/logs/r1_best_test_*.tar.gz`。这两组已于11月10日完成终评，日常开发无需反复重放。

服务器需要原始runs训练目录；Git不包含权重和日志。若缺失，从本地`runs/preserved/r1_best_test_seed37_20261110.tar.gz`上传并在项目根`tar -xzf`恢复；保留目录布局，也可用`--runs-root`指定恢复位置。`--dry-run`不访问数据或核对权重。缺权重时不要用新训练或SAM权重代替。

## 清理后的接口

移除SAM、Lovász、R2/R3预设及`--sam_rho/--sam_start_epoch/--iou_mode/--risk_mode/--risk_beta/--risk_bin_ms`，也移除没有实现行为的`--vis/--save`开关。旧配置仍作为原始证据保留，但不能把所有旧字段直接拼回CLI。`--trajectory_bin_ms`只控制评估轨迹指标。

旧run_iou_ablation/run_all_math/run_lambda003/final_test等批次脚本已删除；统一使用run_ablation.py及上述wrapper。Git迁移要同步删除项、新文件和uv.lock，不只上传diff。保留原runs、数据和HAIS扩展。

## 本地检查

```bash
.venv/bin/python -m unittest discover -s tests
```

本地CPU测试不替代GPU训练、推理或精度复验。当前清理后的检查和限制见NEXT_STEPS.md。
