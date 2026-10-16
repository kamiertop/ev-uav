# 服务器训练命令

更新日期：2026-10-16

以下命令均在项目根目录执行。数据默认读取 `/data/ev-uav`。

研究协议：探索、消融和多种子训练默认关闭自动测试。当前首轮筛选沿用已有 A1 的早停协议：最多 50 轮、patience=10、min_delta=1e-4。固定 50 轮也是可选协议，但所有对照必须一致。

## 0. 上传后先检查

```bash
git status --short
uv sync
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python -m unittest discover -s tests -v
nvidia-smi
```

如果服务器还没有编译 `HAIS_OP.so`，按项目原有方式执行：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python scripts/build_ext.py
```

## 1. 一轮 CUDA 烟雾训练

先验证完整方法能够前向、反向、验证并保存 checkpoint。建议先用 4090 的空闲 GPU 编号替换 `0`：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav \
  --run_name smoke_a4 --epochs 1 --train_workers 0 \
  --patch_attention sequence --motion_gd shallow --loss mbtc \
  --no-test_after_train
```

确认训练和验证完成、损失没有 NaN、生成 checkpoint；保留完整日志和 `config.json`，记录显存峰值与单轮耗时。可检查最近生成的文件：

```bash
find runs -maxdepth 3 -type f | sort | tail -n 20
```

## 2. 第一阶段五组单种子消融

建议先跑 A0、A1、A4，再补齐 A2、A3。所有配置保持相同数据划分、batch size、事件采样上限、优化器和训练轮数。A1 是判断新增方法贡献的关键对照。

可在 3090 和 4090 上并行运行不同配置，但同一配置的计时比较必须使用相同 GPU。

### A0：严格公开行为

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a0_official_s37 \
  --seed 37 --epochs 50 --patch_attention legacy --motion_gd none --loss stc \
  --no-test_after_train --early_stopping_patience 10
```

### A1：修正并增强 Patch Attention

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a1_attention_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd none --loss stc \
  --no-test_after_train --early_stopping_patience 10
```

### A2：仅 MBTC 损失

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a2_mbtc_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd none --loss mbtc \
  --no-test_after_train --early_stopping_patience 10
```

### A3：仅 MC-GDSC 结构

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a3_mcgds_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd shallow --loss stc \
  --no-test_after_train --early_stopping_patience 10
```

### A4：完整方法

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a4_full_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd shallow --loss mbtc \
  --no-test_after_train --early_stopping_patience 10
```

上述命令仅使用 train/val 进行模型选择。默认不自动测试，早停 patience 为 10；固定轮数时使用 `--early_stopping_patience 0`，不能与早停结果混用。

## 3. 找到每组最佳验证结果

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python scripts/summarize_runs.py
```

脚本会选择每个运行的最高验证 IoU，而不是最后一个 epoch；同时列出轨迹指标。

## 4. 最终配置运行三个种子

如果 A4 优于 A1/A2/A3，再补齐三个种子。以下示例只补跑 13 和 73，已有 seed 37 结果仅在配置与训练协议完全一致时复用，否则也需重跑。若最终选择 A2 或 A3，使用对应训练配置替换示例：

```bash
for seed in 13 73; do
  UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
    --gpu 0 --data_dir /data/ev-uav --run_name "a4_full_s${seed}" \
    --seed "$seed" --epochs 50 \
    --patch_attention sequence --motion_gd shallow --loss mbtc \
    --no-test_after_train --early_stopping_patience 10
done
```

基线 A1 也应运行相同三个种子，才能报告均值和标准差：

```bash
for seed in 13 73; do
  UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
    --gpu 0 --data_dir /data/ev-uav --run_name "a1_attention_s${seed}" \
    --seed "$seed" --epochs 50 \
    --patch_attention sequence --motion_gd none --loss stc \
    --no-test_after_train --early_stopping_patience 10
done
```

## 5. 最终测试集评估

先冻结最终配置、模型选择规则和验证集选出的阈值，不再依据测试结果调参。对最终方案和关键基线的各个预定种子分别评估，报告均值与标准差。

从对应运行目录读取 `config.json`，确保评测参数与训练完全一致。下面以完整方法为例：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python test.py \
  --gpu 0 --data_dir /data/ev-uav \
  --model_path runs/train_a4_full_s37_YYYYMMDD_HHMMSS/checkpoints/best_iou_seed37.pt \
  --patch_attention sequence --motion_gd shallow --loss mbtc
```

会输出 IoU、ACC、Pd、Fa、trajectory coverage、longest ratio 和 fragmentation。

如果评估旧的官方 checkpoint，必须使用其对应结构：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python test.py \
  --gpu 0 --data_dir /data/ev-uav \
  --model_path /path/to/official_checkpoint.pt \
  --patch_attention legacy --motion_gd none --loss stc
```

## 6. 常见故障

- `No CUDA GPUs are available`：容器未挂载 GPU，检查 `nvidia-smi` 和启动参数。
- `HAIS_OP` 导入失败：重新运行 `scripts/build_ext.py`，确认 CUDA/GCC 与服务器环境匹配。
- DataLoader worker 崩溃：先加 `--train_workers 0`；稳定后再恢复 4 或 8。
- OOM：项目默认 batch size 为 1；仍 OOM 时降低 `--max_events_num`，但所有对照必须使用相同值。
- deterministic/spconv 报错：保留完整错误日志，不要直接关闭确定性设置后与旧结果混用。

## 7. 网格搜索（笛卡尔积）

使用 `scripts/grid_search.py` 可以自动遍历参数组合，实验按顺序运行，单组失败默认不会阻塞后续组合。每次搜索会在 `runs/grid_searches/` 写入 `manifest.jsonl`。

先预览组合数量和命令：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python scripts/grid_search.py \
  --dry_run --gpu 0 --epochs 50 --seeds 37 \
  --losses stc,mbtc --motion_gds none,shallow
```

网格搜索现在默认关闭自动测试，也支持显式 `--no-test_after_train` 和 `--test_after_train`。用 `--patience 10` 保持当前早停协议，或所有对照统一使用 `--patience 0` 固定训练轮数。`--dry_run` 仅生成命令与清单，不训练。

例如预览原 Attention 上的损失对照：

```bash
uv run python scripts/grid_search.py --dry_run \
  --patch_attentions legacy --motion_gds none --losses stc,mbtc \
  --seeds 37 --patience 10 --no-test_after_train
```

不要扫描与当前损失无关的权重，以免生成重复实验。

## 8. 统一实验预设（显式参数优先）

`--experiment` 只设置 Attention、卷积和损失，不改变训练预算或数据路径。显式指定的参数始终覆盖预设，与参数顺序无关。

| 预设 | Attention | motion_gd | loss |
|---|---|---|---|
| a0 | legacy | none | stc |
| a0_mbtc | legacy | none | mbtc |
| attention_fix | corrected | none | stc |
| a1 | sequence | none | stc |
| a2 | sequence | none | mbtc |
| a3 | sequence | shallow | stc |
| a4 | sequence | shallow | mbtc |

`legacy` 保留原行为；`corrected` 仅修正序列组织并隔离样本，保留单头、原 padding，不增加位置编码、LayerNorm 或内部残差；`sequence` 保持现有 A1 的完整增强行为。

```bash
uv run python train.py --experiment a0_mbtc \
  --data_dir ../ev-uav-data --gpu 0 --train_workers 0 \
  --run_name a0_mbtc_s37 --seed 37 --epochs 50 \
  --early_stopping_patience 10 --early_stopping_min_delta 1e-4 \
  --no-test_after_train
```

损失分项用 `--support_weight`、`--trajectory_weight`、`--motion_direction_weight`、`--motion_speed_weight`、`--route_weight` 控制，置 0 可关闭对应项。动态卷积位置用 `--motion_gd none/shallow/encoder/all`，时间尺度用 `--motion_temporal_dilations 1 2 4 8`。只有实际使用该模块或损失的配置，这些参数才有作用。

评估必须使用训练时相同的预设及覆盖参数；特别是 legacy 与 corrected 的权重形状相同，严格加载成功不能证明模式一致。以保存的 `config.json` 为准。布尔选项支持 `--vis` / `--no-vis`，兼容旧式 `--vis False`。
