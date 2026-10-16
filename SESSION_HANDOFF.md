# EV-UAV 项目会话交接文档

更新时间：2026-10-16

本文档用于下一个 agent 接续当前项目。请先阅读本文档，再查看
`RESEARCH_PLAN.md` 和 `TRAINING_COMMANDS.md`，以 `git status --short` 确认当前工作区状态，不要重置或覆盖已有修改。

## 1. 用户目标

用户希望基于本项目中的两篇论文和第一篇论文源码，提出可发表的创新改动。目标以毕业投稿为主，投稿层级可为中文核心、SCI 二区/三区、CCF B/C。用户允许自由选择损失函数、模型结构、模块、训练范式和评价指标；只要相对原论文有一定指标提升，或轨迹完整性明显改善即可。

数据已经完整下载到 `/data/ev-uav`，服务器有 RTX 3090 和 RTX 4090。用户要求：由 agent 编写代码、记录研究思路并给出训练命令，用户自行上传代码并在服务器训练。

## 2. 论文材料与研究定位

本地论文：

1. `Event_based_Tiny_object_Detection_A_Benchmark_Dataset_and_Baseline.pdf`
   - ICCV 2025 EV-UAV/EV-SpSegNet。
   - 论文报告指标：IoU 55.18、ACC 65.02、Pd 77.53、Fa 1.63e-4。
   - 主要组件：GDSCA/GDSC、Sp-SE、Patch Attention、STC Loss。
   - 原始 GDSC 使用固定多尺度膨胀率 `(1,2,3,4)` 和固定通道分组。

2. `Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection.pdf`
   - CVPR 2026 同团队 SpTopoNet。
   - 论文报告指标：IoU 66.62、ACC 74.43/74.44、Pd 83.36/83.37、Fa 1.29e-4。
   - 组件：TLM、SCM、EvTopoLoss、持久同调拓扑约束。
   - 当前方案有意避开简单复用该论文的拓扑损失/拓扑模块。

暂定论文题目：

> 面向事件微小目标检测的运动条件化稀疏卷积与双向轨迹一致性学习

英文：

> Motion-Conditioned Sparse Convolution with Bidirectional Trajectory Consistency for Event-Based Tiny Object Detection

核心研究问题：固定时空邻域难以同时适应慢速、快速、稀疏和转向轨迹；能否利用 EV-UAV 已有实例和时间标签，使稀疏卷积尺度选择与目标运动状态耦合，同时抑制轨迹缺口和连续背景误检？

## 3. 已实现的创新代码

### 3.1 Motion-Conditioned GDSC（MC-GDSC）

位置：`model/basemodel.py`、`model/evspsegnet.py`。

- 保留原四个分组卷积分支，减少与原模型的不可控差异。
- 每个活动体素预测局部速度和四分支路由权重。
- 路由形式为速度条件化的 softmax；路由最后一层零初始化，使初始 `4 * softmax` 约等于 1，尽量贴近固定 GDSC。
- 默认空间 dilation 为 `(1,2,3,4)`，时间 dilation 为 `(1,2,4,8)`。
- 默认只在浅层 `encoder1` 使用动态 GDSC（`motion_gd=shallow`），因为浅层保留更完整的轨迹几何且计算开销较低。
- `motion_gd=encoder/all` 仅作为后续消融配置。
- 速度输出默认范围 `motion_speed_scale=1.5` 像素/ms。

### 3.2 Motion-Bidirectional Trajectory Consistency（MBTC）Loss

位置：`utils/mbtcloss.py`。

动机：原 STC 对局部事件支持低的正样本赋较小权重，可能忽略轨迹缺口；对局部支持高的负样本赋较小权重，可能忽略连续背景假阳性。

MBTC 包含：

- 困难结构权重：低 GT 支持的正样本和高预测支持的负样本提高权重；
- 类平衡 focal segmentation loss；
- 同一实例相邻 50 ms 时间箱的置信度连续性约束；
- 由实例中心轨迹差分生成像素/ms 速度，约束方向和速度；
- 对动态路由加入弱 batch-average balance regularization，避免所有事件塌缩到单个分支。

旧的 `utils/tacloss.py` 保留作为历史原型/可选消融，不是当前主方案。

### 3.3 轨迹评价指标

位置：`utils/eval.py`。

- `trajectory_coverage`：成功检测的轨迹时间箱比例，越高越好；
- `trajectory_longest_ratio`：最长连续成功段占整条轨迹的比例，越高越好；
- `trajectory_fragmentation`：检测片段数/轨迹时间箱数，越低越好。

默认时间 bin 为 50 ms。阈值必须在验证集选择并冻结，不能依据测试集继续调参。

## 4. 数据审计结论

- `/data/ev-uav`：train 99 个 `.npz`，val 24 个，test 24 个；
- 训练集约 932 万事件，前景比例约 2.94%；
- 99 个训练序列中约 56 个为多目标，单序列最多约 27 个实例；
- `evs_norm` shape `(N,6)`：前 4 列输入特征，第 5 列 segmentation label，第 6 列 instance id；
- `ev_loc` shape `(N,3)`：x/y/t，t 按毫秒整数坐标使用；
- 运动标签已经从单位方向改为由中心轨迹差分得到的像素/ms 速度；大多数速度约 0.01–0.15 像素/ms，极端约 1.26。

## 5. 当前代码改动清单

已修改或新增：

- `dataset/basedataset.py`：collate motion target/valid；
- `dataset/ev_uav.py`：缓存数据、排序 npz、生成轨迹速度 target 和 valid mask；
- `model/basemodel.py`：`MotionConditionedGDConv`，GDBlock fixed/motion 模式；
- `model/evspsegnet.py`：`motion_gd=none/shallow/encoder/all`、各向异性 temporal dilation、sequence/legacy/corrected Patch Attention、auxiliary motion/routing 输出、loss 分支；
- `utils/mbtcloss.py`：新 MBTC；
- `utils/tacloss.py`：历史 TACL 原型；
- `utils/args.py`：默认 data path、run name、Patch Attention、motion、loss、MBTC、轨迹指标等参数；并修复 `args.parse()` 设置 module-level `utils.args.cfg`；
- `utils/eval.py`：严格 checkpoint architecture loading、轨迹指标、DataLoader workers/pin memory；
- `train.py`：支持 `stc/tacl/mbtc`、run prefix、支持训练后自动 test、验证记录轨迹指标；
- `test.py`：输出 IoU/ACC/Pd/Fa/轨迹指标；
- `scripts/summarize_runs.py`：汇总每个 run 最佳验证 IoU 和轨迹指标；
- `tests/test_trajectory_components.py`、`tests/test_eval_metrics.py`：单元测试；
- `RESEARCH_PLAN.md`：完整研究计划；
- `TRAINING_COMMANDS.md`：服务器训练命令。

当前代码默认行为：`train.py` 已加入基于验证 IoU 的 early stopping；`utils/args.py` 中 `test_after_train` 默认关闭。默认连续 10 个验证轮次无至少 `1e-4` 提升即停止，仅在显式传入 `--test_after_train` 时测试最佳 IoU checkpoint。使用 `--no-test_after_train` 可关闭自动测试，使用 `--early_stopping_patience 0` 可禁用早停。探索、消融和多种子训练必须显式添加 `--no-test_after_train`；当前筛选对照统一使用 `--early_stopping_patience 10`，最多 50 轮；固定轮数协议需所有对照一致。

2026-10-16 文档更新前检查工作区为干净状态；后续接续时应重新检查，不能沿用旧会话的 dirty 状态判断。不要执行 `git reset --hard`、`git checkout --` 等会覆盖改动的命令。

## 6. 已完成验证

历史会话已运行并通过（本次仅修改文档，未重新运行）：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python -m unittest discover -s tests -v
```

结果：11 个测试全部通过，包括轨迹指标、MBTC 数值行为、运动标签、Patch Attention 隔离性和 TACL 反向传播。

也已通过 Python 编译检查：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python -m py_compile \
  model/basemodel.py model/evspsegnet.py utils/mbtcloss.py \
  utils/tacloss.py utils/eval.py dataset/basedataset.py \
  dataset/ev_uav.py train.py test.py utils/args.py \
  scripts/summarize_runs.py
```

`git diff --check` 已通过。历史官方 checkpoint 严格兼容性已验证：legacy 参数约 943,749；提议模型约 950,983；动态 block 数为 1。

## 7. 尚未完成与当前阻塞

用户已反馈 A1 在服务器完成 27 轮训练并早停：最佳验证 IoU=0.3945，测试 IoU=0.5452。该运行采用 patience=10、min_delta=1e-4，且使用当时默认的自动测试。后续对照沿用相同早停协议，仅使用验证集筛选。

创新方法尚未在真实 CUDA/spconv 环境完成：

- 1 epoch smoke test；
- 显存、索引、HAIS_OP、前向/反向兼容性验证；
- A0–A4 训练收敛与真实指标；
- 多随机种子均值/标准差；
- 独立测试集评估；
- 论文最终贡献判断。

当前开发容器执行 `nvidia-smi` 失败，提示无法与 NVIDIA driver 通信，因此不能在本容器声称新方法有效。`runs/` 中 2026-08-12 的历史运行不构成完整 A0–A4 结果，不能直接写入论文作为最终结果。

## 8. 推荐的服务器执行顺序

### 8.1 上传后检查

```bash
git status --short
uv sync
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python -m unittest discover -s tests -v
nvidia-smi
```

若缺少 `HAIS_OP.so`：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python scripts/build_ext.py
```

### 8.2 CUDA smoke test

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav \
  --run_name smoke_a4 --epochs 1 --train_workers 0 \
  --patch_attention sequence --motion_gd shallow --loss mbtc \
  --no-test_after_train
```

确认无 CUDA/spconv 错误并生成 checkpoint 后，再进行正式实验。

### 8.3 第一阶段公平消融（seed 37，50 epochs）

- A0：`--patch_attention legacy --motion_gd none --loss stc`
- A1：`--patch_attention sequence --motion_gd none --loss stc`
- A2：`--patch_attention sequence --motion_gd none --loss mbtc`
- A3：`--patch_attention sequence --motion_gd shallow --loss stc`
- A4：`--patch_attention sequence --motion_gd shallow --loss mbtc`

推荐先跑 A0、A1、A4，再补齐 A2、A3；A1 是区分 Attention 修正与新增方法贡献的关键对照。所有正式训练显式添加 `--no-test_after_train --early_stopping_patience 10`。完整命令见 `TRAINING_COMMANDS.md`。所有配置保持相同数据划分、seed、epoch、batch size、`max_events_num` 和优化器。

### 8.4 结果汇总

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python scripts/summarize_runs.py
```

脚本按最高验证 IoU 选择 checkpoint，同时列出轨迹指标。

### 8.5 多种子和测试集

只有当 A4 相对 A1/A2/A3 有实际收益时，才对最终配置和最强 baseline 跑 seed 13/37/73，并报告均值和标准差。测试集只在验证集冻结阈值后独立运行 `test.py`。

## 9. 结果判定规则

以下是内部研究投入标准，不保证投稿层级或录用。轨迹碎片化必须结合覆盖率和最长连续率判断，避免将完全漏检误认为连续性改善。

- 最低继续线：相对本仓库复现 baseline 的 IoU 至少提升 1 个百分点，且轨迹覆盖或最长连续率相对提升约 10%，Fa 不明显恶化；
- 优先推进论文验证线：IoU 提升 2 个百分点以上，ACC/Pd 至少一项提升，Fa 不恶化，同时轨迹指标改善；
- A3 不优于 A1：不要扩大动态卷积路线，保留 MBTC 损失方向；
- A2 有效但 A4 无额外收益：最终论文不要把 MC-GDSC 列为有效主贡献；
- 只有 A1 提升：只能写成 Patch Attention 复现/缺陷分析，不能宣称完整新模型有效。

## 10. 真实 CUDA 运行时重点排查

- `MotionConditionedGDConv` 的 router 输入是 `features + direction + speed`，维度为 `out_channels + 3`，设计上正确；
- MBTC 的 `_support()` 通过 event-to-voxel `scatter_add` 聚合事件概率/标签，再与 sparse convolution 输出对齐，必须用真实 CUDA smoke test 确认 voxel 数量和 `p2v_map` 对齐；
- `train.py` 中 MBTC 优先使用 `auxiliary["local_motion"]`；浅层 motion tensor 与高分辨率输出可能索引不完全一致，代码已有 fallback，但仍需服务器验证；
- 评估 checkpoint 必须使用与训练一致的 `patch_attention`、`motion_gd`、`loss`、`width` 等结构参数；
- 原 `utils/eval.py` ROC/Pd/Fa 逻辑保留历史定义，论文中需明确指标来源，不能与其他论文自定义定义混用；
- 若 worker 崩溃，先用 `--train_workers 0`；若 OOM，降低 `--max_events_num`，但所有公平对照必须保持一致。

## 11. 下一个 agent 的第一步

1. 阅读本文件、`RESEARCH_PLAN.md`、`TRAINING_COMMANDS.md`；
2. 执行 `git status --short`，确认当前状态并保留现有修改；
3. 优先等待用户从服务器反馈 smoke test 日志；
4. 若用户已上传代码但尚未训练，指导其执行第 8.1 和 8.2 节命令；
5. 若 smoke test 报错，先定位 CUDA/spconv/HAIS_OP、张量 shape、索引或显存问题，再考虑修改代码；
6. 若已有 A0–A4 日志，使用 `scripts/summarize_runs.py` 汇总并据结果决定最终论文主线，不预设 MBTC 或 MC-GDSC 一定有效。

### 2026-10-16 配置整理

训练新增 `--experiment a0/a0_mbtc/attention_fix/a1/a2/a3/a4` 预设，显式参数覆盖预设。Attention 新增 `corrected` 作为纯序列修正对照，原 `sequence` 保持 A1 的增强结构。网格搜索已支持关闭自动测试，训练与搜索默认均不访问测试集。完整参数示例见 `TRAINING_COMMANDS.md` 第 8 节。当前首轮筛选沿用已有 A1 的早停协议；不能与固定 50 轮结果混用。

本次配置整理验证：14 项 CPU 单元测试通过，网格搜索两组 dry-run 通过，未新增 CUDA 训练结果。用户当前优先尝试 a0_mbtc，并需同协议 a0 对照支撑损失贡献。
