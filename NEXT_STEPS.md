# EV-UAV 当前路线与交接

更新：2026-10-16。此文件是唯一接手入口；过时研究计划和旧交接文档已移除。

`AGENTS.md` 和 README 仅引导到此文件，不维护另一份计划。本文记录截至上述日期的证据，接手时重新核验工作区和环境；当前功能以代码为准，运行参数以对应 config.json 为准，论文调研中的候选方案不等于已实现功能。

## 1. 当前目标与保留内容

基于 EV-SpSegNet / EV-UAV，借鉴 ICLR、ICML 方法改善分割与轨迹完整性。先验证困难轨迹风险加权，随后根据错误分析决定是否研究可微路径监督。

当前保留六个预设：

| 预设 | 用途 |
|---|---|
| a0 | 原 Attention 行为、固定 GDSC、STC |
| attention_fix | 仅修正 Attention 序列组织 |
| a1 | 增强 Attention、固定 GDSC、STC；默认模型配置 |
| r1_event | A1 + 前景事件均值 BCE 辅助监督 |
| r2_balanced | A1 + 实例/时间箱均衡 BCE |
| r3_term | A1 + 实例/时间箱 TERM |

原运动条件卷积、运动/embedding 辅助头、旧损失、速度标签生成和旧网格搜索均已删除。模型仍保留原固定 GDSC。历史运行日志、数据与原论文不属于清理范围；有用的数值失败证据简要保留在 EXPERIMENT_RESULTS.md，旧路线不再列为待办。

## 2. 阅读顺序

1. 本文件：实现状态与下一步。
2. [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)：用户提供的真实结果和证据限制。
3. [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md)：当前完整训练、验证、诊断命令。
4. [ML_METHOD_SURVEY.md](ML_METHOD_SURVEY.md)：论文链接、公式、后续方案与停止条件。

## 3. 已知事实

A1 的固定 50 轮、seed 37 测试结果：IoU=0.6230、seg_acc=0.6914、Pd=0.8175、Fa=1.27228e-05、coverage=0.7491、longest=0.6233。实际评估配置和权重身份仍待完整日志核实。seg_acc 是前景召回率；Fa 口径尚未与论文充分对齐。

当前本地有 `/data/ev-uav`，PyTorch 2.13.0+cu126，但 CUDA 不可用。用户在 RTX 3090 / 4090 服务器训练，最近数据路径为 `../ev-uav-data`。所有新风险方法尚无 CUDA 训练结果或精度收益证据。

这些是本次会话观察，不是新机器的环境保证。下一位 agent 先完成本地检查与服务器可运行性验证；若仍无 GPU，则整理运行命令并等待用户返回服务器日志，不声称已完成训练。

## 4. 实现说明

- `utils/trajectory_risk.py`：使用 logits 的 softplus，按 sample / instance / observed bin 聚合。balanced 先平均箱、再平均实例、再平均样本；term 在实例内用 log-mean-exp 替换均值；event 为每样本前景事件均值。
- 三组使用相同 eligible foreground：label>0.5 且 instance_id>0。无有效前景样本贡献零但计入样本平均；无 ID 的前景数量被记录，仍受 STC 监督。
- 时间原点为每条采样后轨迹的首个前景时间。空箱不补负标签；TERM 对时间顺序不敏感，不代表空间连接或严格连续性。
- 风险损失与 STC 相加，保留背景监督。risk_weight=0 跳过辅助分支；beta=0 恢复 balanced；半精度辅助计算升 FP32。
- `model/evspsegnet.py` 保留原概率输出，通过 auxiliary 暴露 segmentation_logits；权重名称与 A1 兼容，不增加可训练参数。
- `utils/args.py`、`train.py` 已接入三个预设、风险参数、分项记录和配置保存。
- `test.py --split val` 可在验证集计算 IoU/Pd/Fa/轨迹指标。默认 split 仍为 test，探索时必须显式 val。评估不自动恢复 checkpoint 配置，也不自动落盘结果，请保存完整命令及输出。
- `--motion_gd` 等旧参数已删除。复用历史 A1 命令时去掉 `--motion_gd none`；旧运动模型权重不再支持，不能用非严格加载绕过。

## 5. 本地验证与边界

清理后完整 CPU 测试 **32 项全部通过**，覆盖风险公式/梯度、beta→0、极端 logits、半精度、空前景、实例隔离、voxel 映射、背景监督、Attention、参数预设及数据采样/拼接。Python 编译、改动格式、CLI 入口和本地 Markdown 链接检查通过。

清理前后 A0 / attention_fix / A1 的参数键、形状和 seed 37 初始化值逐项完全一致，严格加载通过。参数量分别为 943749 / 943749 / 950013。核对用快照不是训练权重。已有 train_000.npz 的人工 logits 接入检查通过：39169 个事件、1 条前景轨迹、129 个有效箱。上述均不能替代 CUDA/spconv 完整训练验证。

## 6. 后续顺序

### P0：保护当前工作区

先查看 `git status --short` 和最近提交，确认当前版本及是否有本地修改。上传时包括新文件和删除项；若在旧服务器目录覆盖上传，要同步删除已移除文件，避免旧模块残留。不要 reset 或重新实现 TERM。运行 CPU 测试，确认服务器 HAIS_OP 可用、GPU 空闲。

`.gitignore` 忽略了 `*.pdf`、`runs/`、checkpoint、数据和编译扩展。只克隆或拉取 Git 不会带来它们：原论文、服务器运行的 config.json / metrics.jsonl / 权重需另外提供；数据路径独立配置；HAIS_OP 在目标环境编译。未跟踪的新源码、测试和交接文档也须实际同步，不能只传 `git diff`。

优先索取的已有证据是 A1 固定 50 轮运行的 config.json、metrics.jsonl、实际测试命令和对应 checkpoint。缺少时可先完成环境检查及新损失 smoke，但不能认定已严格复现历史 62.30% IoU。

### P1：可运行性检查和四组对照

按 TRAINING_COMMANDS.md 跑 r3_term 单轮 smoke，再跑 a1/r1_event/r2_balanced/r3_term：seed 37、固定 50 轮、关闭早停与自动测试，统一学习率、事件上限、阈值与 Attention 后端。

auto 保持历史 A1 行为；如需 math，四组统一并重建 A1 对照。NaN 复现保留原 50 轮学习率计划，不能用 epochs=1 代替。保存 config.json、metrics.jsonl、最佳验证轮次、checkpoint、GPU、单轮耗时及峰值显存。

### P2：依据验证集决定路线

比较 R1-R0（加强前景）、R2-R1（分组均衡）、R3-R2（风险倾斜）。先检查阈值影响，补同 Fa 或同 Pd 比较；不使用测试集选择方法。

新增验证集诊断脚本，按序列/实例/实际时间箱输出事件数、召回、概率、速度及背景误检，检验低密度、快速、转向、多目标是否真是失败原因。

当前 longest 只在有事件的箱序列上计算。新增真实时间/可观测片段指标时保留旧指标名，所有模型一致计算；无事件箱不自动算漏检，也不自动跨过当连续检测。

R3 不优于 R2 时保留简单方案。初筛成立后再有限扫描 lambda∈{0.03,0.1}、beta∈{0,1,2}，补类平衡 BCE/Focal 对照；不进行无界调参。

### P3：有证据后实现路径监督

按 ML_METHOD_SURVEY.md 第 5 节，构建有时间方向、包含目标与背景竞争候选的稀疏图，用两次 log-partition DP 的差监督 GT 兼容路径。尚无该实现或 R4–R7 入口。

先处理不可达 GT 路径、空箱/跳边、图预算、全负无穷分支和短路径偏好；小图穷举核对值与梯度，再做 CUDA。与相同候选预算的局部平滑比较。TERM 和 Path 单独有效后才组合。补查可微数据关联、事件跟踪、track-before-detect 的新颖性。

### P4：最终验证

最强基线与最终方案补 seed 13/37/73，报告均值、标准差和实际效率；补 A0 / attention_fix 对照以分离 A1 增强贡献。配置与阈值冻结后才测试。UOT、SSM 等暂作备选。

## 7. 未完成项

CUDA 训练与性能、多种子、验证集分组导出、真实时间指标、路径层、UOT、SSM、风险参数扫描工具、自动配置恢复。代码提交及远程同步状态以 Git 为准；未创建 PR，也未上传训练服务器。下一位模型从当前文件直接继续。
