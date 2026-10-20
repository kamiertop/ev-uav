# EV-UAV 当前路线与交接

更新：2026-10-20。此文件是唯一接手入口；过时研究计划和旧交接文档已移除。

`AGENTS.md` 和 README 仅引导到此文件，不维护另一份计划。本文记录截至上述日期的证据，接手时重新核验工作区和环境；当前功能以代码为准，运行参数以对应 config.json 为准，论文调研中的候选方案不等于已实现功能。

## 1. 当前目标与保留内容

基于 EV-SpSegNet / EV-UAV，借鉴 ICLR、ICML 方法改善分割与轨迹完整性。先验证困难轨迹风险加权，随后根据错误分析决定是否研究可微路径监督。

当前保留八个预设（2026-10-20 新增两个 IoU 对照）：

| 预设 | 用途 |
|---|---|
| a0 | 原 Attention 行为、固定 GDSC、STC |
| attention_fix | 仅修正 Attention 序列组织 |
| a1 | 增强 Attention、固定 GDSC、STC；默认模型配置 |
| a1_lovasz | A1 + binary Lovász hinge（IoU 对照） |
| a1_jaccard | A1 + soft Jaccard（IoU 对照） |
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

2026-10-17 服务器会话核验：RTX 4090 ×2，GPU 0 空闲、GPU 1 被其他进程占用，PyTorch 2.13.0+cu126 CUDA 可用，HAIS_OP 可导入，`../ev-uav-data` train/val/test 齐备，32 项 CPU 测试全部通过。r3_term 单轮 CUDA smoke 完成：`runs/train_smoke_r3_term_20261017_134619`，99 batch 约 72 秒，best loss 0.1861、单轮 IoU 0.1648，无 NaN；训练中出现两次 allocator OOM 告警（约 148/296 MB 分配失败）但自动恢复，说明 batch_size=1 + max_events_num 700000 在 24GB 卡上显存偏紧，四组正式运行需关注峰值显存。该 smoke 仅验证可运行性，不代表精度。四组固定 50 轮对照改由用户自行执行 `scripts/run_s37_fixed50.sh`（含 smoke、四组顺序训练、失败即停、GPU 显存采样与日志落盘）。

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

2026-10-17 已完成：环境核验、r3_term 单轮 smoke（见第 3 节）及四组固定 50 轮对照（用户经 `scripts/run_s37_fixed50.sh` 执行，GPU 4090，每组约 31 分钟，峰值显存 23916 MiB，全程无 NaN/无 training_error，auto 后端在原 50 轮线性学习率计划下未复现历史 NaN）。验证 IoU 均为 prediction_thresh 0.9、每轮验证的最优轮取值：

| 组 | 最优轮 | IoU | seg_acc | coverage | longest |
|---|---|---|---|---|---|
| a1 (R0) | 26 | 0.3960 | 0.7415 | 0.7311 | 0.5396 |
| r1_event (R1) | 42 | **0.4227** | 0.7610 | 0.7284 | 0.5443 |
| r2_balanced (R2) | 44 | 0.4043 | 0.7491 | 0.7138 | 0.5482 |
| r3_term (R3) | 28 | 0.4182 | 0.7448 | 0.7223 | 0.5434 |

差值：R1−R0=+2.67pp（前景事件均值有效）；R2−R1=−1.84pp（实例/箱均衡反而变差）；R3−R2=+1.39pp（TERM 在均衡之上回升）；R3−R1=−0.45pp（TERM 未超过 R1）。单种子、最优轮选择存在同步乐观偏差，只能作排序参考。风险分支诊断健康：末轮 unassigned=0、empty_samples=0、grad_norm≈0.015、risk_weighted≈0.008–0.012。

另澄清：runs 中 9 月 2 日 `a1_attention_s37_20260902_163752`（最优 IoU 0.558）是带 tversky/focal/embedding/route 等旧损失及早停的另一套配置，与当前 A1+STC 预设不可比；历史 62.30% 测试 IoU 的配置身份仍属旧路线，不能用当前协议复现。昨日与今日两次当前协议 a1 分别 0.4012 / 0.3960，一致。

run 目录：`runs/train_{a1,r1_event,r2_balanced,r3_term}_s37_fixed50_auto_20261017_*`；日志与显存采样在 `runs/logs/four_runs_s37_fixed50_auto_20261017_140342.log`、`runs/logs/gpu0_memory_20261017_140342.csv`。

auto 保持历史 A1 行为；如需 math，四组统一并重建 A1 对照。NaN 复现保留原 50 轮学习率计划，不能用 epochs=1 代替。保存 config.json、metrics.jsonl、最佳验证轮次、checkpoint、GPU、单轮耗时及峰值显存。

### P2：依据验证集决定路线

四组 seed 37 初筛已完成（见 P1）：R1 最优，R2 劣于 R1，R3 回升但未超 R1。2026-10-17 阈值敏感性验证集评估完成（`runs/logs/val_eval_s37_20261017_163508.log`，`scripts/eval_s37_val.sh`）：阈值 0.9 下 r1_event IoU 0.4227 / Pd 0.8022 / Fa 7.17e-06 领先，但 Fa 略高于 a1 的 5.86e-06；同 Fa 对比后 R1 优势仍然成立——Fa≈4.5e-06 处（双方 0.95 阈值）r1_event 比a1 IoU +3.3pp（0.4209 vs 0.3879）、Pd +2.8pp（0.7858 vs 0.7577）；Fa≈5.9e-06 处 r1_event@0.95 比 a1@0.9 IoU +2.5pp、Pd −0.4pp。r3_term@0.9（Fa 5.94e-06≈a1 同阈值）比 a1 IoU +2.2pp、Pd +0.5pp，但仍低于 R1。阈值降至 0.7 时四组 coverage/longest 大幅上升（a1 longest 0.606），说明轨迹完整性与阈值强相关且风险损失未改善轨迹层指标；R1 收益仍在分割层。

多种子验证已完成（2026-10-17 晚；`runs/logs/multi_seed_fixed50_auto_20261017_170700.log`，run 目录 `runs/train_{a1,r1_event}_s{13,73}_fixed50_auto_2026101*`）。三种子验证集（阈值 0.9，best_iou checkpoint）：

| 指标 | a1 | r1_event | 配对差（s37/s13/s73） |
|---|---|---|---|
| IoU | 0.4508±0.1105 | 0.4909±0.1366 | +0.0267 / +0.0234 / +0.0702，均值 +4.0pp |
| Pd | 0.7685±0.0186 | 0.7774±0.0298 | +0.012 / +0.026 / −0.012 |
| Fa | 5.18e-06 | 7.89e-06 | r1 略高 |

**结论：R1 的 IoU 增益三种子方向全部一致**（配对 t≈2.66，df=2，n=3 不足达显著）；Pd 混合、Fa 略高。seed 73 两组异常高（a1 0.5779 / r1 0.6481），种子间方差大于方法间差异，报告时须如实呈现均值±标准差。R1 定为保留方案：下一步补 A0 / attention_fix 对照分离 A1 增强贡献，阈值冻结后按 P4 测试集终评；风险损失未改善轨迹层指标的问题留给验证集分组诊断。运行备注：16:57 首次启动因外部进程挤占 GPU 0 约 7 GiB 在 epoch 0 OOM，17:07 重启正常，高峰显存约 23.9 GiB 接近 24 GiB 上限，正式运行期间勿向 GPU 0 派其他任务；multi_seed 首版 VAL 块因 RUN_DIR 未初始化在用户 shell 下报未绑定变量、评估全部未执行，已由 agent 手动补评并加固脚本（初始化 + 缺失跳过 + 评估失败不误杀后续训练）。

新增验证集诊断脚本，按序列/实例/实际时间箱输出事件数、召回、概率、速度及背景误检，检验低密度、快速、转向、多目标是否真是失败原因。重点解释 R2 为何劣于 R1（分组均衡稀释了什么）。

当前 longest 只在有事件的箱序列上计算。新增真实时间/可观测片段指标时保留旧指标名，所有模型一致计算；无事件箱不自动算漏检，也不自动跨过当连续检测。

R3 不优于 R2 时保留简单方案。初筛成立后再有限扫描 lambda∈{0.03,0.1}、beta∈{0,1,2}，补类平衡 BCE/Focal 对照；不进行无界调参。

### P3：有证据后实现路径监督

按 ML_METHOD_SURVEY.md 第 5 节，构建有时间方向、包含目标与背景竞争候选的稀疏图，用两次 log-partition DP 的差监督 GT 兼容路径。尚无该实现或 R4–R7 入口。

先处理不可达 GT 路径、空箱/跳边、图预算、全负无穷分支和短路径偏好；小图穷举核对值与梯度，再做 CUDA。与相同候选预算的局部平滑比较。TERM 和 Path 单独有效后才组合。补查可微数据关联、事件跟踪、track-before-detect 的新颖性。

### P4：最终验证

收尾三步（2026-10-17 已备好脚本）：① `scripts/run_baseline_seeds.sh`：a0 / attention_fix 三种子（13/37/73）训练 + 自动 val 评估，分离 A1 增强贡献；② `scripts/sweep_thresholds_val.sh`：a1 / r1_event 六个 best_iou checkpoint 在 val 上扫 0.7/0.8/0.9/0.95，按同 Fa 预算下 IoU 最优冻结唯一阈值；③ 冻结后对 a1 与 r1_event 各三种子 `--split test` 各评一次出最终数字，此后不再碰测试集。UOT、SSM 等暂作备选。

**2026-10-19 math 全量对照与结论修正**：diag 确认 a0 在 math 后端 50 轮稳定（触发器=SDPA 融合后端，坐实 EXPERIMENT_RESULTS.md 历史定位）。12 组 math 三种子（`runs/logs/all_math_fixed50_20261019_104831.log`，run 目录 `*_fixed50_math_*`）+ 54 次阈值扫描（`runs/logs/thresh_sweep_val_math_20261019_171351.log`，含 r1_event 0.97/0.98 补点）后，**此前 auto 世界的 R1 优势在 math 下反转，撤回"R1 提升 IoU"的结论**：

| 预设（math, val） | 三种子 IoU（各取 val 峰值阈值） | 同 Fa 配对 r1−a1 |
|---|---|---|
| a0 | 0.434/0.401/0.376，峰值阈值 ~0.7–0.8，mean≈0.404 | — |
| attention_fix | 0.193/0.251/0.165，Fa 1e-4 级，mean≈0.203 | — |
| a1 | 0.401/0.454/0.602，峰值阈值 0.7–0.8，mean≈0.485 | — |
| r1_event | 0.393/0.424/0.395，峰值阈值 0.90–0.95，mean≈0.404 | −0.7/−3.0/−20.5pp，全负 |

关键事实：r1_event 同种子跨后端剧烈漂移（s73：auto 0.6481 → math 0.3952），a1 跨后端稳定（s73：0.5779→0.5985，mean 0.45→0.48）——风险损失的 IoU 增益对训练轨迹/后端数值不稳健。r1_event 在同 Fa 下 Pd/coverage/longest 更好（s13：Pd 0.853 vs 0.799、coverage 0.863 vs 0.647），定位改为"召回/轨迹连续性变体"，非 IoU 赢家。a0-math 稳定（±0.031），历史"a0 不稳定且差"部分是 auto 后端毒性；attention_fix 两个后端下都差（序列组织修正单独无益）。

**冻结决定（2026-10-19）**：最终方法 a1，冻结阈值 0.8（三种子 val 峰值平台）；r1_event 作为召回变体，冻结阈值 0.95；a0@0.8 作为架构参考行。终评命令：`./scripts/final_test.sh a1:0.8 r1_event:0.95 a0:0.8`，执行后测试集封存。attention_fix 仅 val 行，不入终评。

**测试集终评已完成（2026-10-19 18:22，`runs/logs/final_test_math_20261019_182010.log`），测试集就此封存**。冻结协议（math 后端、seed 13/37/73、固定 50 轮、各自冻结阈值、best_iou checkpoint）：

| 方法（test） | IoU | Pd | Fa | coverage | longest |
|---|---|---|---|---|---|
| a0 @0.8 | 0.4960±0.1533 | 0.7696±0.0358 | 4.48e-05 | 0.6862 | 0.5227 |
| **a1 @0.8（最终方法）** | **0.5625±0.0936** | 0.7982±0.0211 | **1.03e-05** | 0.7241 | 0.5970 |
| r1_event @0.95（召回变体） | 0.5076±0.1333 | **0.8126±0.0691** | 1.86e-05 | **0.7360** | **0.6176** |

**2026-10-19 后方向调整（用户决定）**：核心指标聚焦 IoU / seg_acc / Pd / Fa（test 上 a1 acc 均值 0.6805、r1 0.7401），coverage/longest 不再作为目标。当前格局为精度-召回拉锯：a1 赢 IoU/Fa，r1_event(λ=0.1) 赢 acc/Pd 但输出过热。下一步执行 P2 预留的 λ 扫描第一步：`./scripts/run_lambda003.sh`（r1_event λ=0.03 × seed 13/37/73，math 协议，每组自动 val 评估 0.85/0.9/0.95）。判定标准：同 Fa 下四核心指标对 a1 与 r1@0.1 的配对优势。注意：测试集已消耗一次；若 λ=0.03 成为新最终方法，需以"预注册的最终一次"方式再做一次 test 并在论文中透明报告两次经历，或仅以 val 报告新迭代。


**2026-10-20 本地日志复核：λ=0.03 已完成**。已对照 `EXECUTED_SCRIPTS.md`、`runs/logs/lambda003_math_20261020_134423.log` 与三个 `train_r1l003_*_20261020_*` 的 config.json / metrics.jsonl：seed 13/37/73 均为 math、risk_weight=0.03、50 轮、50 次验证、early_stopped=false、关闭自动测试；日志有 9 次 val 结果及 ALL DONE。10-19 OOM 假启动不纳入统计。以下均为三种子算术均值，IoU 的 ± 为样本标准差；阈值对每种方法跨种子统一，不逐种子选峰值。

| 方法（val） | 阈值 | IoU | seg_acc | Pd | Fa |
|---|---|---|---|---|---|
| a1 | 0.8 | 0.4846±0.1049 | 0.7310 | 0.7712 | 5.06e-06 |
| r1 λ=0.1 | 0.95 | 0.4007±0.0206 | 0.7524 | 0.7887 | 8.76e-06 |
| r1 λ=0.03 | 0.85 | 0.4831±0.1242 | 0.7522 | 0.7962 | 1.25e-05 |
| r1 λ=0.03 | 0.9 | 0.4875±0.1330 | 0.7411 | 0.7794 | 7.98e-06 |
| r1 λ=0.03 | 0.95 | 0.4811±0.1322 | 0.7185 | 0.7493 | 4.70e-06 |

λ=0.03@0.9 相对 a1@0.8 的 IoU 配对差（s13/s37/s73）为 +24.04/−4.08/−19.07pp，均值仅 +0.30pp，Fa 增加约 58%；无法视为稳定收益。λ=0.03@0.95 与 a1@0.8 仅在平均 Fa 上接近（并非逐种子同 Fa），IoU −0.35pp、seg_acc −1.25pp、Pd −2.20pp。λ=0.03@0.9 相对 λ=0.1@0.95：IoU +8.68pp、平均 Fa 下降约 9%，但 seg_acc −1.13pp、Pd −0.93pp。因此降低 λ 缓解了 IoU 损失，却未带来四核心指标一致优势，当前保留 a1 为最终方案，不据此追加 test。

下一步若继续研究，先在 val 做固定 Fa 预算下的逐种子比较及前景/背景错误诊断，尤其核查 s13/s73 的训练差异；现有离散阈值不足以声称严格同 Fa 优势。本次只读本地日志，未运行训练或新增测试集评估；已改文件为 NEXT_STEPS.md、EXPERIMENT_RESULTS.md。

2026-10-20 论文对照复核：本地第一篇 PDF 第 7 页 Table 2、第二篇 PDF 第 7–8 页 Table 1/4 已核对。A1 三种子 test 均值相对 EV-SpSegNet 论文主表 IoU/seg_acc/Pd 数值差为 +1.07/+3.03/+2.29pp；相对 SpTopoNet 为 −10.37/−6.38/−3.54pp。R1 最佳单次 test（s37）65.97/74.32/81.01%，相对 SpTopoNet 差 −0.65/−0.11/−2.35pp，不能据此宣称全面超过。Fa 统一为 ×10⁻⁴ 后：论文 EV-SpSegNet 1.63、SpTopoNet 1.29，本地 A1 均值 0.1028、R1 均值 0.1858、R1 s37 0.1885；评估口径尚未对齐，仅作数值对照。λ=0.03 仍仅 val。本次只更新本交接文件，未运行新增评估。


## Fa 与原仓库代码核对（2026-10-20）

已只读对照 `../EV-UAV/utils/eval.py`、test.py、configs/evisseg_evuav.yaml 及两边 dataset/ev_uav.py、dataset/basedataset.py。Fa 核心定义完全相同：每个 50ms 窗内取 label=0 且预测>=阈值的事件，将 x/y 投影到 260×346 uint8 图，以 8 连通域数量（不含背景）计虚警；跨序列累加后除以 sum(floor((tmax−tmin)/50))×346×260。不是错误事件比例。两边都读 ev_loc 的 t 列并转 int64，test 不采样，默认 batch_size=1。当前版本只给 Fa/Pd 分母增加 max(...,1) 零分母保护；正常非零分母无差别。旧版 numpy 实例 ID 的 set 改为 tensor.tolist 后 set 不改变正常 ID 分组。

动态核对：直接加载原版 evalute 与当前 evalute，3 个模拟序列，分别用阈值 0.8/0.9/0.95 比较；每次 frame_num/false_num/obj_num/correct_num 及 Pd/Fa 全部完全相等。此项是 CPU 指标核对，不是新增训练或测试集评估。

实质协议差异：原版 test.py 调用 roc_update 不传阈值，固定默认 0.9；本地显式传 prediction_thresh，终评 A1/A0=0.8、R1=0.95。原版 correct_thresh=0.0001 与当前相同，且该参数只影响 Pd、不影响 Fa；原版 ACC 也同样是 TP/(TP+FN)，因此代码层面 ACC 定义已对齐。数据文件内容/划分是否与论文实验完全相同仍未核验；未核对 SpTopoNet 自己的评估仓库，不能把本结论直接扩展为第二篇完全同协议。

共有实现细节：窗口从 0 开始、使用严格开区间（边界事件不计 Pd/Fa）；分子遍历 floor(duration/50)+1 个窗而分母取 floor；uint8 像素累加在 256 次时存在回绕风险。这些不是当前版本新引入的差异，暂不修改以免改变历史口径。结论：撤回“Fa 公式尚不知是否与原 EV-UAV 代码一致”的不确定性；公式已确认一致，保留阈值、数据身份及论文实际运行配置的比较边界。本次修改 NEXT_STEPS.md、EXPERIMENT_RESULTS.md；下一步若严格统一操作点，先在 val 对比同 0.9 或固定 Fa 预算，不追加 test。

2026-10-20 用户明确下一目标为提高 IoU。本轮联网核对 CVPR 2018 Lovász、NeurIPS 2023 JML、JMLR 2023 RankSEG、ICCV 2021 ASL、CVPR 2024 PTv3 等主来源，具体链接/适配/边界见 ML_METHOD_SURVEY.md 第10节。优先 A1+Lovász 与 A1+soft Jaccard 两个独立对照，先补 val 的事件级 TP/FP/FN/precision；R1 在 val 的重建 precision 三种子均低于 A1，说明低连通域 Fa 不能排除高事件 FP。以上尚未实现/训练，未恢复旧模块；不继续优先 TERM/路径监督。当前只更新 ML_METHOD_SURVEY.md、NEXT_STEPS.md，保留其他未提交修改。下一步是在现有 A1 上实现最小 IoU 对齐对照及必要数值测试，再交服务器 math 固定50轮三种子验证；test不用于筛选。

## 2026-10-20 IoU 对照已实现，待服务器执行

新增 `utils/iou_loss.py`：binary Lovász hinge / hard-label soft Jaccard，事件logits经p2v_map映射，包含背景，按样本平均；半精度升FP32、空输入安全、Jaccard空前景回退背景BCE，weight=0跳过。新预设 `a1_lovasz/a1_jaccard` 与 A1 同主干，无新可训练参数、不叠R1；args/train已接入独立模式与权重。训练新增每轮耗时/显存峰值，eval/test新增TP/FP/FN/precision及可选JSON逐序列诊断；原IoU/Pd/Fa实现不变。

新增 `scripts/run_iou_ablation.sh/.py`：环境检查/CPU测试、两个候选单轮smoke、9组50轮math正式训练（37/13/73×A1/Lovász/Jaccard）、每组5个阈值val评估、唯一run目录绑定、失败即停、轻量结果打包（不含checkpoint）。正式训练按val@0.9选择best checkpoint，阈值扫描只能跨种子统一选取，test不参与。本地40项CPU测试通过，覆盖Lovász离散Jaccard顶点一致性/两损失gradcheck、空/半精度/样本分组/事件映射/关闭分支/全局计数。脚本dry-run、CLI入口、编译与shell语法检查通过；以模拟子进程核对2次smoke/3组正式训练/15次val评估及压缩包排除权重的编排通过（不代表真实CUDA训练）。本地PyTorch2.13+cu126 CUDA不可用，未运行服务器训练或推断提升。

本轮改动：utils/iou_loss.py、utils/args.py、train.py、utils/eval.py、test.py、tests/test_iou_loss.py、tests/test_experiment_config.py、scripts/run_iou_ablation.sh/.py、TRAINING_COMMANDS.md、本文件；保留既有未提交文档及用户结果。下一步用户同步全部改动和新增文件到服务器，在空闲GPU执行 `bash scripts/run_iou_ablation.sh --gpu 0 --data-dir ../ev-uav-data`，成功后回传自动生成的结果tar.gz，权重保留服务器。命令及可选分批运行见 TRAINING_COMMANDS.md 当前优先小节。

## 7. 未完成项

P1–P4 主线已闭环：四组对照、阈值敏感性、auto→math 协议切换、12 组 math 三种子表、阈值冻结与测试集终评均完成（见 P1/P2/P4），测试集已封存。剩余研究方向：验证集分组诊断脚本（轨迹层为何不动、R2 为何劣于 R1）、r1_event 方差与校准问题、P3 路径监督（ML_METHOD_SURVEY.md 第 5 节）、真实时间指标、UOT、SSM、自动配置恢复。代码提交及远程同步状态以 Git 为准；未创建 PR。下一位模型从当前文件直接继续。
