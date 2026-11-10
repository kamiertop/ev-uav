# EV-UAV当前状态与接手入口

更新：2026-11-10。开始先读本文件，再查看`git status --short`。不另建重复交接文档。

## 当前目标

主方案已完成服务器test，用户下一步自行复现PACT，然后准备论文。本地只整理代码、必要CPU检查及分析回传日志，不启动训练或实际数据推理，不自动移植PACT或启动新实验。

## 主方案与关键结果

| 方案 | seed/阈值 | split | IoU% | ACC% | Pd% | Fa |
|---|---|---|---|---|---|---|
| **r1_precision：R1+Jaccard+困难背景** | 37/.85 | test | **73.2154** | **77.6941** | **85.7689** | 9.775077e-6 |
| r1_jaccard：低Fa候选 | 37/.85 | test | 68.5301 | 71.8613 | 82.8615 | 7.060276e-6 |

两组均为10月21日训练的旧best_iou权重，已恢复到新服务器，于11月10日冻结阈值终评。权重hash、训练config/metrics和24序列计数核验通过。不是SAM结果，不再只是val73%，不是多seed均值。

主权重目录`runs/train_r1p_r1_precision_w0.1_s37_20261021_104437_461ed8_20261021_115309/checkpoints/best_iou_seed37.pt`，SHA256=`1a4ec27ac0135ae6f4e562724bd836e74bee6e69a6597e4bf52c6b2b6d3aea67`。可版本管理的精确结果/配置/身份见[experiments/verified_r1.json](experiments/verified_r1.json)，完整证据见[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。

主权重训练：50轮/math/seed37/batch1/max_events700000/Adam .01→.001线性、无早停和自动test；风险BCE=.1、Jaccard=.1、困难背景=.03、ratio3/min256。按val@.9选epoch，test阈值.85。组合比Jaccard主要少漏检4165但多FP897，20/24序列IoU提升；不能称hardneg进一步抑制FP。

## 维护范围

只保留8个预设：a0、attention_fix、a1、a1_jaccard、r1_event、r1_jaccard、r1_hardneg、r1_precision。四组R1构成2×2消融，hardneg单项即使不是最佳也需要保留。训练默认仍A1+STC；主方案显式选择r1_precision。

已移除SAM、Lovász、R2实例/时间箱均衡、R3 TERM和一次性历史批次/扫描脚本；不恢复更早运动/embedding模块。已删除无效vis/save开关、风险mode/beta/bin及SAM/IoU模式参数。旧config有这些字段属于原始证据，不能直接全量变成当前CLI。

R1前景损失精简至utils/foreground_loss.py，只有按样本均值的有效前景事件BCE；保留其原公式及事件映射。utils/iou_loss.py仅soft_Jaccard；utils/r1_precision.py保留三项可独立置0的消融。主干、参数键、STC、数据处理与评估口径不改。

## 当前入口

- 安装/训练/验证命令：[TRAINING_COMMANDS.md](TRAINING_COMMANDS.md)。
- `scripts/run_ablation.py`（wrapper `scripts/run_r1_precision.sh`）：默认seed37四组R1、50轮math、val扫描，不访问test；也支持保留基线与指定种子。
- `scripts/test_r1_best.sh`：只用于重放两旧最佳权重，固定test@.85、核对原权重hash，不需要日常重复运行。
- `scripts/summarize_runs.py`：汇总每run最佳val轮；不会将val自动标为test。
- `scripts/build_ext.py`：目标服务器构建HAIS_OP.so。

服务器旧run缺失时必须另传权重；Git不含runs/数据/编译扩展。完整恢复包`runs/preserved/r1_best_test_seed37_20261110.tar.gz`含两旧权重与训练记录；上传项目根按原目录解压，不用新重训权重替代。迁移清理需包含删除项、新文件及uv.lock。

## 本次清理检查

清理前后70组CPU损失/梯度逐元素完全一致；模型/STC/评估/数据源码SHA256保持不变，两份旧最佳权重在CPU严格加载通过。精简后45项CPU测试全部通过；模拟CPU检查+3smoke+4正式训练+24val以及两旧权重重放/打包通过（均未实际推理），8个消融/2个重放命令可被新CLI解析；Python编译、shell语法、文档本地链接及原始两归档hash检查通过。移除无用依赖后uv.lock离线更新，保留包版本完全不变（锁定条目119→58）。没有CUDA训练/精度复验；去掉诊断分支并不证明新的CUDA训练轨迹逐位可复现。

已保存清理前完整文档至runs/preserved/pre_cleanup_docs_20261110.tar.gz，历史源码也在Git及原日志源码快照；失败实验数值保留于EXPERIMENT_RESULTS.md。仅移除活动实现与冗余操作文档，原始结果、权重、数据、论文PDF不删除。

## PACT与论文准备

用户自行在远程服务器运行PACT，回传commit/diff、完整命令/配置、环境、数据清单/hash、验证选择规则、checkpoint hash、预测与四指标。详细记录要求和来源见[ML_METHOD_SURVEY.md](ML_METHOD_SURVEY.md)。表中严格区分作者报告与我们复现，不把未跑的论文数字当本地对照。

我们的四指标数值超过SpTopoNet报告，但与PACT/PointEvent IoU仍差2.68/2.34pp；Fa报告值更低，公开实现仍有口径缺口。PACT当前公开代码每误检事件重复累计连通域，与我们/原EV-UAV每窗一次不同；仅描述版本差异，不推断论文错误。用户复现时保留原生结果及统一评估重算结果。

当前单seed；历史test已多轮使用，不能声称test完全未见。后续开发先val，不为追数字在test扫阈值。必要的后续工作为PACT实测、统一指标、可重复消融/机制证据，不将成熟损失组合自动写成新算法。

本次主要改动：train.py/args与损失实现；新增foreground_loss.py、统一run_ablation.py，删除退役分支/旧批次脚本和对应测试；更新两个runner测试、关键索引experiments/verified_r1.json、pyproject/uv.lock及现有文档，不新增重复交接。

## 工作区与同步

用户现已要求清理并提交推送；包含其已暂存.python-version（3.12.13）和README.zh-CN.md精简内容，本次中文README追加研究入口，保留原修改。按用户要求提交并推送清理，最终状态以Git为准。results/与runs/被忽略，关键小型索引experiments/verified_r1.json随代码维护。独立训练服务器环境、数据路径和GPU占用需要每次重新核验。

## 2026-11-10 历史文件清理与提交

当前仅展开四组R1消融训练目录、对应两批R1日志，以及最近SAM/主方案结果证据。其余历史训练目录、旧扫描日志和重复解压文件共105个顶层项/737文件已压缩到`runs/preserved/historical_runs_20261110.tar.gz`，所有737文件从包内读回核对SHA256、并重核源文件后才删除展开副本；没有丢弃独有原始实验内容。原展开内容668571986字节，归档501859912字节，SHA256=`91ddbb22f25d5fb88404196a75fee07fabf91b67f92ad97eac61f194e03a8769`。完整成员路径/hash及删除清单在同目录historical_runs_20261110.manifest.json。

原始根目录结果包和两份最佳权重归档保留；部分历史CSV/文档记录中的旧物理路径已转入该历史包，需要时在项目根按原布局解压恢复。不要因展开目录已清理就认定历史证据丢失。build中间产物和项目Python缓存可删除，运行时根目录HAIS_OP.so及.venv保留。该本地日志清理不随Git删除服务器上的runs，服务器需自行保留/归档。
