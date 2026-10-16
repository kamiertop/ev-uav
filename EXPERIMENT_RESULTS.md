# EV-UAV 实验结果与论文对照

记录日期：2026-10-16

## 1. 数据来源与状态

本次 A1 数值来自用户提供的服务器测试输出，按对话上下文对应以下运行：

- 运行目录：`runs/train_a1_attention_s37_fixed50_20261016_133736`
- 配置：`patch_attention=sequence`、固定 GDSC、`loss=stc`（历史配置记录为 `motion_gd=none`，该开关现已移除）。
- seed=37，epochs=50，early_stopping_patience=0，train_workers=0。
- 数据路径：`../ev-uav-data`。
- 训练关闭自动测试，结束后用户单独测试并提供结果。
- 预期测试权重：该运行的 `checkpoints/best_iou_seed37.pt`。尚未收到实际测试命令、config.json、metrics.jsonl 或权重文件，权重身份和评估阈值仍待核实。
- 本次最佳验证 IoU、最佳轮次、GPU 型号和实测耗时尚未提供；不可把此前早停运行的验证 IoU=0.3945 当成本次结果。

用户原始输出：

```text
iou=0.6230  seg_acc=0.6914  pd=0.8175  fa=1.27228e-05  coverage=0.7491  longest=0.6233  fragmentation=0.0904
```

A1 是 Attention 修正与增强基线，包含跨 patch 注意力、位置编码、LayerNorm 等调整。该结果属于 A1，不是新风险损失的结果，也不能单独归因于 Attention 维度修正。

## 2. 与两篇论文主表对比

以下为论文报告值与本次单种子测试值的描述性对比，不是同一训练环境下的受控复现实验。

| 方法与来源 | IoU (%) ↑ | ACC / seg_acc (%) ↑ | Pd (%) ↑ | Fa (×10⁻⁴) ↓ |
|---|---:|---:|---:|---:|
| EV-SpSegNet，第一篇论文 Table 2 | 55.18 | 65.02 | 77.53 | 1.63 |
| SpTopoNet，第二篇论文 Table 1 | 66.62 | 74.43 | 83.36 | 1.29 |
| 本次 A1，固定 50 轮，seed 37 | **62.30** | **69.14** | **81.75** | **0.127228** |

Fa 换算：`1.27228e-05 / 1e-4 = 0.127228`。不要将原始输出 `1.27228e-05` 写成论文表格单位下的 `1.27228`。

| 本次 A1 减去论文报告值 | IoU 差值（百分点） | ACC 差值（百分点） | Pd 差值（百分点） | Fa 差值（×10⁻⁴） |
|---|---:|---:|---:|---:|
| 相对 EV-SpSegNet | +7.12 | +4.12 | +4.22 | −1.502772 |
| 相对 SpTopoNet | −4.32 | −5.29 | −1.61 | −1.162772 |

若评估口径一致，Fa 数值分别比上述论文报告值低约 92.19% 和 90.14%。在完成口径核对前，这只是数值差异，不能写成已验证的误检率改进结论。

第二篇论文的消融表 Table 2、Table 3 使用 ACC=74.44、Pd=83.37，与主表分别相差 0.01 个百分点。本记录统一采用主表 Table 1 的 74.43、83.36，不混用表格。

## 3. 与第二篇论文的损失迁移实验对比

第二篇论文 Table 4 的 Ev-SpSegNet* 表示使用 EvTopoLoss：IoU=62.11、ACC=67.80、Pd=81.67、Fa=1.51（×10⁻⁴）。本次 A1 相比该行的数值差异分别为 +0.19、+1.34、+0.08 个百分点；Fa 为 0.127228 对 1.51。

这说明本次 A1 的分割和检测数值已接近该论文的损失迁移实验，但不能据此宣称优于 EvTopoLoss：双方模型配置、训练过程和评估设置未进行受控对齐，0.19 个百分点的 IoU 差异也尚无多种子证据。

## 4. 本次轨迹指标

| 指标 | 本次 A1 |
|---|---:|
| trajectory_coverage ↑ | 74.91% |
| trajectory_longest_ratio ↑ | 62.33% |
| trajectory_fragmentation ↓ | 0.0904 |

两篇论文主结果表没有报告本仓库这三项轨迹指标，不能填写其对应数值或声称超过论文。碎片化必须结合覆盖率和最长连续率分析，完全漏检也可能产生较低碎片化。

## 5. 评估口径与论文表述边界

1. 当前 `utils/eval.py` 中 `seg_acc` 计算 TP/(TP+FN)，实际是前景事件召回率，不是包含背景真负例的整体 accuracy。论文表头为 ACC，最终写作前需核对其补充材料或官方评价定义；此处仅对齐名称展示，尚未独立证明定义完全相同。
2. 当前 IoU 汇总所有事件后计算前景交并比；需核对论文是否采用相同汇总方式、数据划分和采样设置。
3. 当前 Fa 将每个时间窗内误检事件投影到二维平面，统计连通域数，再除以等效帧数与 346×260。它不是简单的错误事件比例。需核对论文的时间窗、阈值、连通域规则和分母定义。
4. 当前参数默认 prediction_thresh=0.9、pd_detT=50、correct_thresh=0.0001；这些只是当前代码默认值，本次真实评估配置尚未收到。不能将它们当作已核实的服务器配置。
5. 原论文报告参数量 4.0M、SpTopoNet 报告 4.4M。当前本地默认 width=12 的 A1 实测 950013 个参数，但服务器该次运行配置尚未完整核实，不能据此补填其参数量。服务器延迟和显存尚未提供，不做效率优势结论。
6. 这是单 seed 结果，没有均值和标准差。需要同协议 A0 对照和多种子实验才能支撑贡献判断。
7. 测试集已评估并留档，后续模块选择和超参数调整应依据验证集。

当前可用结论：本次 A1 的测试 IoU 为 62.30%，在数值上高于 EV-SpSegNet 论文报告的 55.18%，但低于 SpTopoNet 的 66.62%。这些结果为后续研究提供了一个值得保留的增强基线，尚不构成新风险损失有效性的证据。

## 6. 后续实验

- 当前仅推进 A1 与 `r1_event` / `r2_balanced` / `r3_term`，见 TRAINING_COMMANDS.md。
- 统一 seed 37、固定 50 轮、关闭早停与自动测试；Attention 后端、数据、优化器和采样设置一致。
- 收集 config.json、metrics.jsonl、完整评估命令，核实权重身份和最佳验证轮次。
- 新方法仅在验证集筛选，最终对最强基线与候选补多种子。

## 7. 论文来源

- [Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](Event_based_Tiny_Object_Detection_A_Benchmark_Dataset_and_Baseline.pdf)，PDF 第 7 页，Table 2，印刷页码 7215。
- [Towards Persistence: Learning Topological Constraints for Event-based Small Object Detection](Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection.pdf)，PDF 第 7 页 Table 1（印刷页码 22297）；PDF 第 8 页 Table 2、Table 3、Table 4（印刷页码 22298）。

以上表格已从本地 PDF 核对；无额外在线来源。

## 8. 已移除旧路线的数值诊断记录

以下只保留有价值的故障证据，不是待运行实验。原 A2/A3/A4、MBTC 与运动条件卷积已按用户要求删除。

旧运行 `train_a0_mbtc_s37_fixed50_20261016_141735` 在 epoch 3 已出现 NaN，远早于 epoch 48 的显存警告，因此不能认定显存告警是最初原因。其 best validation IoU=0.0159，不是有效的完整训练对照，也未测试。

诊断运行 `train_a0_mbtc_s37_diagnose_20261016_151451` 在 epoch=4、batch=54 捕获 `ScaledDotProductEfficientAttentionBackward0` 的第 0 个输出含 NaN，前向来自 pa3 的 legacy Attention。它定位了被捕获的异常算子，尚未独立证明内核缺陷或排除上游异常梯度。

当前保留通用有限 loss/梯度检查、`--diagnose_numerics`、可选 `--grad_clip_norm` 和 `--attention_backend auto/math`。math 排除融合 SDPA 后端，尚无服务器证据确认其修复效果。正式对照必须统一后端，且不要把单轮 smoke 与原 50 轮学习率计划的异常复现混为一谈。
