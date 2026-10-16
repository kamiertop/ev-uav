# EV-UAV 实验结果与论文对照

记录日期：2026-10-16

## 1. 数据来源与状态

本次 A1 数值来自用户提供的服务器测试输出，按对话上下文对应以下运行：

- 运行目录：`runs/train_a1_attention_s37_fixed50_20261016_133736`
- 配置：`patch_attention=sequence`、`motion_gd=none`、`loss=stc`。
- seed=37，epochs=50，early_stopping_patience=0，train_workers=0。
- 数据路径：`../ev-uav-data`。
- 训练关闭自动测试，结束后用户单独测试并提供结果。
- 预期测试权重：该运行的 `checkpoints/best_iou_seed37.pt`。尚未收到实际测试命令、config.json、metrics.jsonl 或权重文件，权重身份和评估阈值仍待核实。
- 本次最佳验证 IoU、最佳轮次、GPU 型号和实测耗时尚未提供；不可把此前早停运行的验证 IoU=0.3945 当成本次结果。

用户原始输出：

```text
iou=0.6230  seg_acc=0.6914  pd=0.8175  fa=1.27228e-05  coverage=0.7491  longest=0.6233  fragmentation=0.0904
```

A1 是 Attention 修正与增强基线，包含跨 patch 注意力、位置编码、LayerNorm 等调整；未启用 MC-GDSC 或 MBTC。因此本次结果不能归因于这两个创新模块，也不能单独归因于 Attention 维度修正。

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
5. 原论文报告参数量 4.0M、SpTopoNet 报告 4.4M；本次 A1 未提供实测参数量、延迟和显存，不照抄这些值，也不做效率优势结论。
6. 这是单 seed 结果，没有均值和标准差。需要同协议 A0 对照和多种子实验才能支撑贡献判断。
7. 测试集已评估并留档，后续模块选择和超参数调整应依据验证集。

当前可用结论：本次 A1 的测试 IoU 为 62.30%，在数值上高于 EV-SpSegNet 论文报告的 55.18%，但低于 SpTopoNet 的 66.62%。这些结果为后续研究提供了一个值得保留的增强基线，尚不构成 MBTC 或 MC-GDSC 有效性的证据。

## 6. 后续实验

- 优先运行原 Attention + MBTC（`--experiment a0_mbtc`），并补齐同协议 A0 + STC。
- A2 与本次 A1 对照，用于检验 MBTC 在增强 Attention 上的收益。
- 与本次结果直接比较的训练统一使用 seed 37、最多 50 轮且关闭早停，并保持数据、优化器和采样设置一致；不得与早停运行混为同一组实验。
- 收集 config.json、metrics.jsonl、完整测试命令，核实最佳验证轮次和评估设置。

## 7. 论文来源

- [Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](Event_based_Tiny_Object_Detection_A_Benchmark_Dataset_and_Baseline.pdf)，PDF 第 7 页，Table 2，印刷页码 7215。
- [Towards Persistence: Learning Topological Constraints for Event-based Small Object Detection](Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection.pdf)，PDF 第 7 页 Table 1（印刷页码 22297）；PDF 第 8 页 Table 2、Table 3、Table 4（印刷页码 22298）。

以上表格已从本地 PDF 核对；无额外在线来源。

## 8. A0 + MBTC 失败运行记录

用户随后反馈运行 `runs/train_a0_mbtc_s37_fixed50_20261016_141735`：预设 a0_mbtc、seed 37、50 轮、关闭早停、train_workers=0，数据路径 ../ev-uav-data。

- epoch 48、49 的 loss 为 NaN；最早出现 NaN 的轮次未知。
- epoch 48 显存分配警告：申请 201326592 bytes（192 MiB），空闲 66781184 bytes（约 63.7 MiB），总显存约 23.53 GiB。进程继续运行，并非日志中有未捕获 OOM 异常退出。
- 训练输出 best loss=0.2311、best validation IoU=0.0159（1.59%）。未评估测试集。
- 该运行标为数值失败，不能作为有效的完整 50 轮性能对照，也不能仅凭此否定 MBTC 研究方向。最佳早期权重是否有效需另行核实。
- 日志已在 OOM 警告前显示 NaN，不能认定本次 OOM 是最初 NaN 的原因。需要完整 metrics.jsonl 和 GPU 占用情况定位。

本地诊断改动：优化器更新前检查有限 loss 与梯度，异常时停止并记录损失分项、epoch/batch、事件/体素数量和显存；提前清理上一批梯度；新增可选 grad_clip_norm（默认 0 不裁剪）。17 项 CPU 测试通过，尚未通过服务器验证，未声称已修复 NaN 根因。

### 首次 NaN 的新增证据

用户补充：epoch=3、batch=9 的 loss=0.23079845，seg=0.08995280、trajectory=0.11869538、motion_direction=1.04020631、motion_speed=0.07151183，均有限；batch=10 起预测相关分项全部 NaN，而 gt_support=0.00977253 仍有限，batch=11 持续异常。学习率为 0.00944897959。

这将异常起点前移到 epoch 3，远早于 epoch 48 的 OOM 警告。有限的前向损失不保证反向梯度有限；目前需要区分 batch 9 的反向/更新损坏与 batch 10 的数据/前向异常。motion_gd=none 下 route_balance 来自 probabilities.sum()*0，预测含 NaN 时该占位零也会成为 NaN，并不表示启用了路由模块。

新增可选 `--diagnose_numerics`：启用 autograd 异常追踪，检查输入、输出、更新后参数与缓冲区，并记录 failure_stage。该模式有额外耗时，仅用于定位。复现时保留 epochs=50 的学习率计划；改为 epochs=4 会改变线性衰减速度，不能准确复现本次异常。尚未确认根因。

### 诊断运行定位到 Attention 反向算子

运行 `train_a0_mbtc_s37_diagnose_20261016_151451` 在 epoch=4、batch=54 停止，autograd 报告 `ScaledDotProductEfficientAttentionBackward0` 的第 0 个输出含 NaN，前向调用来自 `pa3` 的 legacy Attention。这确定了此次被捕获的异常算子，尚不能独立证明是 PyTorch 内核缺陷，亦不能排除其上游梯度数值异常。

新增 `--attention_backend auto/math`，默认 auto 保持原行为；math 使用 PyTorch 的 `sdpa_kernel(SDPBackend.MATH)` 排除融合 SDPA 后端，保持模型结构与损失不变。先按原 50 轮学习率计划、同 seed 加 math 和 diagnose_numerics 重跑，不同时修改其他超参数。后端变更须记录在实验配置中；若最终采用 math，正式对照应统一后端。CPU 前向/反向对照不能替代服务器验证。
