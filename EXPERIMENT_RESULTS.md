# EV-UAV 实验结果与论文对照

记录日期：2026-10-20（第 1–8 节保留历史记录，当前结果见第 9 节）

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

## 9. 当前服务器证据（2026-10-20 复核）

第 1–8 节是历史单种子结果及当时的证据边界，不是当前最终结论。现已收到 runs 下的配置、训练指标和完整评估日志；历史 62.30% 不作为当前 math 统一协议的结果。math 对照支持 a1 作为当前最终方案；auto 下的 R1 IoU 优势未在 math 下复现。math 稳定运行支持后端与异常有关，但不能据此独立断言融合内核缺陷。

冻结阈值、三种子（13/37/73）测试集结果，经 `runs/logs/final_test_math_20261019_182010.log` 复核：

| 方法 | IoU（均值±样本标准差） | seg_acc | Pd | Fa |
|---|---|---|---|---|
| a0@0.8 | 0.4960±0.1533 | 0.6589 | 0.7696 | 4.48e-05 |
| a1@0.8 | 0.5625±0.0936 | 0.6805 | 0.7982 | 1.03e-05 |
| r1_event λ=0.1@0.95 | 0.5076±0.1333 | 0.7401 | 0.8126 | 1.86e-05 |

新增 λ=0.03 三种子均完成固定 50 轮 math 训练且关闭早停与自动测试，验证日志 `runs/logs/lambda003_math_20261020_134423.log` 有 9 次 val 结果。统一阈值 0.9 时 IoU=0.4875±0.1330、seg_acc=0.7411、Pd=0.7794、Fa=7.98e-06；a1@0.8 对应 0.4846±0.1049、0.7310、0.7712、5.06e-06。IoU 配对差为 +24.04/−4.08/−19.07pp，不能据均值 +0.30pp 声称稳健改进。λ=0.03@0.95 的平均 Fa=4.70e-06 接近 a1，但 IoU/seg_acc/Pd 均下降，且平均 Fa 接近不等于逐种子同 Fa。

结论：λ=0.03 比 λ=0.1 改善 IoU 与平均 Fa，但未超过 a1 的整体权衡；未追加 test。完整阈值表、配置核对及下一步见 NEXT_STEPS.md 末尾。本次复核不验证 checkpoint 二进制与日志身份的一致性，也未重新核验服务器环境。

## Fa 与原仓库代码核对（2026-10-20）

已只读对照 `../EV-UAV/utils/eval.py`、test.py、configs/evisseg_evuav.yaml 及两边 dataset/ev_uav.py、dataset/basedataset.py。Fa 核心定义完全相同：每个 50ms 窗内取 label=0 且预测>=阈值的事件，将 x/y 投影到 260×346 uint8 图，以 8 连通域数量（不含背景）计虚警；跨序列累加后除以 sum(floor((tmax−tmin)/50))×346×260。不是错误事件比例。两边都读 ev_loc 的 t 列并转 int64，test 不采样，默认 batch_size=1。当前版本只给 Fa/Pd 分母增加 max(...,1) 零分母保护；正常非零分母无差别。旧版 numpy 实例 ID 的 set 改为 tensor.tolist 后 set 不改变正常 ID 分组。

动态核对：直接加载原版 evalute 与当前 evalute，3 个模拟序列，分别用阈值 0.8/0.9/0.95 比较；每次 frame_num/false_num/obj_num/correct_num 及 Pd/Fa 全部完全相等。此项是 CPU 指标核对，不是新增训练或测试集评估。

实质协议差异：原版 test.py 调用 roc_update 不传阈值，固定默认 0.9；本地显式传 prediction_thresh，终评 A1/A0=0.8、R1=0.95。原版 correct_thresh=0.0001 与当前相同，且该参数只影响 Pd、不影响 Fa；原版 ACC 也同样是 TP/(TP+FN)，因此代码层面 ACC 定义已对齐。数据文件内容/划分是否与论文实验完全相同仍未核验；未核对 SpTopoNet 自己的评估仓库，不能把本结论直接扩展为第二篇完全同协议。

共有实现细节：窗口从 0 开始、使用严格开区间（边界事件不计 Pd/Fa）；分子遍历 floor(duration/50)+1 个窗而分母取 floor；uint8 像素累加在 256 次时存在回绕风险。这些不是当前版本新引入的差异，暂不修改以免改变历史口径。结论：撤回“Fa 公式尚不知是否与原 EV-UAV 代码一致”的不确定性；公式已确认一致，保留阈值、数据身份及论文实际运行配置的比较边界。本次修改 NEXT_STEPS.md、EXPERIMENT_RESULTS.md；下一步若严格统一操作点，先在 val 对比同 0.9 或固定 Fa 预算，不追加 test。
