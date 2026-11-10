# 方法依据与PACT复现记录要求

更新：2026-11-10。当前只维护已验证的R1路线与必要消融，未实现候选不再作为训练待办。

## 当前方法及证据

主方案是在A1增强Attention主干上组合已有监督项，不增加预测头或推理参数：

```text
L = STC + 0.1 × foreground_BCE + 0.1 × soft_Jaccard + 0.03 × hard_background_BCE
```

foreground_BCE只作用于label>.5且instance_id>0的事件，样本内事件均值、样本间等权；无有效前景的样本贡献零且计入均值。其作用是普通前景事件监督，不是轨迹连通性或实例/时间箱均衡。

soft_Jaccard使用事件logits和硬二值GT，含背景，按样本平均。空前景回退背景BCE；不称为软标签JML方法。hard_background_BCE在每样本GT背景中选最高logit的k个事件，k=min(Nbg,max(256,ceil(3*Nfg)))，再取softplus均值；仅训练时使用GT选择，无测试后处理。

三个辅助项各自可用权重0关闭。R1原始、Jaccard单项、困难背景单项和组合保留为完整2×2消融。当前单seed组合test IoU73.22/ACC77.69/Pd85.77/Fa9.78e-6；错误分析表明组合比Jaccard主要减少漏检，不能仅由“hard background”命名声称它降低误检。

上述成熟损失组合及单数据集精度提升不自动构成论文算法新颖性；需结合实际机制、可重复消融、误检/漏检取舍及复现基线论证。当前主干保留A0/Attention修正/A1的对照，不将历史测试集当未见数据。

## 直接相关论文

| 方法 | 来源 | EV-UAV作者报告IoU/ACC/Pd/Fa |
|---|---|---|
| EV-SpSegNet/STC | [ICCV2025原文](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Event-based_Tiny_Object_Detection_A_Benchmark_Dataset_and_Baseline_ICCV_2025_paper.pdf)，[原仓库](https://github.com/ChenYichen9527/Ev-UAV) | 55.18/65.02/77.53/1.63e-4 |
| SpTopoNet | [CVPR2026官方页](https://openaccess.thecvf.com/content/CVPR2026/html/He_Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection_CVPR_2026_paper.html) | 66.62/74.43/83.36/1.29e-4 |
| PACT | [论文](https://arxiv.org/html/2606.22378v1)，[代码](https://github.com/fulongcai/PACT)，作者标注ECCV2026 | 75.90/80.05/91.84/7.60e-5 |
| PointEvent | [2026-09预印本](https://arxiv.org/html/2609.20066v1)，[代码](https://github.com/wzz-z/PointEvent) | 75.56/81.61/91.75/1.20e-5 |

前3指标单位%，Fa为绝对值；论文表中的0.12×10^-4=1.2e-5。上述报告值未在本地统一复现，不能替代用户即将执行的PACT实测。

Jaccard相关参考：[Lovász-Softmax，CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_Lovasz-Softmax_Loss_CVPR_2018_paper.html)、[Jaccard Metric Losses，NeurIPS2023](https://arxiv.org/abs/2302.05666)。当前只保留硬标签soft_Jaccard计算，不保留Lovász实验实现，不等同复现这些论文完整方法。

## PACT由用户独立复现

用户下一步自行在训练服务器运行PACT，本项目不自动移植PACT模块或启动训练。论文中应分开列出“作者报告”与“我们的复现”，不把不同阈值/采样/训练设置的结果合并。

回传时至少保留：

1. PACT仓库commit、任何修改的diff、实际配置/命令、环境/GPU及数据路径。
2. train/val/test文件清单与内容哈希；尤其不能只由文件名相同推断数据身份一致。
3. 随机种子、训练事件上限/采样、8秒窗口、batch、优化器、学习率计划、轮数和早停。
4. 验证集选epoch和阈值的规则、实际checkpoint SHA256、测试逐序列TP/FP/FN及四指标。
5. 完整预测（事件顺序、坐标、分数、GT/实例ID映射）或足以通过统一评估器重算的输出。

先区分原生设置复现和统一协议重测：不能在没有对应配置/权重证据时把一次失败运行当成原方法性能。不能为了接近论文数字在test反复选阈值。

### Fa必须核对

本项目与本地原EV-UAV均为每个50ms窗口投影背景误检事件后统计一次8连通域，再除以总窗数×346×260；不是FP事件比例。保留原边界窗口/uint8行为，不在清理时改变历史指标。

11月10日核对的PACT[公开eval.py](https://raw.githubusercontent.com/fulongcai/PACT/main/utils/eval.py)把连通域统计放在逐误检事件循环内，会重复累计，与我们的每窗一次不同。只能确认该公开版本差异，不能推断论文用的版本或数值有错。用户复现时记录原生指标，另用统一评估器重算，若修复该处须保存diff并说明。

PointEvent所查看[公开测试入口](https://raw.githubusercontent.com/wzz-z/PointEvent/main/engines/test.py)没有输出Pd/Fa，需利用保存的预测统一重算或进一步确认作者评估实现。这里没有穷尽其全部历史代码。

## 不再维护的探索

R2实例/时间箱均衡、R3 TERM、Lovász、SAM及更早的运动/embedding路线已移除可运行实现。SAM本次Jaccard组test62.22优于同期基线58.21，但不及已验证主方案；不能写成“SAM普遍无效”。历史失败数值和证据路径保留在EXPERIMENT_RESULTS.md及原日志归档，旧文档快照保存在runs/preserved/pre_cleanup_docs_20261110.tar.gz。不将路径监督、UOT、SSM等未实现想法当作当前论文方法。
