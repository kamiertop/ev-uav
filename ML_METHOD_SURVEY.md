# 从 ICLR / ICML 方法出发的 EV-UAV 改进调研

调研日期：2026-10-16。最新状态：已实现第一阶段 event / balanced / TERM 风险对照并通过 CPU 测试（最新数量见 NEXT_STEPS.md），尚无新 CUDA 训练结果。路径、UOT 与结构方案仍是待验证设计。接手入口见 [NEXT_STEPS.md](NEXT_STEPS.md)。

## 1. 推荐结论

保留已有 A1（sequence Attention + fixed GDSC + STC）作为工程起点。建议按以下顺序推进：

1. **首先验证按实例和时间箱组织的困难风险加权**：借鉴 ICLR 2021 TERM，用很小的改动检验“稀疏、短小、困难轨迹是否被事件平均损失掩盖”。这是投入最低的诊断性实验。
2. **以时间有向图上的可微路径学习作为主要研究候选**：借鉴 ICML 2018 Differentiable Dynamic Programming，用完整路径的结构监督替代仅平滑平均置信度。它与事件轨迹的时间方向、断裂和多目标混淆更直接相关。
3. **非平衡最优传输作为备选几何损失**：借鉴 ICML 2020 UOT，在事件密度变化、预测与标注质量不等时比较局部时空分布；不先做全事件 Sinkhorn。

TERM 的简单迁移本身不构成充分创新。更值得验证的论文问题是：**在不增加推理阶段跟踪器的条件下，能否通过困难轨迹风险与时间有向路径监督，减少事件分割中的连续漏检？** 两部分必须先单独成立，不能直接堆成完整模型后归因。

本轮做了针对性检索，包括 2024 年 Mamba-2、2026 年 OT 工作，但并非历届会议的穷尽综述。方法优先级来自与当前实现的契合度、可证伪性和计算成本，不以年份新旧排序。下文“文献方法”与“本项目设计”严格区分；没有文献证明这些迁移已能提升 EV-UAV。

## 2. 当前证据与真正需要检验的问题

### 2.1 有效结果及其边界

以 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) 为最新结果来源：A1 固定 50 轮、seed 37，测试 IoU=62.30%，seg_acc=69.14%，Pd=81.75%，coverage=74.91%，longest=62.33%。这是单次服务器结果，实际评估配置和权重身份仍待核实。

- 当前 seg_acc 是前景召回率；因此还有漏检空间，但仅凭汇总数值不能认定漏检主要发生于稀疏、快速或转向目标。
- 低 Fa 的口径尚未与论文对齐，不能由它推断可以随意提高召回，也不能把降阈值后的变化算作模型贡献。
- 历史失败运行曾捕获 Attention 反向 NaN。旧路线已删除，通用诊断开关保留；该记录不是当前风险损失的失败证据。
- 受控对照统一采用最新固定 50 轮协议，不能混用历史早停结果；最新接手顺序见 NEXT_STEPS.md。

### 2.2 当前实现的边界

- STC 按事件平均，局部权重被 detach；目标事件多的片段可能贡献更多梯度。
- 新风险损失按实例/时间箱组织 BCE，但对箱顺序不敏感，不直接约束空间连接、身份切换或真实时间连续性。
- 当前轨迹评价只排列 occupied bins，因此 longest 不等于真实时间中无间断的持续检测。
- 训练超过事件上限时对整个序列随机采样；是否损害稀疏轨迹需要采样前后统计，不假定所有序列都触发采样。

原论文 PDF 第 6 页 Eq. (2)–(3) 有 STC 权重指数 gamma，并写成邻域支持和的 sigmoid；当前代码使用去均值支持、detach，且没有该指数。A0 是当前公开代码行为基线，不能称逐公式严格复现。保留现实现以维持 A1 的比较基础。

原论文来源：本地 `Event_based_Tiny_Object_Detection_A_Benchmark_Dataset_and_Baseline.pdf` 第 6 页。近邻 SpTopoNet 的 EvTopoLoss 使用持久同调、Betti 与持久图距离，见本地 `Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection.pdf` 第 5 页。本文拟议路径和风险方案不复用这些组件，但新颖性仍需进一步排查。

## 3. 值得读的论文与取舍

以下链接以会议论文集、作者论文和作者代码为依据。部分 OpenReview 页面直接打开遇到验证页，因此交叉使用作者站点、PMLR、arXiv 和会议下载目录核对；不把仅有“under review”的稿件当作录用证明。

| 论文 | 会议 / 年份 | 可迁移的核心方法 | 本项目取舍 |
|---|---|---|---|
| [Tilted Empirical Risk Minimization](https://openreview.net/forum?id=K5YasWXZT3O)；[作者说明](https://blog.ml.cmu.edu/2021/04/02/term/) | ICLR 2021 | 用指数倾斜连续调节平均风险与困难样本风险 | **首个实验**：在轨迹时间箱上聚合，而不是对海量事件直接做最大损失 |
| [Differentiable Dynamic Programming for Structured Prediction and Attention](https://proceedings.mlr.press/v80/mensch18a.html) | ICML 2018 | 平滑 DP 的 max 运算，使路径分数和结构解可微 | **主研究候选**：构造时间有向、包含竞争背景候选的路径监督 |
| [Soft-DTW: a Differentiable Loss Function for Time-Series](https://proceedings.mlr.press/v70/cuturi17a.html) | ICML 2017 | 对动态时间规整进行平滑，允许序列对齐可微 | 辅助对照；现有标签有明确时间，不应无约束地允许时间扭曲 |
| [On Unbalanced Optimal Transport: An Analysis of Sinkhorn Algorithm](https://proceedings.mlr.press/v119/pham20a.html) | ICML 2020 | 放松质量守恒，对不等总质量测度求熵正则传输；提供算法复杂度分析 | 备选几何目标，适合研究密度变化，但成本和退化解需控制 |
| [Distributionally Robust Neural Networks](https://openreview.net/forum?id=ryxGuJrFvS)；[作者代码与论文信息](https://github.com/kohpangwei/group_DRO) | ICLR 2020 | 优化最差组风险，并研究正则化对组外泛化的重要性 | 当分组统计确证明显差距后，再比较 Group DRO；不先为少量序列建立大量小组 |
| [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html) | ICML 2018 | 按共享参数上的梯度大小、任务训练速度调整损失权重 | 多损失确实产生梯度尺度失衡时使用；不是 NaN 修复方案或独立主创新 |
| [Efficiently Modeling Long Sequences with Structured State Spaces](https://openreview.net/forum?id=uYLFoz1vlAC)；[Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933) | ICLR 2022 / 2023 | S4 / S5 的状态空间长序列建模 | 结构备选；先证明现有池化 Attention 的时序分辨率或开销是瓶颈 |
| [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://proceedings.mlr.press/v235/dao24a.html) | ICML 2024 | SSD 理论联系与 Mamba-2 | 可做池化 token 时序分支对照；不能搬用语言任务加速比来宣称本模型更快 |
| [HOTA: Hamiltonian framework for Optimal Transport Advection](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8911a007b983c6a4c33fcf03630fa107-Abstract-Conference.html) | ICLR 2026 | 通过 Hamilton–Jacobi–Bellman 与势函数求动态 OT | 本轮只核对摘要层面的相关性；其分布输运轨迹不等于物体检测轨迹，不列首选 |
| [Transport or Discard: Robust Unbalanced Optimal Transport for Cross-Domain Policy Adaptation](https://proceedings.mlr.press/v306/chen26gd.html) | ICML 2026 | 在跨域离线 RL 中用非平衡传输处理不适配转移 | 新近方法参考；任务和假设差异大，不直接迁移其策略学习目标 |

## 4. 第一优先实验：实例与时间箱上的 TERM

### 4.1 动机与定义

**文献方法**：TERM 的正倾斜提高高损失样本权重；倾斜趋近 0 时恢复均值，趋近正无穷时趋近最大值。它是已知的风险聚合机制，不是本项目的新理论。来源：[作者说明与原论文入口](https://blog.ml.cmu.edu/2021/04/02/term/)。

**本项目设计**：沿用 50 ms 时间箱，按 `(sample, instance_id, bin_id)` 分组。对有 GT 前景事件的箱定义

\[
\ell_{m,b}=\frac{1}{|I_{m,b}|}\sum_{i\in I_{m,b}}\operatorname{softplus}(-z_i),
\]

其中 z 是分割 logit。每箱先取均值，避免事件数量直接决定其权重。以下公式表示一个样本内部的风险，M 为该样本的有效实例数；实例内做倾斜，再对实例取均值：

\[
R_\beta=\frac{1}{M}\sum_{m=1}^{M}
\frac{1}{\beta}\left[
\operatorname{logsumexp}_{b\in B_m}(\beta\ell_{m,b})-\log |B_m|
\right],\qquad \beta>0.
\]

当前实现 `L = L_STC + lambda * R_beta`。batch 内先分别计算各样本风险，再对样本取均值；有效前景条件为 label>0.5 且 instance_id>0。无有效前景的样本风险为零，仍计入样本平均并记录数量。STC 保持全体背景和无实例 ID 前景的监督；新增项仍可能提高假阳性，必须一起评价。原始 logit 已通过模型辅助输出暴露，没有从 clamp 后的概率反算。

初筛建议 lambda=0.1，beta=1；对照 beta=0 使用解析均值分支。只有初筛成立，才按预先限定预算测试 lambda∈{0.03,0.1}、beta∈{0,1,2}。这些是待验证起点，不是文献在 EV-UAV 上的推荐参数。

### 4.2 可以精确说明什么

对固定实例及有限损失，有

\[
\partial R_{\beta,m}/\partial\ell_{m,b}
=\operatorname{softmax}_b(\beta\ell_{m,b}),\quad
\operatorname{mean}(\ell_m)\le R_{\beta,m}\le\max(\ell_m).
\]

这解释了它为何增加困难时间箱的梯度占比。**它不保证 IoU 提升，也不保证连续性**：把同一组箱损失的时间顺序打乱，TERM 值不变。若 TERM 已有收益而路径项没有收益，应把结论收敛到困难片段风险，不能声称学习了轨迹拓扑。

样本量很小的箱也可能是噪声。记录事件数分层结果、倾斜权重最大值和有效箱数量；先以有限 beta 限制极端聚焦，不删除所有稀疏箱，因为它们恰是研究对象。batch size=1 下这里估计的是样本内轨迹风险，再随训练对样本平均，**不是整个训练集的全局 TERM**。

### 4.3 必须区分的四种收益

1. A1 原始 STC。
2. A1 + 普通前景事件均值 BCE 辅助项：检查是否只是加强正样本。
3. A1 + 实例/时间箱均衡均值（beta=0）：检查是否只是改变采样单位。
4. A1 + 实例/时间箱倾斜（beta>0）：检验困难风险本身的贡献。

最终还应补一个相同主干上的类平衡 BCE/Focal 对照，避免把常规类别不平衡处理包装成轨迹创新。

## 5. 主研究候选：时间有向图上的可微路径学习

### 5.1 为什么比继续叠加局部平滑更有针对性

仅对时间箱聚合损失不能知道两箱之间是否接错目标。借鉴 [ICML 2018 可微 DP](https://proceedings.mlr.press/v80/mensch18a.html)，把训练时轨迹监督表达成路径集合上的概率学习。以下是拟议适配，不是该论文的原始 EV-UAV 方法。

### 5.2 最小实现设计

1. 用固定时间网格及空间小网格将事件聚合为候选节点，保留实际时间、坐标、特征与分割 logit。图结构的离散构建可以停止梯度，节点分数保留梯度。
2. 每个训练实例建立局部候选图，**同时包含目标事件、附近背景和其他目标候选**。GT 只定义训练的可接受路径集合，不作为推理输入。
3. 只连向更晚时间的节点，使用空间距离/时间差限制邻居；最初使用固定速度上限，不引入额外运动头。这样先检验路径目标，之后再测试学习到的运动代价。
4. 按真实 bin_id 处理时间。GT 本身无事件的时间箱是未观测，使用有限跨度的跳边或可见性状态；不能无条件标成漏检，也不能不记录时间跨度地压缩掉。
5. 训练用该辅助损失，推理仍输出原分割概率。第一版不添加追踪后处理，保持推理成本可比。

设节点分数为 a_j，边代价为 c_ij，路径分数

\[
S(\pi)=\sum_{j\in\pi} a_j-\sum_{(i,j)\in\pi}c_{ij}.
\]

边代价先用按真实时间差归一化的位移惩罚，并设置宽松门控；转向目标不能被强制直线化。固定起终有效时间范围，明确跳边惩罚，避免短路径因累计分数长度不同获得无关优势。

通过 log 域 DP 计算：

\[
A(j)=a_j/\tau+
\operatorname{logsumexp}_{i\to j}\{A(i)-c_{ij}/\tau\}.
\]

起点取源节点 A(source)=0，终点做同样归并。所有可行候选路径构成 Pi，GT 身份和可见性约束允许的路径构成 Pi_gt，得到

\[
L_{path}=\frac{\log Z(\Pi)-\log Z(\Pi_{gt})}{T_{obs}},
\quad Z(\Pi)=\sum_{\pi\in\Pi}\exp(S(\pi)/\tau).
\]

这是对“路径落在 GT 兼容集合内”的负对数概率，T_obs 为有效观测箱数。只要 Pi_gt 是非空子集，该损失非负。自动微分可沿两次 DP 回传到节点分数。有限 DAG 和 tau>0 保证该 log-partition 对有限分数可微；不保证整个神经网络优化凸或训练收敛。

### 5.3 会导致方案失败的细节

- **只构建正轨迹图**：若 Pi=Pi_gt，损失为零，无法抑制背景。必须有竞争路径。
- **没有 GT 可行路径**：空间门控、候选裁剪或真实断点可能导致 Z_gt=0。返回诊断并跳过该实例，记录跳过率；不能把无穷值简单 clamp 后当成有效监督。
- **训练只取预测 top-k**：早期预测差时会永久丢掉正确候选。先用与预测无关的网格候选；对候选预算记录 GT 可达率。GT 保底候选若使用，需明确属于训练监督并报告其影响。
- **全背景或重复选择容易节点**：路径项不能替代逐事件分割监督，路径中选中少量正确节点不代表整条目标事件均被恢复。
- **多目标交叉**：分别构建实例训练图不等价于推理时排他多目标分配，必须专门做交叉场景检验。
- **时序错误**：对不同样本不建边；显式按时间排序；不能依赖 sparse tensor 内部索引顺序。
- **数值问题**：使用 FP32 logsumexp，处理没有前驱的节点，避免全负无穷分支产生无效梯度；先验证小图枚举与 DP 一致。

候选预算可从每实例 T<=160、每箱 K<=32、每节点最多 8 条前驱开始（8 秒/50 ms 为 160 箱）。DP 本身的规模约 O(|E|)，但**建图与候选池化也必须计入总耗时**；直接全连接建图仍可能很贵。多实例依次计算或分块，限制每批辅助图预算，报告显存峰值。

### 5.4 必要对照与研究贡献边界

- 同样的候选、分箱和预算下，比局部相邻箱平滑、逐箱分类、非路径的局部几何约束。
- 去掉几何边代价；去掉时间门控；有/无跳边；固定与预测运动代价。
- TERM 单独、Path 单独、二者组合。组合只在单项有效后进行。
- 小型受控例子：相同漏检数量的集中断裂/分散漏检、两个目标交叉、连续背景干扰、真实无事件区间。
- 保留 STC / BCE 的负样本监督，检查是否把背景连成轨迹。

可研究的贡献是事件稀疏观测、时间方向、身份监督与局部竞争路径的具体建模及证据。logsumexp DP、CRF 式路径概率本身已存在，不应包装为新优化算法。正式投稿前仍需补充事件跟踪、可微数据关联和 track-before-detect 的近邻工作检索，本轮并未完成新颖性排查。

## 6. 备选：非平衡最优传输，而非全局点云匹配

[ICML 2020 UOT](https://proceedings.mlr.press/v119/pham20a.html)研究不同总质量测度之间的传输及 Sinkhorn 复杂度。它提供可借鉴的数学工具，不证明事件分割会受益。

拟在同一训练局部区域比较预测测度与 GT 测度：

\[
\mu_p=\sum_u a_u\delta_{q_u},\quad
\mu_y=\sum_v b_v\delta_{q_v},\quad
\min_{P\ge0}\langle C,P\rangle+\varepsilon\sum_{uv}P_{uv}(\log P_{uv}-1)
+\rho\,KL(P\mathbf1\|a)+\rho\,KL(P^T\mathbf1\|b).
\]

这里 a、b 是共享时空网格上的预测/GT 前景质量。先明确按事件数还是按网格占据统计，并使用共同尺度；**不要把双方各自归一化为总质量 1 后，还声称保留了前景总量差异**。坐标代价使用空间和时间各自的尺度，避免像素与毫秒直接相加。

与 SpTopoNet 的区别是：此处匹配事件的空间时间质量，不计算持久图或 Betti 数。UOT 本身也不保证有向连续路径，因此若主要错误是身份交换和断裂，它不是路径监督的替代品。

主要风险是过度销毁预测质量、忽略少数目标、把不同实例连接、对密度变化过敏。保留分割监督、限制时间跨度、按实例附近区域计算并包含背景；若需要防止总量塌缩，再单独消融质量约束。严格测试全零预测、空 GT、极低预测质量和极端距离。

每局部区域最多 128–256 个候选，FP32/log 域、有限 Sinkhorn 迭代，并监控残差。默认最多 700000 个事件若直接构造两两 float32 代价矩阵，单矩阵即约 1.96 TB（十进制），远超 24 GB 显存。低复杂度理论界不能消除不受控的候选数量。

只有“局部逐点/质心损失不能处理几何偏移和质量差异”的验证证据成立，才值得投入 UOT；必须与便宜的局部几何基线比较。

## 7. 为什么暂不先换 SSM 或做自动损失平衡

S4/S5 与 [Mamba-2](https://proceedings.mlr.press/v235/dao24a.html)提供长序列建模方法。但当前 Attention 已经过多次池化，不能将原始事件数量直接视为 Attention 的 token 数量。先记录 pa2/pa3/pa4 token 数、池化后的实际时间分辨率和耗时占比。

如果诊断显示远距离时间信息不足，可加入按空间区域组织、按时间排序的状态空间分支，保留位置编码、空箱 mask 和局部空间交互。目标跨区域移动需要邻域交换；不能把所有稀疏体素按存储顺序输入 Mamba。若用双向扫描，只能作为离线模型；在线因果设置要另行报告。与相同参数预算的时序卷积、GRU 或 Attention 对照，再决定结构是否有价值。

GradNorm 只在共享参数梯度均有限、且分项存在尺度失衡时考虑。先每隔若干步记录分割与辅助项的梯度范数及夹角；不能只比较原始 loss 数值。频繁求多项梯度会增加时间与显存，现有近满显存运行不宜直接加入。

Group DRO 更适合已有可靠分组且样本充足时使用。先按训练集确定的速度、前景密度、目标数阈值做少量分组分析；不要给每一条轨迹设置独立 adversarial group，也不要用测试结果定义困难组。若改变正则化或早停，应作为独立协议重新跑基线。

## 8. 实验安排与停止条件

### 8.1 先做验证集诊断

使用已经训练好的 A1，在验证集保存按序列/实例/时间箱的计数、召回、平均概率、速度和实际 bin_id；统计背景误检。先检查：

1. 错误是否集中在低事件密度箱、快速或转向轨迹、多目标交叉处？这些是待检验假设。
2. 是整条轨迹低置信度，还是局部连续断裂？TERM 与 Path 分别针对不同情况。
3. 低召回能否主要由阈值解释？在验证集做统一阈值扫描，报告同 Fa 或同 Pd 条件下的比较，避免仅以单阈值得分判断。
4. 有多少样本触发 700000 事件采样？采样前后 GT 时间箱保留率如何？
5. 当前 occupied-bin 轨迹指标与保留真实时间间隔的辅助指标是否结论一致？旧指标保留原名与版本；新定义单列，对所有方法重新计算。无事件箱不自动计为漏检，也不自动视为跨过后的连续检测，可按可观测片段分段报告。

### 8.2 分阶段最小矩阵

以下 R 编号是计划标识。当前 R0 对应 `--experiment a1`，R1–R3 分别对应已实现的 `r1_event`、`r2_balanced`、`r3_term`；R4–R7 尚未实现。

| 阶段 / ID | 结构 | 损失 / 变化 | 要回答的问题 |
|---|---|---|---|
| R0 | A1 | STC | 同协议锚点；旧运行仅在配置、权重、后端核实一致时复用 |
| R1 | A1 | STC + 前景事件均值辅助 BCE | 简单加强正样本是否足够 |
| R2 | A1 | STC + 实例/时间箱均衡均值 | 粒度均衡是否有效 |
| R3 | A1 | STC + 实例/时间箱 TERM | 风险倾斜是否超出 R2 |
| R4 | A1 | STC + 局部时间约束 | 以相同候选预算建立廉价结构对照 |
| R5 | A1 | STC + Path | 全路径是否超出 R4 |
| R6 | A1 | STC + TERM + Path | 只有 R3、R5 成立才检验组合 |
| R7，可选 | A1 | STC + 局部 UOT | 仅当错误分析支持分布几何路线时启动 |

先做 R0–R3；若只需风险均衡即可解决主要错误，不急于实现路径层。最终主方案补齐同协议 A0 / 纯 Attention 修正对照，以区分基线增强与新贡献。

统一 seed 37、50 epochs、关闭早停与训练后自动测试；保持学习率计划、优化器、事件上限、体素化和数据划分一致。先诊断并选定可稳定运行的 Attention 后端；若从 auto 改为 math，正式对照统一并重建相应 A1 锚点，不能直接沿用历史结果。新损失可以做短程可运行性检查，但若复现原 NaN，必须保留原 50 轮学习率计划。

先跑有限梯度与显存检查，再完整训练。筛选用验证集，最终候选与最强基线补 seed 13/37/73，报告均值和标准差；三个种子用于判断稳定性，不自动构成统计显著性证明。对序列指标可增加按序列配对 bootstrap，不能把同一序列内高度相关事件当独立样本。

### 8.3 预先约定的决策标准

- R3 若不超过 R2，说明目前没有困难风险倾斜的额外证据，保留更简单的 R2。
- R5 若不超过相同候选/分箱的 R4，不保留昂贵路径目标。
- 以验证 IoU 绝对提升约 1 个百分点作为继续投入的参考门槛，并结合种子方差；这是工程预算标准，不是理论阈值。
- 若 IoU 基本持平而实际时间上的连续检测明显改善，可以转为轨迹完整性贡献，但需预先固定指标、Fa 容忍范围和主要终点。
- 新增训练成本初步预算不超过 A1 的约 30%；超过时先缩减图预算，不能掩盖开销。最终报告实际单轮时间、显存、推理延迟与参数量。
- Fa / Pd、coverage / longest 联合看；不单独优化 fragmentation，不以测试集反复筛选方案。

## 9. 当前实现与后续接入位置

| 文件 | 计划变更 |
|---|---|
| `model/evspsegnet.py` | 已暴露分割原始 logits，保留现有概率接口；第一阶段不改主干 |
| `utils/trajectory_risk.py` | 已实现实例/时间箱聚合、TERM、空组与极端值处理 |
| 新的 `utils/path_consistency.py` | 候选池化、时间有向图、GT 可行路径检查、两次 log-partition DP |
| `dataset/ev_uav.py` | 如路径训练需要，提供采样前的可见箱元数据；不默认修改采样分布 |
| `train.py`、`utils/args.py` | 独立开关、配置落盘、分项与诊断日志；保持既有实验可复现 |
| `utils/eval.py` 或独立分析脚本 | 保留旧指标；新增真实时间间隔及分组统计，所有模型用相同规则 |

已完成的风险损失 CPU 验证包括 beta→0 与均值一致、箱内复制事件不改变该箱均值、样本与实例隔离。路径方案未来需验证小图枚举与 DP 的值/梯度一致、无 GT 路径处理、竞争背景存在、缺失时间箱语义及多目标交叉。两条路线均需 CUDA 验证后才进入完整训练。

第一阶段风险损失现已实现；第 9 节中的 logits、trajectory_risk 与训练参数接入已经完成，其余仍为后续计划。具体已实现语义、测试证据和未完成项见 NEXT_STEPS.md；没有新方法精度收益的证据。
