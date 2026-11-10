# 当前训练与验证命令

更新：2026-10-20。当前优先 IoU 对齐对照，旧对照命令留作复现。接手顺序见 [NEXT_STEPS.md](NEXT_STEPS.md)。以下命令在项目根目录执行。

## 当前优先：A1 / Lovász / soft Jaccard（2026-10-20）

新增 `a1_lovasz`、`a1_jaccard`，均为 A1 主干 + STC + 0.1×IoU 辅助项，**不叠加 R1**、不增加可训练参数。数学实现按事件计算、按样本平均；训练时仍受 700000 事件采样限制，逐样本损失不等同于全验证集累计 IoU。Lovász 为 binary hinge，soft Jaccard 为硬标签 soft IoU；空前景样本的 Jaccard 分支回退到背景 BCE 防止梯度消失。两个分支用原始 logits，半精度输入升 FP32。iou_weight=0 完全跳过辅助计算。

先把当前工作区的已改文件和新增文件同步到服务器（不能只传 tracked diff）；其中新文件包括 `utils/iou_loss.py`、`tests/test_iou_loss.py`、`scripts/run_iou_ablation.py`、`scripts/run_iou_ablation.sh`。服务器保持原 HAIS_OP 环境，确认所选 GPU 空闲。全部实验在项目根目录执行：

```bash
bash scripts/run_iou_ablation.sh --gpu 0 --data-dir ../ev-uav-data
```

默认：CPU 测试 → 两个候选各 1 轮 CUDA smoke → seed 37/13/73 × A1/Lovász/Jaccard，共 **9 组正式训练**。所有正式训练固定 math、50轮、batch=1、700000事件、Adam、0.01→0.001线性学习率、每轮val@0.9选择 best_iou、关闭早停与自动test。每组完成后对其唯一匹配的 checkpoint 在 val 扫描 0.7/0.8/0.85/0.9/0.95，共45次评估。每种方法最终跨种子统一选阈值，不逐种子挑峰值。smoke不能替代50轮协议验证。为避免复用错权重，脚本从本次训练日志读取精确run目录，不按“最新目录”盲选。

默认重新训练 A1 以提供相同代码/环境的锚点。按过去每组约31分钟估计，9组训练约4.7小时，另加smoke和评估；新增损失的实际耗时未知，以日志为准。脚本要求 CUDA/HAIS_OP 可用、GPU空闲显存至少22GiB、train/val数据存在；这不能保证后续显存不被其他任务挤占。任一训练/评估失败立即停止，已生成结果保留，不自动减事件上限或改变协议。

可选先只跑 seed37（3组正式训练），此后补另外两种子会再次做smoke，但不会覆盖第一批：

```bash
bash scripts/run_iou_ablation.sh --gpu 0 --data-dir ../ev-uav-data --seeds 37
bash scripts/run_iou_ablation.sh --gpu 0 --data-dir ../ev-uav-data --seeds 13 73
```

只预览正式训练计划（不加载数据/启动CUDA）：

```bash
bash scripts/run_iou_ablation.sh --dry-run --seeds 37
```

每个训练与评估分别写日志，脚本控制台打印路径；中途可另开终端 `tail -f` 对应训练日志查看进展。训练指标新增 `iou_surrogate` / `iou_weighted`、总grad_norm、每轮耗时与PyTorch峰值allocated/reserved（不是nvidia-smi整卡占用）；每轮验证新增 TP/FP/FN/precision/FP每GT。val JSON 含全部评估flags、原始全局计数、逐序列计数/IoU，保留原有IoU/Pd/Fa定义。

成功后回传控制台最后输出的 `runs/logs/iou_ablation_时间_标识.tar.gz`。它包含环境、执行命令清单、全部训练/验证日志、45份验证JSON及9组config.json/metrics.jsonl；**不含权重**，请把权重留在服务器。若分两批则两个包都回传。失败时直接回传该批 `runs/logs/iou_ablation_时间_标识/` 和已产生run的config/metrics，不要覆盖重试日志。

本地40项CPU测试通过，但无CUDA，新增两种损失尚未完成服务器smoke或正式训练；不声称IoU提升。测试集保持封存，脚本只评val。

## 0. 环境检查

```bash
uv sync
nvidia-smi
```

若缺少 HAIS_OP，先运行 `uv run python scripts/build_ext.py` 编译；环境就绪后再运行 `uv run python -m unittest discover -s tests`。全套测试会导入数据和模型模块，因此全新环境不能跳过扩展准备。

本轮实验固定 50 轮，关闭早停与自动测试，所有组使用相同采样、优化器、阈值和后端。**代码默认 early_stopping_patience=10；预设只选择模型与损失，不替你关闭早停。** 正式命令必须保留 `--early_stopping_patience 0 --no-test_after_train`。

## 1. A1 与风险损失对照

这四组不改变主干或参数量，只比较辅助监督。新功能尚未完成 CUDA 训练验证。

| 预设 | 含义 |
|---|---|
| `a1` | R0：原 A1 + STC |
| `r1_event` | R1：STC + 样本内前景事件均值 BCE |
| `r2_balanced` | R2：STC + 实例/时间箱均衡 BCE |
| `r3_term` | R3：STC + 实例/时间箱 TERM |

先做一轮可运行性检查；该检查不作为性能结果，也不能替代原 50 轮学习率计划的 NaN 复现：

```bash
uv run python train.py --experiment r3_term \
  --data_dir ../ev-uav-data --gpu 0 --train_workers 0 \
  --run_name smoke_r3_term --seed 37 --epochs 1 \
  --attention_backend auto --diagnose_numerics \
  --early_stopping_patience 0 --no-test_after_train
```

通过后顺序运行四组；任何一组失败即停止循环，先检查日志：

```bash
for experiment in a1 r1_event r2_balanced r3_term; do
  uv run python train.py --experiment "$experiment" \
    --data_dir ../ev-uav-data --gpu 0 --train_workers 0 \
    --run_name "${experiment}_s37_fixed50_auto" \
    --seed 37 --epochs 50 --batch_size 1 --max_events_num 700000 \
    --optim Adam --lr 0.01 --final_lr 0.001 --scheduler linear \
    --attention_backend auto --risk_weight 0.1 --risk_beta 1 --risk_bin_ms 50 \
    --early_stopping_patience 0 --no-test_after_train || break
done
```

`../ev-uav-data` 改为服务器实际路径；本地已知路径为 `/data/ev-uav`。auto 与历史 A1 行为一致；如诊断确认需要 math，四组统一改为 math、修改 run_name 并重新建立 A1 对照，不能将历史 auto 结果作为同协议基线。

新参数：`--loss stc_risk`、`--risk_mode event/balanced/term`、`--risk_weight`（默认 0.1）、`--risk_beta`（默认 1）、`--risk_bin_ms`（默认 50）。TERM 的 beta=0 等于均衡均值；risk_weight=0 完全跳过辅助损失。risk_bin_ms 只改变训练分组，不改变评价时间箱。三个预设均使用固定 GDSC，不含额外辅助头；旧运动模块已删除。

日志新增 `seg`、`risk`、`risk_weighted`、`risk_tracks`、`risk_bins`、`risk_foreground_events`、`risk_unassigned_foreground`、`risk_empty_samples`、`risk_max_bin_weight`、`risk_effective_bins`。后两项是实例内箱权重统计；event 模式下记录的是用于诊断的均匀箱权重，并非实际事件损失的权重。

训练后先汇总验证结果：

```bash
uv run python scripts/summarize_runs.py
```

需要 Pd/Fa 时，新入口可评估验证集。替换 checkpoint 路径，并使用其 config.json 中的预设、后端、宽度和阈值：

```bash
uv run python test.py --split val --experiment r3_term \
  --data_dir ../ev-uav-data --gpu 0 --train_workers 0 \
  --attention_backend auto --prediction_thresh 0.9 \
  --model_path runs/train_REPLACE_WITH_ACTUAL_RUN/checkpoints/best_iou_seed37.pt
```

`test.py` 的默认 split 仍为 test，因此探索时务必显式写 `--split val`。配置不会从 checkpoint 自动恢复。验证集模型选择完成后，对最终方案与基线补 seed 13/73；冻结后才使用 `--split test`。风险参数暂用显式单次命令调整；旧网格脚本已删除。

## 2. 原始与 Attention 修正对照

`--experiment a0` 保留原 Attention 行为；`--experiment attention_fix` 仅修正注意力序列组织；`--experiment a1` 含跨 patch 注意力、位置编码和 LayerNorm。三者均使用固定 GDSC + STC。显式参数覆盖预设。默认不指定预设时也使用 A1 结构及 STC。

现有 A1 权重名称和形状保持不变；评估旧 A1 权重时，删除历史命令中的 `--motion_gd none`，其余参数按 config.json 对齐。旧运动/辅助头模型的权重不再支持，不应以非严格加载绕过。

## 3. 数值诊断与实验边界

若出现 Attention 反向 NaN，在同一 50 轮学习率计划下加 `--diagnose_numerics`，需要隔离融合 SDPA 后端时加 `--attention_backend math`。切换后端不能视为修复已获证实，且正式对照应统一后端。单轮 smoke 只验证可运行性，不适合复现依赖原学习率计划的异常。

收集 config.json、metrics.jsonl、完整评估命令、GPU、耗时与显存。当前轨迹指标仅统计有标注事件的时间箱，不等于真实时间连续检测时长。测试集只在模型和阈值冻结后评估。

## IoU runner 的 HAIS_OP 导入排查

若旧版脚本在预检阶段报 `ModuleNotFoundError: HAIS_OP`，先同步最新版 `scripts/run_iou_ablation.py`，它已将项目根目录加入模块搜索路径；单纯chdir不足以做到这一点。可在服务器项目根目录运行 `uv run python -c 'import torch, HAIS_OP; print(HAIS_OP.__file__)'` 验证；若此命令也报告缺少HAIS_OP，再运行 `uv run python scripts/build_ext.py` 在服务器构建，不复制其他机器的编译扩展。更新后原实验命令不变。

## 本轮 IoU 候选测试（用户已决定不先补 R1）

同步新增 `scripts/test_iou_ablation.py` 到训练服务器，在项目根目录运行：

```bash
uv run python scripts/test_iou_ablation.py --batch 20261020_163642_2d5b7c --gpu 0 --data-dir ../ev-uav-data
```

只评本轮A1/Jaccard各seed13/37/73，合计6次test、不训练。预先按val跨seed均值选择并冻结A1@0.85、Jaccard@0.9，不能在test扫阈值。脚本精确绑定批次，启动前核验六个配置、50轮完成记录及全部best_iou权重，缺少任何一项即报错；不使用旧final_test.sh的历史目录匹配。评估flags从各训练config恢复必要模型/指标参数，保存冻结计划、训练config、6份test JSON与日志，最后生成 `runs/logs/iou_test_*.tar.gz` 回传。

本地仅用解压config/metrics dry-run验证6项绑定与参数生成、编译检查通过，无本地权重/CUDA，未执行test。已有测试集曾使用，本次属于用户授权的后续迭代终评，应在实验记录中透明披露。

## R1 改进：IoU 对齐与困难背景对照（2026-10-20）

保留原R1（risk_weight=.1），新增三个预设，无新模型参数：

| 预设 | 训练目标 |
|---|---|
| r1_event | STC + .1×原前景事件BCE |
| r1_jaccard | R1 + .1×soft Jaccard |
| r1_hardneg | R1 + .03×困难背景BCE |
| r1_precision | R1 + .1×soft Jaccard + .03×困难背景BCE |

困难背景在每个样本内从GT背景事件选择最高logit的k个，k=min(背景数,max(256,ceil(3×全部前景事件数)))；选中事件用稳定softplus(logit)均值监督，不能由模型预测或测试场景身份定义GT负例。无背景样本该项为可微零但仍计入样本平均；无前景样本选最难的最多256个背景。STC对全部事件的监督不变，原R1仍仅label>.5且instance_id>0的前景。背景选择与辅助监督仅训练使用，不增加推理步骤。hardneg_weight=0/iou_weight=0分别跳过分支；两项均0时精确恢复R1。额外项权重是工程初始值，不是已验证最优值；不恢复旧focal/embedding/运动模块，不宣称该常规组合为独立新颖算法。

先同步本次全部已改源码和新增文件到服务器（尤其 utils/r1_precision.py、utils/args.py、train.py、scripts/run_iou_ablation.py、scripts/run_r1_precision.sh、tests/test_r1_precision.py、tests/test_experiment_config.py）。项目根目录执行：

```bash
bash scripts/run_r1_precision.sh --gpu 0 --data-dir ../ev-uav-data
```

默认只seed37：48项CPU测试→3个新增组各1轮CUDA smoke→上述4组各50轮math正式训练→每组best_iou权重在val评估 .8/.85/.9/.95/.97/.98（24次评估）。50轮训练依旧batch1/700000事件/Adam.01→.001线性/val@.9选best，禁早停与自动test；R1基线重跑以避免混用历史漂移结果。原run_iou_ablation.sh默认行为仍为之前3方法×3种子，不要混淆。

按当前每组31–32分钟估计本轮约2–2.5小时，硬负例实际GPU开销待验证。GPU可用显存预检22GiB、HAIS_OP路径初始化沿用已修复入口、任一步失败立即停止。控制台打印训练日志路径，进度在相应日志中查看。

只预览（不启动GPU）：

```bash
bash scripts/run_r1_precision.sh --dry-run
```

若要三种子一起执行，可显式 `--seeds 37 13 73`（12组，约6.5–7.5小时起）；默认先单seed不以波动否决候选，也不能声称单seed已证明泛化。`--weight 0.03`仅改变启用Jaccard分支的权重、原R1仍.1/困难背景仍.03；本轮不默认扫描。

结束后回传 `runs/logs/r1_precision_时间_标识.tar.gz`，服务器保留权重。本轮额外打包训练时源码快照/sha256、train/val文件名及大小和val文件SHA256，用于排查历史基线漂移；不读取test数据。训练config/metrics、各项损失、计数、每轮耗时和显存、val逐序列JSON/命令清单均入包，模型权重不入包。

判定预先看同阈值val@.9 IoU/precision/seg_acc/Pd/Fa以及全局与逐序列计数；另看跨seed统一阈值扫描，不只挑最高单次，不以单seed波动拒绝。检验R1_jaccard相对R1的收益、hardneg单独作用、组合是否比两单项更好。IoU仅靠召回大跌换取时不称四指标全面改进。只用val选方案，不因已查看的test_008对test做针对调参；后续投稿需披露test历史使用及最好独立数据补证。

本地48项CPU测试通过，runner模拟编排验证通过（3smoke/4正式/24val、分支关闭权重恢复及打包），CLI/编译/shell检查通过。本地CUDA不可用，**未运行新方案CUDA训练，不承诺IoU收益**。

## R1批次中途显存被占用（2026-10-21）

r1_precision_20261020_224425_6e80b8在r1_jaccard_s37 epoch0/batch40 OOM：其他进程占约6.93GiB，训练前向仅余24.44MiB。无需据此改变损失/事件上限，也不应删除或终止不属于自己的进程。先在服务器看nvidia-smi并协调独占空闲GPU。

同步最新版scripts/run_iou_ablation.py后，只重跑失败及未开始的三组：

```bash
nvidia-smi
bash scripts/run_r1_precision.sh --gpu 0 --data-dir ../ev-uav-data --experiments r1_jaccard r1_hardneg r1_precision
```

这是新批次、三组从头50轮训练，不是从失败batch恢复；已完成原r1_event不重跑。新脚本每次启动训练/评估子进程前重查22GiB空闲显存，但不能防止他人在训练中途占卡。原批次 `runs/logs/r1_precision_20261020_224425_6e80b8/` 及原r1_event对应run目录不要删，之后需与新结果一并提供以保留基线。若GPU0忙，改为经nvidia-smi确认空闲的GPU编号。新测试仅CPU通过，无新CUDA结果。

## 当前下一轮：R1较好方案 + SAM（2026-11-03）

保留此前归档，不覆盖原模型/权重；在R1+Jaccard和R1_precision两个较好方案上分别对照普通Adam与SAM+Adam。SAM只改变训练过程，没有新损失、模型头或推理步骤。所有新选项默认关闭：`--sam_rho 0`保持旧训练，`.05`启用两遍前后向、一次Adam步进；`--sam_start_epoch 0`默认全程启用，参数从0计。详细文献和方案在ML_METHOD_SURVEY.md第13节。

先将本次全部改动/新增文件同步到服务器，包含train.py、utils/args.py、utils/sharpness.py、scripts/run_iou_ablation.py、scripts/run_r1_sam.sh及相应tests。项目根目录执行：

```bash
bash scripts/run_r1_sam.sh --gpu 0 --data-dir ../ev-uav-data
```

默认：seed37，2个SAM单轮smoke → R1_jaccard(rho0/.05)和R1_precision(rho0/.05)，共4组正式50轮 → 每组val扫描.8/.85/.9/.95/.97/.98，共24次评估。其他协议与前轮一致：math、batch1、700000事件、Adam .01→.001线性、risk=.1、iou=.1、组合hardneg=.03、禁止早停和自动test，每轮val@.9选best_iou。控制组也重跑，不从旧best权重微调。

按过去31–32分钟/普通组及SAM两倍前后向开销估计，约3–4小时，峰值显存和实际耗时需看服务器；一张空闲24GiB或更大GPU顺序运行。每个子进程前复查22GiB空闲显存，但不能防止运行中被其他任务挤占。任一错误即停并打印真实错误，不自动降低事件数、切后端或使用非严格权重加载。

如果先只做较均衡的组合方案（2组，预计约1.5–2小时）：

```bash
bash scripts/run_r1_sam.sh --gpu 0 --data-dir ../ev-uav-data --experiments r1_precision
```

只预览命令、不训练：

```bash
bash scripts/run_r1_sam.sh --dry-run
```

半径参数可用 `--sam-rhos 0 0.03`替换默认0/.05；本轮不要同时扩展多个半径和种子。后期启用可显式 `--sam-start-epoch 25`，但非默认，且best_iou可能来自启用前；final metrics的best_iou_sam_active=false时不能归因为SAM。smoke会强制start=0以测试SAM分支。若用rho>0的训练单命令，start必须小于epochs。

结果按variant分开命名，`sam0.05_e0`标记不会覆盖rho0基线。配置、源码快照/hash、数据清单、日志/指标、训练完整目录身份、冻结命令计划、逐序列val JSON自动入包。训练指标增加sam_perturbed_loss（扰动位置loss）、sam_first_grad_norm、sam_perturbation_norm、sam_active；原loss和seg/risk/iou/hardneg分项仍是第一遍未扰动目标，用于横向观察，不能与第二遍loss混称。

跑完回传 `runs/logs/r1_sam_时间_标识.tar.gz`；权重留服务器，暂不跑test。我们先比较固定阈值.9和同Fa预算的val表现，再决定保留哪个variant、是否补种子/有限调参或终评。现有test曾被多轮查看，后续论文需透明披露。

本地仅60项CPU测试/流程模拟/编译/dry-run通过，没有执行实际训练或CUDA验证；用户在独立服务器运行后回传。旧run_r1_precision.sh和run_iou_ablation.sh默认计划保持原样，不自动启用SAM。

## SAM本批次正式测试（2026-11-10）

用户决定测试SAM。同步 `scripts/test_r1_sam.py` 和 `scripts/test_r1_sam.sh` 到训练服务器同一项目，保留原runs目录与四组checkpoint，在项目根运行：

```bash
bash scripts/test_r1_sam.sh --gpu 0 --data-dir ../ev-uav-data
```

固定批次20261110_121715_c91ef8、seed37、best_iou权重、math，四组（Jaccard及组合，各有/无SAM）统一test阈值.9；该阈值来自本批val峰值，不扫test阈值。不重训，不依赖旧10月21日权重。启动前校验所有config、训练完成状态与权重存在，缺任一个即停止；可加--dry-run预览（不检查权重、不访问数据）。如runs不在默认位置，用--runs-root PATH。

保存四指标/precision/TP/FP/FN的summary.csv、完整逐序列JSON、日志、冻结计划、checkpoint SHA256与训练config/metrics，成功或测试中途失败均打包已有产物。回传 `runs/logs/r1_sam_test_*.tar.gz`；包不含权重。脚本已完成本地真实配置dry-run、模拟四次测试和打包、错误rho拒绝、编译和shell检查，未在本地启动实际推理。服务器实际耗时尚未测定。
