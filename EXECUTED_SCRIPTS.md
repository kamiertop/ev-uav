# 本次会话执行脚本记录

| # | 脚本 | 内容 | 后端/种子/实验 | 输出日志 | 状态 |
|---|---|---|---|---|---|
| 1 | `scripts/run_s37_fixed50.sh` | 四组固定 50 轮对照 | auto，seed 37，a1/r1_event/r2_balanced/r3_term | `runs/logs/four_runs_s37_fixed50_auto_20261017_140342.log` | ✅ |
| 2 | `scripts/eval_s37_val.sh` | 四组 val 评估 + a1/r1 阈值敏感性 | thresh 0.9/0.7/0.95 | `runs/logs/val_eval_s37_20261017_163508.log` | ✅ |
| 3 | `scripts/run_multi_seed.sh` | a1/r1_event 多种子 | auto，seed 13/73 | `runs/logs/multi_seed_fixed50_auto_20261017_170700.log` | ✅（首启 OOM，重启后跑完） |
| 4 | `scripts/diag_a0_math.sh` | a0 math 诊断（验证后端触发 NaN） | math，seed 37，a0 | `runs/logs/a0_s37_math_diag.log` | ✅ |
| 5 | `scripts/run_baseline_seeds.sh` | a0/attention_fix 三种子 | auto，seed 13/37/73 | `runs/logs/baseline_seeds_20261017_201741.log` | ❌ 崩溃（a0 s37 epoch8 梯度 inf） |
| 6 | `scripts/run_all_math.sh` | 全量 math 12 组对照 | math，4 预设 × 3 种子 | `runs/logs/all_math_fixed50_20261019_104831.log` | ✅ |
| 7 | `scripts/sweep_thresholds_val.sh` | 四组 val 阈值扫描（含 r1 0.97/0.98 补点） | math，0.7/0.8/0.9/0.95 | `runs/logs/thresh_sweep_val_math_20261019_171351.log` | ✅ |
| 8 | `scripts/final_test.sh` | 冻结阈值测试集终评 | math，a1@0.8 / r1@0.95 / a0@0.8 | `runs/logs/final_test_math_20261019_182010.log` | ✅（测试集封存） |
| 9 | `scripts/run_lambda003.sh` | r1_event λ=0.03 扫描 | math，seed 13/37/73 | `runs/logs/lambda003_math_20261020_134423.log` | ✅ |

## 失败/废弃的尝试（供排查，不算正式结果）

- `scripts/run_multi_seed.sh` 首次启动（16:57）——外部进程挤占 GPU 0 约 7 GiB，epoch 0 OOM，残留已删
- `scripts/run_baseline_seeds.sh`（#5）——a0 在 auto 后端梯度爆炸，是关键证据（坐实历史 NaN 定位），不是普通失败
- `runs/logs/lambda003_math_20261019_183301.log`（8K）——λ=0.03 的 10-19 晚 OOM 假启动，10-20 重跑后才成功

## 配套文件

- 每个 `runs/train_*/` 下的 `config.json` + `metrics.jsonl`
- `runs/logs/gpu0_memory_*.csv`（5 份显存采样）

以上全部已包含在 `transfer_results.tar.gz` 里。

## 给分析 agent 的阅读顺序

1. `NEXT_STEPS.md`（结论与证据链）
2. #8 终评日志（`final_test_math_20261019_182010.log`）
3. #6 全量对照（`all_math_fixed50_20261019_104831.log`）
4. #7 阈值扫描（`thresh_sweep_val_math_20261019_171351.log`）
5. #5 崩溃日志（`baseline_seeds_20261017_201741.log`）理解 auto→math 的动机
