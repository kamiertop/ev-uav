[English](README.md) | 简体中文


## 数据集

- [百度网盘](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)
- [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link)

## 环境

1. [uv](https://github.com/astral-sh/uv)
2. NVIDIA GPU(4090)
3. Ubuntu 22.04
4. [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)
5. GCC(g++ 11-13)
    - gcc --version
    - g++ --version
6. libsparsehash-dev(`apt install libsparsehash-dev`)

## 安装

1. 克隆仓库

```bash
# 配置 ssh key 后使用 SSH 克隆
git clone git@github.com:kamiertop/ev-uav.git

# 或者使用 HTTPS 克隆
git clone https://github.com/kamiertop/ev-uav.git
```

2. 安装依赖

```bash
cd ev-uav

uv sync
```

3. 编译 C++/CUDA 算子（在项目根目录生成 `HAIS_OP.so`）

```bash
# 安装cudatoolkit
# 检查 cuda toolkit 版本
nvcc --version
# 安装google/dense_hash_map
sudo apt install libsparsehash-dev
# 检查gcc版本
gcc --version
# 检查g++版本
g++ --version
# 编译HAIS_OP.so
uv run scripts/build_ext.py
```

常用选项：

```bash
uv run scripts/build_ext.py --gcc 13          # 系统 GCC 过新时，指定 GCC 13
uv run scripts/build_ext.py --cuda 12.6       # 指定 CUDA 版本，也可直接传路径
uv run scripts/build_ext.py --clean --gcc 13  # 清理旧构建产物后重新编译
```

# 🎯 运行代码

**1) 训练**：数据集路径通过 `--data_dir` 指定（默认 `/data/ev-uav`），完整参数见 `utils/args.py`

```bash
uv run train.py --data_dir /path/to/EV-UAV --gpu 0
```

**2) 测试**

```bash
uv run python test.py --model_path runs/train_xxx/checkpoints/best_iou_seed37.pt
```


## 致谢

本代码基于 [HAIS](https://github.com/hustvl/HAIS) 和 [spconv](https://github.com/traveller59/spconv)。

## 当前研究入口

主方案`r1_precision`已完成test：IoU73.22%、ACC77.69%、Pd85.77%、Fa9.78e-6（seed37、阈值.85，非多种子均值）。完整结果与权重身份见[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)及[experiments/verified_r1.json](experiments/verified_r1.json)。

接手先读[NEXT_STEPS.md](NEXT_STEPS.md)，服务器命令见[TRAINING_COMMANDS.md](TRAINING_COMMANDS.md)。只维护A0/Attention修正/A1/A1+Jaccard与四组R1消融；旧SAM/R2/R3/Lovász脚本已移除。下一步由用户独立复现PACT，作者报告与本地复现分开记录。
