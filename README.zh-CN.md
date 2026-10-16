[English](README.md) | 简体中文

# 🚁 基于事件相机的微小目标检测：基准数据集与基线

<p align="center">
<img src="imgs/logo.png"  width='800' />
 </a>
</p>

论文 "Event-based Tiny Object Detection: A Benchmark Dataset and Baseline"（ICCV 2025）的官方实现。[论文链接](https://arxiv.org/abs/2506.23575)

## 🌟 摘要

反无人机任务中的小目标检测（SOD）由于 UAV 目标尺寸极小、背景复杂，一直是极具挑战性的问题。传统帧相机受限于帧率低、动态范围有限和数据冗余，难以在复杂环境中检测小目标。事件相机凭借微秒级时间分辨率和高动态范围，为 SOD 提供了更有效的解决方案。然而，现有的基于事件的目标检测数据集规模有限、目标尺寸偏大、缺乏多样化背景，不适合作为 SOD 基准。本文提出基于事件的小目标检测（EVSOD）数据集 **EV-UAV**——首个面向反无人机任务的大规模、高多样性基准，包含 147 个序列、超过 **230 万条事件级标注**，目标**极其微小**（平均仅 6.8 × 5.4 像素），覆盖城市杂波、极端光照等**多样化场景**。此外，基于"小运动目标在时空事件点云中形成连续曲线"这一观察，我们提出了基于事件的稀疏分割网络（EV-SpSegNet）——一种在点云空间进行事件分割的新基线，并设计了利用运动连续性引导网络保留目标事件的时空相关性（STC）损失。在 EV-UAV 数据集上的大量实验证明了我们方法的优越性，为未来的 EVSOD 研究提供了基准。

---

## 📊 EV-UAV 数据集

- ### 事件相机与 RGB 相机对比

<p align="center">
<img src="imgs/left_top.jpg"  width='500' />
 </a>
</p>

RGB 相机只能在正常光照下拍摄目标，而事件相机可以在各种极端光照条件下拍摄目标，并且能够捕捉小目标的连续运动轨迹（图中红色曲线）。

- ### EV-UAV 与其他基于事件的目标检测数据集对比

<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">数据集</th>
    <th class="tg-c3ow" rowspan="2">UAV 平均尺度</th>
    <th class="tg-c3ow" rowspan="2">标注类型</th>
    <th class="tg-c3ow" rowspan="2">UAV 序列占比</th>
    <th class="tg-c3ow" rowspan="2">以 UAV 为中心</th>
    <th class="tg-c3ow" colspan="3">光照条件</th>
    <th class="tg-0pky" colspan="2">目标</th>
    <th class="tg-0pky" rowspan="2">年份</th>
  </tr>
  <tr>
    <th class="tg-0pky">BL</th>
    <th class="tg-0pky">NL</th>
    <th class="tg-0pky">LL</th>
    <th class="tg-0pky">MS</th>
    <th class="tg-0pky">MT</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark">VisEvent</a></td>
    <td class="tg-0pky">84×66 像素</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">15.97</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">2023</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/Event-AHU/EventVOT_Benchmark">EventVOT</a></td>
    <td class="tg-0pky">129×100 像素</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">8.41</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/Event-AHU/OpenEvDET">EvDET200K</a></td>
    <td class="tg-0pky">68×45 像素</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">3.57</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://zenodo.org/records/10281437">F-UAV-D</a></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">100</td>
    <td class="tg-0pky"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky"><a href="https://github.com/MagriniGabriele/NeRDD">NeRDD</a></td>
    <td class="tg-0pky">55×31 像素</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">100</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">2024</td>
  </tr>
  <tr>
    <td class="tg-0pky">EV-UAV</td>
    <td class="tg-0pky">6.8×5.4 像素</td>
    <td class="tg-0pky">Seg</td>
    <td class="tg-0pky">100</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-0pky">2025</td>
  </tr>
</tbody></table>

目前，基于事件的目标检测数据集主要聚焦于自动驾驶和通用目标检测，专门为 UAV 检测设计的数据集鲜有关注。我们对现有数据集进行了全面总结，凸显了基于事件的 UAV 目标检测数据集的稀缺性。

- ### 基准特征与统计

<img src="imgs/datasets.jpg" style="zoom: 25%;" />

EV-UAV 包含 147 个带有**事件级标注**的事件序列，覆盖高亮度、低照度等**挑战性场景**，目标尺寸平均仅为**现有数据集的 1/50**。

---

## 📂 EV-UAV 目录结构

数据集的文件结构如下：

```
EV-UAV/
├── test/
│   ├── test_000.npz
│   ├── test_001.npz
│   ├── .....
├── train
├── val

```

---

## 📝 数据格式

事件数据以 `.npz` 格式存储，其中包含三个文件（即 `evs_norm`、`ev_loc` 和 `ev`）。

**`ev`** 为原始事件数据。

- **x、y：** 事件的像素坐标。
- **t：** 事件发生的时间戳（毫秒）。
- **p：** 亮度变化的极性（1 或 0）。
- **label：** 是否为目标事件（0 或 1）。
- **name：** 目标的身份编号。

示例：

```
x    y   t  p label name
100 200  1            1        0    0
128 258  4000        0        1    5
```

**`evs_norm`** 为归一化后的事件数据。

示例：

```
x    y   t  p label name
0.289 0.769  0            1        0    0
0.369 0.992  0.5         0        1    5
```

**`ev_loc`** 为事件在点云空间中的坐标。

示例：

```
x    y   t
100 200  1
128 258  4000
```

## ⬇️ 数据集下载

EV-UAV 数据集可从 [百度网盘](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)（提取码：sbr2）或 [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link) 下载。

---

# :triangular_flag_on_post: 基线方法

利用运动目标在事件数据中的时空相关性特性，我们提出了 EV-SpSegNet——一种直接面向稀疏事件点云的分割网络，并设计了通过评估局部时空一致性来优化目标事件提取的时空相关性损失函数。

### 基于事件的稀疏分割网络

基于事件的稀疏分割网络（EV-SpSegNet）采用 U 形编码器-解码器结构，集成三个关键组件：用于多尺度时间特征提取的 GDSCA 模块（Grouped Dilated Sparse Convolution，分组空洞稀疏卷积）、用于特征融合的 Sp-SE 模块，以及用于体素下采样和全局上下文建模的 Patch Attention 块。

<img src="imgs/framework.png" width='900' />

- ### STC 损失

我们提出了时空相关性损失，鼓励网络保留具有高时空相关性的事件，同时丢弃孤立的噪声。

<p align="center">
<img src="imgs/stcloss1.png"  width='300' />
 </a>
</p>
<p align="center">
<img src="imgs/stcloss2.png"  width='300' />
 </a>
</p>

# 🧰 环境要求

本项目的 Python 依赖与虚拟环境由 [uv](https://docs.astral.sh/uv/) 管理。编译 CUDA 扩展前，请确认下表中的工具链版本：

| 依赖                   | 版本要求                  | 说明                                                                                                                                                   |
| ---------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Linux + NVIDIA GPU     | —                         | 编译 `HAIS_OP` CUDA 扩展仅支持 Linux                                                                                                                   |
| Python                 | `>=3.10, <3.14`           | 无需手动安装，由 uv 自动管理                                                                                                                           |
| uv                     | 任意较新版本              | 依赖与虚拟环境管理                                                                                                                                     |
| CUDA Toolkit（`nvcc`） | **12.x，推荐 12.6**       | 仅在编译 `HAIS_OP` 扩展时需要。PyTorch 与 spconv 均为 cu126 构建，`nvcc` 主版本必须为 **12**，推荐直接安装 12.6 保持一致                               |
| GCC / g++              | **<= 13（推荐 11 – 13）** | CUDA 12.x 官方支持的编译器上限。若系统默认 g++ 版本过新，编译脚本会自动在 `/opt`、`/usr/local`、`/usr` 下查找 GCC 13/12/11，也可用 `--gcc 13` 显式指定 |
| NVIDIA 驱动            | 需支持 CUDA 12.x          | PyTorch 的 cu126 轮子自带 CUDA 运行时，训练/推理无需系统安装 CUDA，仅编译扩展时需要 Toolkit                                                            |
| ninja                  | —                         | 由 `uv sync` 自动安装（dev 依赖组），用于并行编译                                                                                                      |

补充说明：

- 编译脚本会依次检测 `CUDA_HOME` / `CUDA_ROOT`、`/usr/local/cuda` 以及 `/usr/local/cuda-12.x`；找不到时可加 `--cuda 12.6` 指定版本或直接传入路径。
- 编译脚本已自动附加 `-allow-unsupported-compiler`，但 GCC > 13 属于 CUDA 未官方支持的组合，出问题时请改用 GCC ≤ 13。

# 🚀 安装

1. 安装依赖（uv 会创建虚拟环境，并安装 PyTorch、spconv 等全部依赖）

```bash
uv sync
```

2. 编译 C++/CUDA 算子（在项目根目录生成 `HAIS_OP.so`）

```bash
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
uv run python train.py --data_dir /path/to/EV-UAV --gpu 0
```

**2) 测试**

```bash
uv run python test.py --model_path runs/train_xxx/checkpoints/best_iou_seed37.pt
```

**3) 预训练权重**

预训练权重可在 [百度网盘](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4)（提取码：ztp4）或 [Google Drive](https://drive.google.com/file/d/1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq/view?usp=drive_link) 下载。

---

## 引用

如果你在研究中使用了本工作，请引用：

```bibtex
@misc{chen2025eventbasedtinyobjectdetection,
      title={Event-based Tiny Object Detection: A Benchmark Dataset and Baseline},
      author={Nuo Chen and Chao Xiao and Yimian Dai and Shiman He and Miao Li and Wei An},
      year={2025},
      eprint={2506.23575},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23575},
}
```

## ⚖️ 许可证

本项目基于 **Apache 2.0 许可证**发布。
但**未经许可，严禁商业使用与修改**。

如果你引用或基于本工作进行构建，请按上述方式引用我们的论文。

## 致谢

本代码基于 [HAIS](https://github.com/hustvl/HAIS) 和 [spconv](https://github.com/traveller59/spconv)。

## Ablation experiments / 消融实验

See [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md) for presets, parameter overrides and evaluation protocol.

```bash
uv run python train.py --experiment a0_mbtc --run_name a0_mbtc_s37 --seed 37 --train_workers 0
```

Presets: `a0`, `a0_mbtc`, `attention_fix`, `a1`, `a2`, `a3`, `a4`. Explicit flags override presets. Training and grid search default to no test-set evaluation; use `--test_after_train` only after freezing the experimental configuration.
