English | [简体中文](README.zh-CN.md)

# 🚁 Event-based Tiny Object Detection: A Benchmark Dataset and Baseline

<p align="center">
<img src="imgs/logo.png"  width='800' />
 </a>
</p>
The official implementation of "Event-based Tiny Object Detection: A Benchmark Dataset and Baseline" (ICCV 2025) [Paper](https://arxiv.org/abs/2506.23575)

## 🌟 Abstract

Small object detection (SOD) in anti-UAV task is a challenging problem due to the small size of UAVs and complex backgrounds. Traditional frame-based cameras struggle to detect small objects in complex environments due to their low frame rates, limited dynamic range, and data redundancy. Event cameras, with microsecond temporal resolution and high dynamic range, provide a more effective solution for SOD. However, existing event-based object detection datasets are limited in scale, feature large targets size, and lack diverse backgrounds, making them unsuitable for SOD benchmarks. In this paper, we introduce a Event-based Small object detection (EVSOD) dataset (namely **EV-UAV**), the first large-scale, highly diverse benchmark for anti-UAV tasks. It includes 147 sequences with over **2.3 million event-level annotations,** featuring **extremely small targets** (averaging 6.8 × 5.4 pixels) and **diverse scenarios** such as urban clutter and extreme lighting conditions. Furthermore, based on the observation that small moving targets form continuous curves in spatiotemporal event point clouds, we propose Event based Sparse Segmentation Network (EV-SpSegNet), a novel baseline for event segmentation in point cloud space, along with a Spatiotemporal Correlation (STC) loss that leverages motion continuity to guide the network in retaining target events. Extensive experiments on the EV-UAV dataset demonstrate the superiority of our method and provide a benchmark for future research in EVSOD.

---

## 📊EV-UAV dataset

- ### Comparison between event camera and RGB camera

<p align="center">
<img src="imgs/left_top.jpg"  width='500' />
 </a>
</p>
The RGB camera can only capture the objects under normal light, while the event camera  can capture objects under various extreme lighting conditions. And the event camera can capture the continuous motion  trajectory of the small object (shown as the red curve).

- ### Comparison between EV-UAV and other event-based object detection datasets

<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Dataset</th>
    <th class="tg-c3ow" rowspan="2">#AGV.UAV scale</th>
    <th class="tg-c3ow" rowspan="2">Label Type</th>
    <th class="tg-c3ow" rowspan="2">UAV Sequence Ratio</th>
    <th class="tg-c3ow" rowspan="2">UAV centric</th>
    <th class="tg-c3ow" colspan="3">Lighting conditions</th>
    <th class="tg-0pky" colspan="2">Object</th>
    <th class="tg-0pky" rowspan="2">Year</th>
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
    <td class="tg-0pky">84×66 pixels</td>
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
    <td class="tg-0pky">129×100 pixels</td>
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
    <td class="tg-0pky">68×45 pixels</td>
    <td class="tg-0pky">BBox</td>
    <td class="tg-0pky">3.57</td>
    <td class="tg-0pky">×</td>
    <td class="tg-0pky">×</td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
    <td class="tg-lqch"><span style="color:#FE0000">√</span></td>
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
    <td class="tg-0pky">55×31 pixels</td>
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
    <td class="tg-0pky">6.8×5.4 pixels</td>
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
Currently, event-based object detection datasets  primarily focus on autonomous driving and general object detection.  There is limited attention given to  datasets that are  exclusively designed for UAV detection. We provide a comprehensive summary of existing datasets, highlighting the scarcity of event-based datasets for UAV object detection.

- ### Benchmark Features and Statistics

<img src="imgs/datasets.jpg" style="zoom: 25%;" />

EV-UAV contains 147 event sequences with **event-level annotations**, covering **challenging scenarios** like high-brightness and low-light conditions, with **targets** **averaging 1/50 the size in existing datasets**.

---

## 📂 Structure of EV-UAV

The file structure of the dataset is as follows:

```
EV-UAV/
├── test/
│   ├── test_000.npz
│   ├── test_001.npz
│   ├──.....
├── train
├── val

```

---

## 📝 Data Format

Event data is stored in `.npz` format, it contains three files (i.e., 'evs_norm', 'ev_loc' and 'ev').

**'ev'** is the raw event data.

- **x, y:** Pixel coordinates of the event.
- **t:** Timestampof event occurrence (millisecond).
- **p:** Polarity of brightness change (1 or 0).
- **label:** Indicates if it's the target (0 or 1).
- **name:** Identity of the target .

Example:

```
x    y   t  p label name
100 200  1            1        0    0
128 258  4000        0        1    5
```

**'evs_norm'** is the normalized event data.

Example:

```
x    y   t  p label name
0.289 0.769  0            1        0    0
0.369 0.992  0.5         0        1    5
```

**'ev_loc'** is the coordinate of the event in point cloud space.

Example:

```
x    y   t
100 200  1
128 258  4000
```

## ⬇️ Dataset

The EV-UAV dataset can be download from [Baidu Netdisk](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2). Extracted code: sbr2. [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link)

---

# :triangular_flag_on_post:Baseline

Leveraging the spatiotemporal correlation characteristics of moving targets in event data, we propose EV-SpSegNet, a direct segmentation network for sparse event point clouds, and design a spatiotemporal correlation loss function that optimizes target event extraction by evaluating local spatiotemporal consistency.

### Event based Sparse Segmentation Network

Event based Sparse Segmentation Network (EV-SpSegNet) employs a U-shaped encoder-decoder architecture, integrating three key components: the GDSCA module (Grouped Dilated Sparse Convolution) for multi-scale temporal feature extraction, the Sp-SE module for feature fusion, and the Patch Attention block for voxel downsampling and global context modeling.

<img src="imgs/framework.png" width='900' />

- ### STCLOSS

We introduce a spatiotemporal correlation loss that encourages the network to retain more events with high spatiotemporal correlation while discarding more isolated noise.
<p align="center">
<img src="imgs/stcloss1.png"  width='300' />
 </a>
</p>
<p align="center">
<img src="imgs/stcloss2.png"  width='300' />
 </a>
</p>

# 🧰Environment requirements

Python dependencies and the virtual environment are managed by [uv](https://docs.astral.sh/uv/). Before compiling the CUDA extension, make sure your toolchain matches the table below:

| Dependency | Version | Notes |
| --- | --- | --- |
| Linux + NVIDIA GPU | — | compiling the `HAIS_OP` CUDA extension is Linux-only |
| Python | `>=3.10, <3.14` | managed automatically by uv, no manual install needed |
| uv | any recent version | dependency & virtualenv management |
| CUDA Toolkit (`nvcc`) | **12.x, 12.6 recommended** | only needed to compile the `HAIS_OP` extension. PyTorch and spconv are cu126 builds, so the `nvcc` major version must be **12**; 12.6 keeps it identical |
| GCC / g++ | **<= 13 (11–13 recommended)** | upper bound officially supported by CUDA 12.x. If the system g++ is newer, the build script auto-searches for GCC 13/12/11 under `/opt`, `/usr/local`, `/usr`, or pass `--gcc 13` |
| NVIDIA driver | CUDA 12.x support | the PyTorch cu126 wheels bundle the CUDA runtime, so training/inference work without a system CUDA install; the toolkit is only needed for extension builds |
| ninja | — | installed automatically by `uv sync` (dev group), used for parallel builds |

Additional notes:

- The build script probes `CUDA_HOME` / `CUDA_ROOT`, `/usr/local/cuda`, and `/usr/local/cuda-12.x` in order; if none is found, pass `--cuda 12.6` or an explicit path.
- The script already passes `-allow-unsupported-compiler`, but GCC > 13 is not officially supported by CUDA — if the build fails, switch to GCC <= 13.

# 🚀Installation

1. Install dependencies with [uv](https://docs.astral.sh/uv/) (creates the virtualenv and installs PyTorch, spconv, etc.)

```
uv sync
```

2. Compile the external C++ and CUDA ops (produces `HAIS_OP.so` in the project root)

```
uv run python scripts/build_ext.py
```

Common options:

```bash
uv run python scripts/build_ext.py --gcc 13          # pin GCC 13 when the system GCC is too new
uv run python scripts/build_ext.py --cuda 12.6       # pin the CUDA version, or pass a path directly
uv run python scripts/build_ext.py --clean --gcc 13  # clean previous build artifacts and rebuild
```

# 🎯Running code

**1) Training** — pass the dataset root via `--data_dir` (default `/data/ev-uav`); see `utils/args.py` for the full list of options

```
uv run python train.py --data_dir /path/to/EV-UAV --gpu 0
```

**2) Testing**

```
uv run python test.py --model_path runs/train_xxx/checkpoints/best_iou_seed37.pt
```

**3) Pre_trained weights**

The pre_trained weights can be download [here](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4). Extracted code:ztp4. [Google Drive](https://drive.google.com/file/d/1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq/view?usp=drive_link)

---

## Citation

If you use this work in your research, please cite it:

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

## ⚖️ License

This project is released under the **Apache 2.0 License**.
However, **commercial use and modification without permission are strictly prohibited**.

If you reference or build upon this work, please acknowledge our paper as above.

## Acknowledgement

The code is based on [HAIS](https://github.com/hustvl/HAIS) and [spconv](https://github.com/traveller59/spconv).

## Ablation experiments / 消融实验

See [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md) for presets, parameter overrides and evaluation protocol.

```bash
uv run python train.py --experiment a0_mbtc --run_name a0_mbtc_s37 --seed 37 --train_workers 0
```

Presets: `a0`, `a0_mbtc`, `attention_fix`, `a1`, `a2`, `a3`, `a4`. Explicit flags override presets. Training and grid search default to no test-set evaluation; use `--test_after_train` only after freezing the experimental configuration.
