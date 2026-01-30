# 激光-压电超声信号映射系统

基于神经网络的激光超声信号到压电超声信号映射系统，实现信号质量增强。

## 🎯 项目概述

| 组件 | 描述 |
|------|------|
| **输入** | 21×21 网格激光超声信号 (→ 插值至 41×41) |
| **目标** | 41×41 网格压电超声信号 |
| **物理** | 铝板，200kHz 中心频率 |
| **模型** | 1D DenoiseNet + **2D FluxUNet (基于CWT)** |

---

## 🚀 快速开始

```bash
# 1. 使用 uv 安装依赖
uv sync

# 2. 运行 Gravity Loop (FluxUNet 2D U-Net)
uv run python src/scripts/train_flux.py --sanity-check
uv run python src/scripts/train_flux.py --overfit-test --samples 50 --epochs 50

# 3. 完整训练
uv run python src/scripts/train_flux.py --epochs 100

# 4. 可视化结果
uv run python src/core/visualize.py --num-samples 10

# 传统方案: 1D DenoiseNet
uv run python src/scripts/train_denoisenet.py --overfit-test --epochs 50 --samples 100
```

---

## 📁 项目结构

```
DnCNN/
├── src/
│   ├── models/                  # 神经网络架构
│   │   ├── denoisenet_1d/       # DenoiseNet (1D CNN)
│   │   ├── dncnn_cwt/           # DnCNN-CWT (2D ResNet)
│   │   └── flux/                # FluxUNet (信号转图像)
│   ├── data/                    # 数据管理
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── loader.py            # MATLAB 数据加载
│   ├── core/                    # 核心工具
│   │   ├── transmuters/         # 小波处理
│   │   └── visualize.py         # 绘图工具
│   └── scripts/                 # 训练脚本
│       ├── train_flux.py        # FluxUNet 训练
│       └── train_denoisenet.py  # DenoiseNet 训练
├── configs/
│   └── default.yaml             # 超参数和数据路径
├── checkpoints/
│   └── gravity_loop/            # FluxUNet 检查点
└── pyproject.toml               # 依赖配置
```

---

## 🧠 网络架构

### FluxUNet (2D U-Net) - 推荐方案

**信号→图像→信号 流水线:**

```
1D 信号 → CWT → 2D 时频图 → FluxUNet → ICWT → 1D 信号
 (1024)         (2×64×1024)                    (1024)
```

| 组件 | 详情 |
|------|------|
| 变换 | 复数 Morlet 小波 (cmor1.5-1.0) |
| 通道 | 2 (实部/虚部，保留相位信息) |
| 编码器 | 3 级: 64→128→256，带 ResBlocks |
| 解码器 | 跳跃连接 + ConvTranspose2d |
| 参数量 | ~930 万 |

**损失函数:**

```
L_total = MSE + 0.5 × 频谱收敛损失 + 0.1 × 相位损失
```

### DenoiseNet (1D CNN) - 传统方案

```
编码器: Conv1d(1→64, k=7) → MaxPool(2) → Conv1d(64→128, k=5) → MaxPool(2)
瓶颈层: Conv1d(128→128, k=3)
解码器: ConvTranspose1d(128→64) → ConvTranspose1d(64→1)
```

---

## 📊 验证方法

### FluxUNet 流水线

1. **健全性检查**: CWT→ICWT 重建 (相关系数 > 0.999)
2. **过拟合测试**: 50 样本，观察损失收敛
3. **可视化**: `checkpoints/gravity_loop/random_samples_comparison.png`

### 评估指标

- **SNR (dB)**: 信噪比提升
- **SSIM**: 时频图结构相似性
- **相关系数**: 时域波形相似度

---

## 🔧 配置

编辑 `configs/default.yaml`:

```yaml
data:
  noisy_path: "path/to/41_41.mat"
  target_path: "path/to/51_51.mat"
  
signal:
  sampling_rate: 6.25e6    # 6.25 MHz
  target_length: 1024
  bandpass_low: 100e3      # 100 kHz
  bandpass_high: 300e3     # 300 kHz

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  device: "cpu"
```

---

## 📚 文档

| 文档 | 位置 |
|------|------|
| DenoiseNet 架构 | [src/models/denoisenet_1d/README.md](src/models/denoisenet_1d/README.md) |
| DnCNN-CWT 架构 | [src/models/dncnn_cwt/README.md](src/models/dncnn_cwt/README.md) |
| FluxUNet 架构 | [src/models/flux/README.md](src/models/flux/README.md) |

---

## 🛠️ 依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install torch numpy scipy matplotlib pyyaml tqdm PyWavelets
```
