# FluxUNet (信号转图像)

**FluxUNet** 是一种 **维流体 (Dimensional-Fluid)** 架构，通过将 1D 信号转换为 2D 时频频谱图进行去噪。它使用带有 **ResNet** 骨干的 **U-Net** 架构来处理复杂的连续小波变换 (CWT) 系数。

## 架构 (Architecture)

```mermaid
graph TB
    %% Styles
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef encoder fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef bottleneck fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef decoder fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    subgraph Signal_Transform ["维度升空 (Dimensional Lift-Off)"]
        S1[1D 信号<br/>B x 1 x 1024]:::input
        CWT[CWT 变换]
        S2[2D 频谱图<br/>B x 2 x 64 x 1024]:::input
    end

    subgraph FluxUNet ["FluxUNet (2D ResNet U-Net)"]
        direction TB
        
        subgraph Encoder_Path ["编码器路径"]
            E0[输入投影]:::encoder
            E1[编码块]:::encoder
            E2[编码块]:::encoder
            E3[编码块]:::encoder
        end
        
        B[瓶颈层]:::bottleneck
        
        subgraph Decoder_Path ["解码器路径"]
            D3[解码块]:::decoder
            D2[解码块]:::decoder
            D1[解码块]:::decoder
            OUT[输出投影]:::decoder
        end
    end

    subgraph Signal_Reconstruct ["维度重入 (Dimensional Re-Entry)"]
        R1[2D 输出<br/>B x 2 x 64 x 1024]:::input
        ICWT[ICWT 逆变换]
        R2[1D 信号<br/>B x 1 x 1024]:::input
    end

    %% Main Flow
    S1 --> CWT --> S2 --> E0
    E0 --> E1 --> E2 --> E3 --> B
    B --> D3 --> D2 --> D1 --> OUT --> R1
    R1 --> ICWT --> R2

    %% Skip Connections
    E3 -.->|skip| D3
    E2 -.->|skip| D2
    E1 -.->|skip| D1
```

### 核心原理 (Core Principles)

1. **维度流动性 (Dimensional-Fluidity)**: 无缝切换 1D 信号空间和 2D 时频空间。
2. **相位保留 (Phase Preservation)**: 处理 CWT 的实部和虚部（2 通道）以保留对精确重建至关重要的相位信息。
3. **混合损失 (Combined Loss)**: 使用多目标损失函数进行训练：
    $$ L_{total} = L_{MSE} + 0.5 \cdot L_{Spectral} + 0.1 \cdot L_{Phase} $$

## 使用方法 (Usage)

### 训练 (Training)

```bash
python src/scripts/train_flux.py --epochs 100
```

### 推理 (Inference)

```python
from src.models.flux import FluxUNet
from src.core.transmuters import DimensionalTransmuter
import torch

# 初始化
transmuter = DimensionalTransmuter(fs=6.25e6)
model = FluxUNet(in_channels=2, base_channels=64)
model.eval()

# 升空 -> 去噪 -> 重入
signal = ... # numpy array
coeffs, meta = transmuter.lift_off(signal) 
with torch.no_grad():
    pred_coeffs = model(torch.from_numpy(coeffs))
denoised_signal = transmuter.re_entry(pred_coeffs.numpy(), meta)
```
