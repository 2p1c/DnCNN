# DnCNN-CWT (2D ResNet)

**DnCNN-CWT** 是一种利用 **连续小波变换 (Continuous Wavelet Transform, CWT)** 在时频域处理信号的去噪架构。它使用 **2D ResNet** 骨干网络来学习小波域中的噪声模式。

## 架构 (Architecture)

```mermaid
graph TB
    %% Styles
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef trans fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef conv fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef res fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    subgraph Input ["信号处理"]
        In(输入信号<br/>1x1024):::input
        CWT[CWT 层<br/>1D → 2D]:::trans
        Norm[InstanceNorm]:::trans
    end

    subgraph Feature_Extraction ["2D 特征编码器"]
        Enc[Conv2d + BN + ReLU]:::conv
    end

    subgraph ResNet_Traj ["ResNet 骨干"]
        R1[ResBlock 1]:::res
        R2[ResBlock 2]:::res
        R_dots[...]:::res
        R8[ResBlock 8]:::res
    end

    subgraph Output_Head ["1D 解码器 & 预测"]
        Collapse[Conv2d (频率坍缩)<br/>Scales → 1]:::conv
        Dec[1D Conv 层]:::conv
        Pred(纯净信号预测):::input
    end

    %% Flow
    In --> CWT --> Norm --> Enc
    Enc --> R1
    R1 --> R2 --> R_dots --> R8
    R8 --> Collapse --> Dec --> Pred
    In -.->|Skip Connection| Pred
```

### 核心原理 (Core Principles)

1. **CWT 特征提取**: 模型不尝试进行完美的逆变换（这对 CWT 来说很困难），而是将 CWT 幅度作为丰富的 2D 特征集。
2. **直接预测**: 模型直接从 2D 特征预测时域中的纯净信号，从而绕过了在网络图中显式使用逆 CWT 层。
3. **ResNet 骨干**: 使用 8 个残差块（默认）来处理复杂的时频模式，同时避免梯度消失问题。

## 使用方法 (Usage)

### 训练 (Training)

训练模型（注意：当前与 DenoiseNet 共享脚本，但需要更改配置）：

```python
# 在 src/scripts/train_denoisenet.py 中，将模型导入更改为 DnCNN_CWT
# (未来更新：即将推出专用训练脚本)
```

### 推理 (Inference)

```python
from src.models.dncnn_cwt import DnCNN_CWT
import torch

# 初始化
model = DnCNN_CWT(signal_length=1024, num_scales=64)
model.eval()

# 去噪
noisy_signal = torch.randn(1, 1, 1024)
with torch.no_grad():
    clean_signal = model(noisy_signal)
```
