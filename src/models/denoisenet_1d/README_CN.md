# DenoiseNet (1D)

**DenoiseNet** 是一个专为超声导波 (UGW) 信号去噪设计的 1D 卷积自编码器 (Convolutional Autoencoder)。它利用 **残差学习 (Residual Learning)** 和 **编码器-解码器 (Encoder-Decoder)** 架构来预测并从输入信号中减去噪声。

## 架构 (Architecture)

```mermaid
graph LR
    %% Styles
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef conv fill:#ffecb3,stroke:#ff6f00,stroke-width:2px;
    classDef pool fill:#ffe0b2,stroke:#e65100,stroke-width:2px;
    classDef deconv fill:#e1bee7,stroke:#4a148c,stroke-width:2px;
    classDef op fill:#cfd8dc,stroke:#455a64,stroke-width:2px;

    %% Nodes
    subgraph Input ["输入层"]
        In(含噪信号<br/>1x1024)
    end

    subgraph Encoder ["编码器"]
        direction TB
        E1[Conv1d<br/>k=7, c=64]:::conv
        P1[MaxPool<br/>k=2]:::pool
        E2[Conv1d<br/>k=5, c=128]:::conv
        P2[MaxPool<br/>k=2]:::pool
    end
    
    subgraph Bottleneck ["瓶颈层"]
        B1[Conv1d<br/>k=3, c=128]:::conv
    end

    subgraph Decoder ["解码器"]
        direction TB
        D1[ConvT1d<br/>k=4, c=64]:::deconv
        D2[ConvT1d<br/>k=4, c=1]:::deconv
    end

    subgraph Output ["残差块"]
        Noise(预测的噪声)
        Sub((相减)):::op
        Out(纯净信号)
    end

    %% Flow
    In --> E1 --> P1 --> E2 --> P2 --> B1
    B1 --> D1 --> D2 --> Noise
    In -.-> Sub
    Noise --> Sub --> Out
```

### 核心原理 (Core Principles)

1. **残差学习 (Residual Learning)**: 模型预测噪声分布 $\hat{v}$ 而不是直接预测纯净信号。
    $$ \hat{x} = y - \text{DenoiseNet}(y) $$
    这利用了噪声通常比纯净导波信号具有更简单的统计分布这一特性。

2. **1D 处理**: 直接在时域信号上操作，使其高效且适用于实时应用。

## 使用方法 (Usage)

### 训练 (Training)

使用默认配置训练模型：

```bash
python src/scripts/train_denoisenet.py --epochs 100
```

### 推理 (Inference)

```python
from src.models.denoisenet_1d import DenoiseNet
import torch

# 初始化
model = DenoiseNet()
model.load_state_dict(torch.load('checkpoints/denoisenet/best_model.pt'))
model.eval()

# 去噪
noisy_signal = torch.randn(1, 1, 1024)
with torch.no_grad():
    clean_signal = model(noisy_signal)
```
