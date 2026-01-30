# DnCNN-CWT (2D ResNet)

**DnCNN-CWT** is a denoising architecture that processes signals in the Time-Frequency domain using Continuous Wavelet Transform (CWT). It utilizes a **2D ResNet** backbone to learn noise patterns in the wavelet domain.

## Architecture

```mermaid
graph TB
    %% Styles
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef trans fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef conv fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef res fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    subgraph Input ["Signal Processing"]
        In(Input Signal<br/>1x1024):::input
        CWT[CWT Layer<br/>1D → 2D]:::trans
        Norm[InstanceNorm]:::trans
    end

    subgraph Feature_Extraction ["2D Feature Encoder"]
        Enc[Conv2d + BN + ReLU]:::conv
    end

    subgraph ResNet_Traj ["ResNet Backbone"]
        R1[ResBlock 1]:::res
        R2[ResBlock 2]:::res
        R_dots[...]:::res
        R8[ResBlock 8]:::res
    end

    subgraph Output_Head ["1D Decoder & Prediction"]
        Collapse[Conv2d (Freq Collapse)<br/>Scales → 1]:::conv
        Dec[1D Conv Layers]:::conv
        Pred(Clean Signal Prediction):::input
    end

    %% Flow
    In --> CWT --> Norm --> Enc
    Enc --> R1
    R1 --> R2 --> R_dots --> R8
    R8 --> Collapse --> Dec --> Pred
    In -.->|Skip Connection| Pred
```

### Core Principles

1. **CWT Feature Extraction**: Instead of attempting perfect reconstruction (which is difficult with CWT), the model uses CWT magnitudes as a rich 2D feature set.
2. **Direct Prediction**: The model directly predicts the clean signal in the time domain from 2D features, bypassing the need for an inverse CWT layer in the network graph.
3. **ResNet Backbone**: Uses 8 residual blocks (default) to process complex time-frequency patterns without vanishing gradients.

## Usage

### Training

To train the model (note: currently shares script with DenoiseNet but requires config change):

```python
# In src/scripts/train_denoisenet.py, swap the model import to DnCNN_CWT
# (Future update: dedicated training script coming soon)
```

### Inference

```python
from src.models.dncnn_cwt import DnCNN_CWT
import torch

# Initialize
model = DnCNN_CWT(signal_length=1024, num_scales=64)
model.eval()

# Denoise
noisy_signal = torch.randn(1, 1, 1024)
with torch.no_grad():
    clean_signal = model(noisy_signal)
```
