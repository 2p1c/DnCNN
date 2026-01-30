# FluxUNet (Signal-to-Image)

**FluxUNet** is a **Dimensional-Fluid** architecture that denoises 1D signals by transforming them into 2D time-frequency spectrograms. It uses a **U-Net** architecture with **ResNet** backbones to process complex Continuous Wavelet Transform (CWT) coefficients.

## Architecture

```mermaid
graph TB
    %% Styles
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef encoder fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef bottleneck fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef decoder fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    subgraph Signal_Transform ["Dimensional Lift-Off"]
        S1[1D Signal<br/>B x 1 x 1024]:::input
        CWT[CWT Transform]
        S2[2D Spectrogram<br/>B x 2 x 64 x 1024]:::input
    end

    subgraph FluxUNet ["FluxUNet (2D ResNet U-Net)"]
        direction TB
        
        subgraph Encoder_Path ["Encoder Path"]
            E0[Input Proj]:::encoder
            E1[EncoderBlock]:::encoder
            E2[EncoderBlock]:::encoder
            E3[EncoderBlock]:::encoder
        end
        
        B[Bottleneck]:::bottleneck
        
        subgraph Decoder_Path ["Decoder Path"]
            D3[DecoderBlock]:::decoder
            D2[DecoderBlock]:::decoder
            D1[DecoderBlock]:::decoder
            OUT[Output Proj]:::decoder
        end
    end

    subgraph Signal_Reconstruct ["Dimensional Re-Entry"]
        R1[2D Output<br/>B x 2 x 64 x 1024]:::input
        ICWT[ICWT Transform]
        R2[1D Signal<br/>B x 1 x 1024]:::input
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

### Core Principles

1. **Dimensional-Fluidity**: Seamlessly transitions between 1D signal space and 2D time-frequency space.
2. **Phase Preservation**: Processes both Real and Imaginary CWT components (2 channels) to preserve phase information crucial for accurate reconstruction.
3. **Combined Loss**: Trained with a multi-objective loss function:
    $$ L_{total} = L_{MSE} + 0.5 \cdot L_{Spectral} + 0.1 \cdot L_{Phase} $$

## Usage

### Training

```bash
python src/scripts/train_flux.py --epochs 100
```

### Inference

```python
from src.models.flux import FluxUNet
from src.core.transmuters import DimensionalTransmuter
import torch

# Initialize
transmuter = DimensionalTransmuter(fs=6.25e6)
model = FluxUNet(in_channels=2, base_channels=64)
model.eval()

# Lift-off -> Denoise -> Re-entry
signal = ... # numpy array
coeffs, meta = transmuter.lift_off(signal) 
with torch.no_grad():
    pred_coeffs = model(torch.from_numpy(coeffs))
denoised_signal = transmuter.re_entry(pred_coeffs.numpy(), meta)
```
