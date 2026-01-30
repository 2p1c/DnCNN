# Laser-to-Piezoelectric Signal Mapping for Ultrasonic Guided Waves

A neural network system to map laser ultrasonic signals to piezoelectric ultrasonic signals for enhanced signal quality.

## 🎯 Project Overview

| Component | Description |
|-----------|-------------|
| **Input** | 21×21 grid laser ultrasonic signals (→ interpolated to 41×41) |
| **Target** | 41×41 grid piezoelectric ultrasonic signals |
| **Physics** | Aluminum plate, 200kHz center frequency |
| **Models** | 1D DenoiseNet + **2D FluxUNet (CWT-based)** |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies with uv
uv sync

# 2. Run Gravity Loop (FluxUNet 2D U-Net)
uv run python src/scripts/train_flux.py --sanity-check
uv run python src/scripts/train_flux.py --overfit-test --samples 50 --epochs 50

# 3. Full training
uv run python src/scripts/train_flux.py --epochs 100

# 4. Visualize results
uv run python src/core/visualize.py --num-samples 10

# Legacy: 1D DenoiseNet
uv run python src/scripts/train_denoisenet.py --overfit-test --epochs 50 --samples 100
```

---

## 📁 Project Structure

```
DnCNN/
├── src/
│   ├── models/                  # Neural Network Architectures
│   │   ├── denoisenet_1d/       # DenoiseNet (1D CNN)
│   │   ├── dncnn_cwt/           # DnCNN-CWT (2D ResNet)
│   │   └── flux/                # FluxUNet (Signal-to-Image)
│   ├── data/                    # Data Management
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── loader.py            # MATLAB Data Loading
│   ├── core/                    # Core Utilities
│   │   ├── transmuters/         # Wavelet Processing
│   │   └── visualize.py         # Plotting Tools
│   └── scripts/                 # Training Scripts
│       ├── train_flux.py        # FluxUNet Training
│       └── train_denoisenet.py  # DenoiseNet Training
├── configs/
│   └── default.yaml             # Hyperparameters and data paths
├── checkpoints/
│   └── gravity_loop/            # FluxUNet checkpoints
└── pyproject.toml               # Dependencies
```

---

## 🧠 Network Architectures

### FluxUNet (2D U-Net) - Recommended

**Signal-to-Image-to-Signal Pipeline:**

```
1D Signal → CWT → 2D Spectrogram → FluxUNet → ICWT → 1D Signal
 (1024)         (2×64×1024)                         (1024)
```

| Component | Details |
|-----------|---------|
| Transform | Complex Morlet CWT (cmor1.5-1.0) |
| Channels | 2 (Real/Imaginary for phase preservation) |
| Encoder | 3 stages: 64→128→256 with ResBlocks |
| Decoder | Skip connections + ConvTranspose2d |
| Parameters | ~9.3M |

**Loss Function:**

```
L_total = MSE + 0.5 × SpectralConvergence + 0.1 × PhaseLoss
```

### DenoiseNet (1D CNN) - Legacy

```
Encoder: Conv1d(1→64, k=7) → MaxPool(2) → Conv1d(64→128, k=5) → MaxPool(2)
Bottleneck: Conv1d(128→128, k=3)
Decoder: ConvTranspose1d(128→64) → ConvTranspose1d(64→1)
```

---

## 📊 Verification

### FluxUNet Pipeline

1. **Sanity Check**: CWT→ICWT reconstruction (Correlation > 0.999)
2. **Overfit Test**: 50 samples, expect loss convergence
3. **Visualization**: `checkpoints/gravity_loop/random_samples_comparison.png`

### Metrics

- **SNR (dB)**: Signal-to-Noise Ratio improvement
- **SSIM**: Structural similarity on spectrograms
- **Correlation**: Time-domain waveform similarity

---

## 🔧 Configuration

Edit `configs/default.yaml`:

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

## 📚 Documentation

| Document | Location |
|----------|----------|
| DenoiseNet Architecture | [src/models/denoisenet_1d/README.md](src/models/denoisenet_1d/README.md) |
| DnCNN-CWT Architecture | [src/models/dncnn_cwt/README.md](src/models/dncnn_cwt/README.md) |
| FluxUNet Architecture | [src/models/flux/README.md](src/models/flux/README.md) |

---

## 🛠️ Dependencies

```bash
pip install torch numpy scipy matplotlib pyyaml tqdm PyWavelets
```
