# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Install Dependencies**: `uv sync` (or `uv sync --dev` for dev tools)
- **Run Training (FluxUNet)**: `uv run python src/scripts/train_flux.py`
  - Sanity check: `uv run python src/scripts/train_flux.py --sanity-check`
  - Overfit test: `uv run python src/scripts/train_flux.py --overfit-test --samples 50 --epochs 50`
- **Run Training (Legacy DenoiseNet)**: `uv run python src/scripts/train_denoisenet.py`
- **Visualize Results**: `uv run python src/core/visualize.py --num-samples 10`
- **Run Tests**: `uv run pytest`
- **Run Linter**: `uv run ruff check .`

## Architecture

This project implements a neural network system to map noisy laser ultrasonic signals (input) to clean piezoelectric ultrasonic signals (target).

### Core Pipeline (FluxUNet)
The primary architecture follows a **Signal-to-Image-to-Signal** approach:
1.  **Input**: 1D time-domain signal (interpolated grid).
2.  **Transform**: Continuous Wavelet Transform (CWT) converts 1D signal to 2D complex spectrogram (Real/Imaginary channels).
3.  **Model**: `FluxUNet` (2D U-Net with ResBlocks) processes the spectrogram.
4.  **Inverse**: Inverse CWT (ICWT) reconstructs the cleaned 1D time-domain signal.

### Components
- **Models** (`src/models/`):
  - `flux/`: Current standard. 2D U-Net operating on CWT spectrograms.
  - `denoisenet_1d/`: Legacy 1D CNN operating directly on time-domain signals.
  - `dncnn_cwt/`: Alternative 2D ResNet architecture.
- **Data** (`src/data/`): Handles loading of MATLAB (.mat) files containing signal grids.
- **Core** (`src/core/`):
  - `transmuters/`: Wavelet transform logic (CWT/ICWT) using `PyWavelets`.
  - `visualize.py`: Plotting tools for waveforms and spectrograms.

### Configuration
Hyperparameters and paths are defined in `configs/default.yaml`. Code typically loads this configuration at runtime.
