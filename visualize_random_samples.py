"""
@Antigravity.Viz - Random Sample Visualization for FluxUNet (2D U-Net)
======================================================================

Visualizes denoising results from the Signal-to-Image-to-Signal pipeline.
Randomly selects spatial points and compares original vs denoised signals.

Output: checkpoints/gravity_loop/random_samples_comparison.png
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for direct script execution
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.data_loader import load_mat_file, pad_or_crop_signal, interpolate_target_grid
from src.flux import FluxUNet
from src.transmuters import DimensionalTransmuter

# Matplotlib settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_gravity_model(checkpoint_path: Path):
    """
    Load trained FluxUNet model and transmuter config.
    
    Args:
        checkpoint_path: Path to best_model.pt
        
    Returns:
        model: Loaded FluxUNet model
        transmuter_config: Dict with wavelet, scales, fs, freq_range
        norm_params: Dict with normalization parameters for bilateral denormalization
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Reconstruct model
    model = FluxUNet(in_channels=2, base_channels=64, num_res_blocks=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get transmuter config
    transmuter_config = checkpoint.get('transmuter_config', {
        'wavelet': 'cmor1.5-1.0',
        'num_scales': 64,
        'fs': 6.25e6,
        'freq_range': (50e3, 500e3)
    })
    
    # Get normalization params (for bilateral denormalization)
    norm_params = checkpoint.get('norm_params', {
        'type': 'instance',
        'input_mean': 0.0, 'input_std': 1.0,
        'target_mean': 0.0, 'target_std': 1.0
    })
    
    print(f"[Loaded] Model from: {checkpoint_path}")
    print(f"[Loaded] Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"[Loaded] Loss: {checkpoint.get('loss', 'unknown'):.6f}")
    print(f"[Loaded] Normalization: {norm_params.get('type', 'unknown')}")
    
    return model, transmuter_config, norm_params


def denoise_signal_2d(model, transmuter, signal, norm_params=None):
    """
    Denoise a single signal using FluxUNet pipeline with bilateral normalization.
    
    Args:
        model: FluxUNet model
        transmuter: DimensionalTransmuter instance
        signal: 1D numpy array of shape (length,)
        norm_params: Dict with normalization parameters
                     - 'input_mean', 'input_std': for normalizing input
                     - 'target_mean', 'target_std': for denormalizing output
        
    Returns:
        denoised: Denoised 1D signal (in original amplitude scale of target)
    """
    # Determine normalization mode
    if norm_params is None or norm_params.get('type') == 'instance':
        # Instance normalization (legacy)
        signal_mean = signal.mean()
        signal_std = signal.std() + 1e-20
        target_mean = signal_mean  # Denorm using input stats
        target_std = signal_std
    else:
        # Bilateral normalization: use global stats
        signal_mean = norm_params['input_mean']
        signal_std = norm_params['input_std'] + 1e-8
        target_mean = norm_params['target_mean']
        target_std = norm_params['target_std'] + 1e-8
    
    # Normalize input with INPUT stats
    signal_norm = (signal - signal_mean) / signal_std
    
    # Lift off: 1D -> 2D
    coeffs_2d, meta = transmuter.lift_off(signal_norm, apply_compression=True)
    
    # Model inference
    with torch.no_grad():
        coeffs_tensor = torch.from_numpy(coeffs_2d).float()  # (1, 2, S, T)
        pred_2d = model(coeffs_tensor).numpy()
    
    # Re-entry: 2D -> 1D
    denoised_norm = transmuter.re_entry(pred_2d, meta, decompress=True)
    denoised_norm = denoised_norm.squeeze()  # (length,)
    
    # Denormalize with TARGET stats
    denoised = denoised_norm * target_std + target_mean
    
    return denoised


def plot_random_samples(
    raw_data: np.ndarray,
    denoised_data: np.ndarray,
    ground_truth_data: np.ndarray,
    time: np.ndarray,
    num_samples: int = 10,
    output_path: Path = None
):
    """
    Plot random samples comparison (Original vs Denoised vs Ground Truth).
    
    Args:
        raw_data: (n_y, n_x, n_time) noisy signals
        denoised_data: (n_y, n_x, n_time) denoised signals
        ground_truth_data: (n_y, n_x, n_time) target signals (can be None)
        time: time vector
        num_samples: number of random samples to plot
        output_path: save path
    """
    n_y, n_x, n_time = raw_data.shape
    total_points = n_y * n_x
    
    # Randomly select spatial indices
    np.random.seed(42)
    selected_indices = np.random.choice(total_points, size=min(num_samples, total_points), replace=False)
    
    # Create figure
    rows = (num_samples + 1) // 2
    fig = plt.figure(figsize=(16, 3 * rows))
    gs = GridSpec(rows, 2, figure=fig, hspace=0.4, wspace=0.25)
    
    for i, spatial_idx in enumerate(selected_indices):
        row = spatial_idx // n_x
        col = spatial_idx % n_x
        
        # Extract signals
        raw_signal = raw_data[row, col, :]
        denoised_signal = denoised_data[row, col, :]
        
        # Create subplot
        ax = fig.add_subplot(gs[i // 2, i % 2])
        
        # Time in microseconds
        time_us = time[:len(raw_signal)] * 1e6
        
        # Plot signals
        ax.plot(time_us, raw_signal, 'b-', alpha=0.5, linewidth=0.8, label='Original (Noisy)')
        ax.plot(time_us, denoised_signal, 'r-', linewidth=1.2, label='Denoised (FluxUNet)')
        
        if ground_truth_data is not None:
            gt_signal = ground_truth_data[row, col, :]
            ax.plot(time_us, gt_signal, 'g--', linewidth=1.0, alpha=0.7, label='Ground Truth')
        
        # Formatting
        ax.set_xlabel('Time (us)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.set_title(f'Point ({row}, {col}) | Index {spatial_idx}', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Compute SNR improvement if ground truth available
        if ground_truth_data is not None:
            gt_signal = ground_truth_data[row, col, :]
            # SNR of denoised vs ground truth
            signal_power = np.mean(gt_signal ** 2)
            raw_error = np.mean((gt_signal - raw_signal) ** 2)
            denoised_error = np.mean((gt_signal - denoised_signal) ** 2)
            
            snr_raw = 10 * np.log10(signal_power / (raw_error + 1e-12))
            snr_denoised = 10 * np.log10(signal_power / (denoised_error + 1e-12))
            snr_improvement = snr_denoised - snr_raw
            
            stats_text = f'SNR: {snr_raw:.1f} -> {snr_denoised:.1f} dB (+{snr_improvement:.1f})'
        else:
            # Simple noise reduction metric
            raw_std = np.std(raw_signal)
            denoised_std = np.std(denoised_signal)
            noise_reduction = (1 - denoised_std / raw_std) * 100 if raw_std > 0 else 0
            stats_text = f'Std Reduction: {noise_reduction:.1f}%'
        
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    
    fig.suptitle('FluxUNet Denoising Results: Random Sample Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] Random samples visualization: {output_path}")
    
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize FluxUNet denoising results")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/gravity_loop/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of random samples to visualize')
    parser.add_argument('--output-dir', type=str, default='checkpoints/gravity_loop',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FluxUNet Random Sample Visualization")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading FluxUNet model...")
    model, transmuter_config, norm_params = load_gravity_model(Path(args.checkpoint))
    
    # Initialize transmuter
    print("\n[2/4] Initializing DimensionalTransmuter...")
    transmuter = DimensionalTransmuter(
        wavelet=transmuter_config.get('wavelet', 'cmor1.5-1.0'),
        num_scales=transmuter_config.get('num_scales', 64),
        fs=float(transmuter_config.get('fs', 6.25e6)),
        freq_range=tuple(transmuter_config.get('freq_range', (50e3, 500e3)))
    )
    print(f"    Wavelet: {transmuter.wavelet}")
    print(f"    Scales: {transmuter.num_scales}")
    
    # Load raw data
    print("\n[3/4] Loading and processing data...")
    noisy_path = config['data']['noisy_path']
    noisy_grid_size = config['data']['noisy_grid_size']
    target_length = config['signal']['target_length']
    
    raw_data = load_mat_file(noisy_path, n_points=noisy_grid_size)
    raw_xyt = raw_data['data_xyt']
    time = raw_data['time']
    
    print(f"    Raw data shape: {raw_xyt.shape}")
    
    # Load ground truth if available
    ground_truth_xyt = None
    try:
        target_path = config['data']['target_path']
        target_grid_size = config['data']['target_grid_size']
        target_data = load_mat_file(target_path, n_points=target_grid_size)
        ground_truth_xyt = target_data['data_xyt']
        # Interpolate to match noisy grid
        ground_truth_xyt = interpolate_target_grid(ground_truth_xyt, (noisy_grid_size, noisy_grid_size))
        print(f"    Ground truth shape: {ground_truth_xyt.shape}")
    except Exception as e:
        print(f"    [Info] Ground truth not loaded: {e}")
    
    # Denoise all signals
    print("\n[4/4] Denoising signals with FluxUNet...")
    n_y, n_x, n_time = raw_xyt.shape
    denoised_xyt = np.zeros_like(raw_xyt)
    
    for i in range(n_y):
        for j in range(n_x):
            signal = raw_xyt[i, j, :]
            # Pad/crop to model input size
            signal_processed = pad_or_crop_signal(signal[np.newaxis, :], target_length)[0]
            
            # Denoise with FluxUNet (using bilateral normalization params)
            denoised = denoise_signal_2d(model, transmuter, signal_processed, norm_params)
            
            # Crop back to original length
            denoised_xyt[i, j, :] = denoised[:n_time]
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"    Processed row {i+1}/{n_y}")
    
    print(f"    Denoised shape: {denoised_xyt.shape}")
    
    # Generate visualization
    print("\n[Output] Generating visualization...")
    output_path = output_dir / "random_samples_comparison.png"
    plot_random_samples(
        raw_xyt, 
        denoised_xyt, 
        ground_truth_xyt,
        time,
        num_samples=args.num_samples,
        output_path=output_path
    )
    
    print("\n" + "=" * 60)
    print("[DONE] Visualization complete!")
    print(f"    Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
