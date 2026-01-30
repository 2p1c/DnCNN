"""
@VizAgent - Visualization Utilities for UGW Denoising
=====================================================

Provides plotting and analysis tools for:
- Signal comparison (Raw vs Ground Truth vs Denoised)
- Power Spectral Density (PSD) analysis
- Training metrics visualization
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'lines.linewidth': 1.5,
})


def plot_signal_comparison(
    raw: np.ndarray,
    ground_truth: np.ndarray,
    denoised: np.ndarray,
    time: Optional[np.ndarray] = None,
    fs: float = 6.25e6,
    title: str = "Signal Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot stacked comparison of Raw, Ground Truth, and Denoised signals.
    
    Args:
        raw: Noisy input signal (1D array)
        ground_truth: Clean reference signal (1D array)
        denoised: Model output signal (1D array)
        time: Time vector in seconds (optional, will be generated if None)
        fs: Sampling frequency in Hz (default: 6.25 MHz)
        title: Figure title
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    n_samples = len(raw)
    if time is None:
        time = np.arange(n_samples) / fs * 1e6  # Convert to microseconds
    else:
        time = time * 1e6  # Convert to microseconds
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Raw (Noisy)
    axes[0].plot(time, raw, color='#e74c3c', alpha=0.8, label='Raw (Noisy)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Raw Signal (Single-Shot)', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Ground Truth (Averaged)
    axes[1].plot(time, ground_truth, color='#27ae60', alpha=0.8, label='Ground Truth')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Ground Truth (Averaged & Interpolated)', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Denoised
    axes[2].plot(time, denoised, color='#3498db', alpha=0.8, label='Denoised')
    axes[2].set_xlabel('Time (μs)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Model Output (Denoised)', fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VizAgent] Figure saved: {save_path}")
    
    return fig


def plot_psd_comparison(
    raw: np.ndarray,
    ground_truth: np.ndarray,
    denoised: np.ndarray,
    fs: float = 6.25e6,
    center_freq: float = 200e3,
    freq_range: Tuple[float, float] = (0, 500e3),
    title: str = "Power Spectral Density Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot PSD comparison to verify 200kHz peak preservation.
    
    Physics Check: The denoised signal should preserve the 200kHz center
    frequency component while reducing broadband noise.
    
    Args:
        raw: Noisy input signal (1D array)
        ground_truth: Clean reference signal (1D array)
        denoised: Model output signal (1D array)
        fs: Sampling frequency in Hz
        center_freq: Expected center frequency (for annotation)
        freq_range: Frequency range to plot (Hz)
        title: Figure title
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    # Compute PSD using Welch's method
    # nperseg chosen for good frequency resolution around 200kHz
    nperseg = min(1024, len(raw))
    
    f_raw, psd_raw = sp_signal.welch(raw, fs, nperseg=nperseg)
    f_gt, psd_gt = sp_signal.welch(ground_truth, fs, nperseg=nperseg)
    f_dn, psd_dn = sp_signal.welch(denoised, fs, nperseg=nperseg)
    
    # Convert to dB
    psd_raw_db = 10 * np.log10(psd_raw + 1e-12)
    psd_gt_db = 10 * np.log10(psd_gt + 1e-12)
    psd_dn_db = 10 * np.log10(psd_dn + 1e-12)
    
    # Convert frequency to kHz for plotting
    f_khz = f_raw / 1e3
    center_freq_khz = center_freq / 1e3
    freq_range_khz = (freq_range[0] / 1e3, freq_range[1] / 1e3)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(f_khz, psd_raw_db, color='#e74c3c', alpha=0.6, 
            linewidth=1, label='Raw (Noisy)')
    ax.plot(f_khz, psd_gt_db, color='#27ae60', alpha=0.8, 
            linewidth=2, label='Ground Truth')
    ax.plot(f_khz, psd_dn_db, color='#3498db', alpha=0.8, 
            linewidth=2, linestyle='--', label='Denoised')
    
    # Mark center frequency
    ax.axvline(x=center_freq_khz, color='#9b59b6', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'Center Freq ({center_freq_khz:.0f} kHz)')
    
    # Mark bandpass region (100-300 kHz)
    ax.axvspan(100, 300, alpha=0.1, color='gray', label='Bandpass Region')
    
    ax.set_xlim(freq_range_khz)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Power Spectral Density (dB)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VizAgent] Figure saved: {save_path}")
    
    return fig


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training History",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training loss values per epoch
        val_losses: List of validation loss values per epoch (optional)
        title: Figure title
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'o-', color='#3498db', 
            linewidth=2, markersize=4, label='Training Loss')
    
    if val_losses:
        ax.plot(epochs, val_losses, 's-', color='#e74c3c', 
                linewidth=2, markersize=4, label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale for better visualization of loss decay
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VizAgent] Figure saved: {save_path}")
    
    return fig


def compute_metrics(
    ground_truth: np.ndarray,
    denoised: np.ndarray
) -> dict:
    """
    Compute denoising quality metrics.
    
    Args:
        ground_truth: Clean reference signal
        denoised: Model output signal
    
    Returns:
        Dictionary with metrics: MSE, RMSE, SNR improvement, correlation
    """
    # MSE and RMSE
    mse = np.mean((ground_truth - denoised) ** 2)
    rmse = np.sqrt(mse)
    
    # Signal-to-Noise Ratio (SNR) of denoised signal
    signal_power = np.mean(ground_truth ** 2)
    noise_power = np.mean((ground_truth - denoised) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
    
    # Correlation coefficient
    correlation = np.corrcoef(ground_truth.flatten(), denoised.flatten())[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'snr_db': snr_db,
        'correlation': correlation
    }


if __name__ == "__main__":
    print("=" * 60)
    print("@VizAgent - Visualization Test (Synthetic Data)")
    print("=" * 60)
    
    # Generate synthetic test signals
    fs = 6.25e6  # 6.25 MHz
    duration = 160e-6  # 160 microseconds
    n_samples = 1024
    t = np.linspace(0, duration, n_samples)
    
    # Ground truth: 200kHz sine wave with envelope
    center_freq = 200e3
    envelope = np.exp(-((t - 80e-6) ** 2) / (2 * (20e-6) ** 2))
    ground_truth = envelope * np.sin(2 * np.pi * center_freq * t)
    
    # Raw: Ground truth + noise
    noise = np.random.randn(n_samples) * 0.3
    raw = ground_truth + noise
    
    # Denoised: Simulated (partially cleaned)
    denoised = ground_truth + np.random.randn(n_samples) * 0.05
    
    # Test signal comparison plot
    print("\n[✓] Creating signal comparison plot...")
    fig1 = plot_signal_comparison(raw, ground_truth, denoised, fs=fs)
    plt.show()
    
    # Test PSD plot
    print("[✓] Creating PSD comparison plot...")
    fig2 = plot_psd_comparison(raw, ground_truth, denoised, fs=fs)
    plt.show()
    
    # Test metrics
    print("\n[✓] Computing metrics...")
    metrics = compute_metrics(ground_truth, denoised)
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}")
    
    print("\n" + "=" * 60)
    print("@VizAgent - Test Complete")
    print("=" * 60)
