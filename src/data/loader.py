"""
@DataAgent - MATLAB Data Loader for Ultrasonic Guided Wave Signals
==================================================================

This module handles loading wavefield data from MATLAB .mat files and
provides spatial interpolation utilities.

Data Structure (from MATLAB code analysis):
- x: time vector (1D array)
- y: scan data (n_points*n_points, n_time) - column-major flattened
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom


def load_mat_file(
    file_path: Path | str,
    n_points: int,
    spacing: float = 1e-3
) -> dict:
    """
    Load MATLAB wavefield data and reshape into 3D array format.
    
    Based on MATLAB mat_loader function analysis:
    - data_struct.x -> time vector
    - data_struct.y -> scan data (n_points^2 × n_time), column-major order
    
    Args:
        file_path: Path to .mat file
        n_points: Grid size (e.g., 41 for 41×41 grid)
        spacing: Physical spacing between points in meters (default: 1mm)
    
    Returns:
        dict with keys:
            - 'data_xyt': 3D array (n_points, n_points, n_time)
            - 'time': Time vector in seconds
            - 'x_coords': X coordinate vector in meters
            - 'y_coords': Y coordinate vector in meters
            - 'fs': Sampling frequency in Hz
    
    Example:
        >>> data = load_mat_file('41_41.mat', n_points=41, spacing=1e-3)
        >>> print(data['data_xyt'].shape)  # (41, 41, 1000)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load MATLAB data
    mat_data = loadmat(str(file_path))
    
    # Extract time vector and scan data
    # x: time vector, y: scan data matrix
    time_vec = mat_data['x'].flatten()  # Shape: (n_time,)
    scan_data = mat_data['y']           # Shape: (n_points^2, n_time)
    
    # Calculate sampling frequency
    dt = time_vec[1] - time_vec[0]
    fs = 1.0 / dt
    
    # Calculate spatial coordinates
    x_coords = np.arange(n_points) * spacing
    y_coords = np.arange(n_points) * spacing
    
    # Reshape scan data to 3D: (n_points, n_points, n_time)
    # MATLAB uses column-major order, so we iterate by columns
    n_time = len(time_vec)
    data_xyt = np.zeros((n_points, n_points, n_time), dtype=np.float32)
    
    for col in range(n_points):
        start_idx = col * n_points
        end_idx = (col + 1) * n_points
        # col_data shape: (n_points, n_time)
        col_data = scan_data[start_idx:end_idx, :]
        data_xyt[:, col, :] = col_data
    
    return {
        'data_xyt': data_xyt,
        'time': time_vec,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'fs': fs
    }


def interpolate_target_grid(
    data: np.ndarray,
    target_shape: Tuple[int, int] = (41, 41),
    order: int = 3
) -> np.ndarray:
    """
    Resample spatial grid to target shape using bicubic interpolation.
    
    Supports both upsampling (e.g., 11×11 → 41×41) and downsampling (e.g., 51×51 → 41×41).
    
    Physics Rationale:
    - Bicubic (order=3) interpolation preserves wave smoothness
    - For downsampling from higher-resolution data (51×51), this maintains signal fidelity
    - For upsampling from lower-resolution data (11×11), this creates smooth transitions
    
    Args:
        data: Input array with shape (src_h, src_w, n_time)
        target_shape: Target spatial dimensions (default: (41, 41))
        order: Interpolation order (0=nearest, 1=linear, 3=bicubic)
    
    Returns:
        Interpolated array with shape (target_shape[0], target_shape[1], n_time)
    
    Example:
        >>> # Upsampling: 11×11 → 41×41
        >>> small = np.random.randn(11, 11, 1024)
        >>> large = interpolate_target_grid(small, (41, 41))
        >>> print(large.shape)  # (41, 41, 1024)
        
        >>> # Downsampling: 51×51 → 41×41
        >>> big = np.random.randn(51, 51, 1024)
        >>> matched = interpolate_target_grid(big, (41, 41))
        >>> print(matched.shape)  # (41, 41, 1024)
    """
    src_h, src_w, n_time = data.shape
    tgt_h, tgt_w = target_shape
    
    # Calculate zoom factors for each dimension
    # No interpolation on time axis (factor = 1.0)
    zoom_factors = (tgt_h / src_h, tgt_w / src_w, 1.0)
    
    # Log interpolation type for debugging
    if src_h < tgt_h:
        interp_type = "upsampling"
    elif src_h > tgt_h:
        interp_type = "downsampling"
    else:
        interp_type = "no spatial change"
    
    # Use specified order for interpolation
    # order=0: nearest, order=1: linear, order=3: bicubic
    interpolated = zoom(data, zoom_factors, order=order)
    
    # Ensure exact target shape (zoom may produce slightly different sizes)
    interpolated = interpolated[:tgt_h, :tgt_w, :]
    
    return interpolated.astype(np.float32)


def normalize_signal(
    signal: np.ndarray,
    method: str = 'zscore'
) -> Tuple[np.ndarray, dict]:
    """
    Normalize signal for neural network training.
    
    Args:
        signal: Input signal array
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        Tuple of (normalized_signal, normalization_params)
        - normalization_params can be used for denormalization
    
    Example:
        >>> sig = np.random.randn(1024)
        >>> norm_sig, params = normalize_signal(sig, method='zscore')
        >>> print(norm_sig.mean())  # ≈ 0
    """
    if method == 'zscore':
        mean = signal.mean()
        std = signal.std()
        if std < 1e-8:
            std = 1.0  # Avoid division by zero
        normalized = (signal - mean) / std
        params = {'method': 'zscore', 'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = signal.min()
        max_val = signal.max()
        range_val = max_val - min_val
        if range_val < 1e-8:
            range_val = 1.0
        normalized = (signal - min_val) / range_val
        params = {'method': 'minmax', 'min': min_val, 'range': range_val}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32), params


def pad_or_crop_signal(
    signal: np.ndarray,
    target_length: int = 1024
) -> np.ndarray:
    """
    Pad or crop time-series to fixed length.
    
    Signal Params: 160μs × 6.25MHz ≈ 1000 points
    We use 1024 for efficient FFT computation (power of 2).
    
    Args:
        signal: Input signal of shape (..., n_time)
        target_length: Target time dimension length
    
    Returns:
        Signal with shape (..., target_length)
    """
    current_length = signal.shape[-1]
    
    if current_length == target_length:
        return signal
    
    elif current_length > target_length:
        # Crop from the beginning (keep early part of signal)
        return signal[..., :target_length]
    
    else:
        # Zero-pad at the end
        pad_width = [(0, 0)] * (signal.ndim - 1) + [(0, target_length - current_length)]
        return np.pad(signal, pad_width, mode='constant', constant_values=0)


if __name__ == "__main__":
    # Quick test with actual data paths
    from pathlib import Path
    
    print("=" * 60)
    print("@DataAgent - Data Loader Test")
    print("=" * 60)
    
    # Test paths (update if needed)
    noisy_path = Path(r"E:\数据\260126\DnCNN\raw\41_41.mat")
    averaged_path = Path(r"E:\数据\260126\DnCNN\averaged\11_11.mat")
    
    if noisy_path.exists():
        print(f"\n[✓] Loading noisy data: {noisy_path}")
        noisy_data = load_mat_file(noisy_path, n_points=41)
        print(f"    Shape: {noisy_data['data_xyt'].shape}")
        print(f"    Sampling rate: {noisy_data['fs']/1e6:.2f} MHz")
        print(f"    Time range: {noisy_data['time'][0]*1e6:.2f} - {noisy_data['time'][-1]*1e6:.2f} μs")
    else:
        print(f"\n[!] Noisy data file not found: {noisy_path}")
    
    if averaged_path.exists():
        print(f"\n[✓] Loading averaged data: {averaged_path}")
        avg_data = load_mat_file(averaged_path, n_points=11)
        print(f"    Shape: {avg_data['data_xyt'].shape}")
        
        # Test interpolation
        print("\n[✓] Testing bicubic interpolation (11×11 → 41×41)...")
        interpolated = interpolate_target_grid(avg_data['data_xyt'], (41, 41))
        print(f"    Interpolated shape: {interpolated.shape}")
    else:
        print(f"\n[!] Averaged data file not found: {averaged_path}")
    
    print("\n" + "=" * 60)
    print("@DataAgent - Test Complete")
    print("=" * 60)
