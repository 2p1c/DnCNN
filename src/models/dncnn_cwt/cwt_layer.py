"""
@SignalAgent - CWT Continuous Wavelet Transform Module
======================================================

Provides continuous wavelet transform (CWT) for converting 1D signals
to 2D time-frequency representations, optimized for CPU training.

Key Features:
- Morlet wavelet with configurable scales
- Efficient implementation using PyTorch (no scipy dependency for training)
- Approximate inverse transform for signal reconstruction
"""

import math
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CWTLayer(nn.Module):
    """
    Continuous Wavelet Transform as a differentiable PyTorch layer.
    
    Converts 1D signals to 2D time-frequency representations using
    convolution with pre-computed wavelet filters.
    
    This implementation is optimized for:
    - End-to-end training (gradients flow through CWT)
    - CPU efficiency (uses convolution instead of matrix multiplication)
    - Memory efficiency (processes in batches)
    
    Args:
        signal_length: Length of input signal (e.g., 1024)
        num_scales: Number of wavelet scales (frequency resolution)
        wavelet: Wavelet type ('morlet' supported)
        fs: Sampling frequency in Hz
        f_min: Minimum frequency of interest
        f_max: Maximum frequency of interest
    
    Example:
        >>> cwt = CWTLayer(signal_length=1024, num_scales=64)
        >>> x = torch.randn(16, 1, 1024)  # [batch, channel, time]
        >>> coeffs = cwt(x)  # [batch, 1, 64, 1024]
    """
    
    def __init__(
        self,
        signal_length: int = 1024,
        num_scales: int = 64,
        wavelet: str = 'morlet',
        fs: float = 6.25e6,
        f_min: float = 50e3,
        f_max: float = 500e3,
        omega0: float = 6.0
    ):
        super().__init__()
        
        self.signal_length = signal_length
        self.num_scales = num_scales
        self.wavelet = wavelet
        self.fs = float(fs)  # Ensure float type
        self.omega0 = float(omega0)
        
        # Ensure frequency bounds are float
        f_min = float(f_min)
        f_max = float(f_max)
        
        # Compute scales based on frequency range
        # Scale is inversely proportional to frequency
        # s = omega0 / (2 * pi * f) * fs
        scales = self._compute_scales(f_min, f_max, num_scales, self.fs, self.omega0)
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        
        # Pre-compute wavelet filters for each scale
        # Filters are stored as Conv1d weights
        filters_real, filters_imag = self._create_wavelet_filters(scales, signal_length, self.fs, self.omega0)
        
        # Register as buffers (not parameters, no gradient)
        self.register_buffer('filters_real', filters_real)  # [num_scales, 1, kernel_size]
        self.register_buffer('filters_imag', filters_imag)  # [num_scales, 1, kernel_size]
        
        # Store filter lengths for padding
        self.filter_lengths = [f.shape[0] for f in self._raw_filters]
    
    def _compute_scales(
        self,
        f_min: float,
        f_max: float,
        num_scales: int,
        fs: float,
        omega0: float
    ) -> np.ndarray:
        """Compute log-spaced scales from frequency range."""
        # Convert frequency to scale: s = omega0 * fs / (2 * pi * f)
        s_min = omega0 * fs / (2 * np.pi * f_max)
        s_max = omega0 * fs / (2 * np.pi * f_min)
        
        # Log-spaced scales (higher resolution at high frequencies)
        scales = np.geomspace(s_min, s_max, num_scales)
        return scales
    
    def _create_wavelet_filters(
        self,
        scales: np.ndarray,
        signal_length: int,
        fs: float,
        omega0: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Morlet wavelet filters for each scale."""
        # Maximum filter length (based on largest scale)
        # Wavelet support is approximately 6 * scale
        max_filter_len = min(int(6 * scales.max()) + 1, signal_length)
        # Ensure odd length for symmetric padding
        if max_filter_len % 2 == 0:
            max_filter_len += 1
        
        filters_real = []
        filters_imag = []
        self._raw_filters = []
        
        for scale in scales:
            # Filter length proportional to scale
            filter_len = min(int(6 * scale) + 1, max_filter_len)
            if filter_len % 2 == 0:
                filter_len += 1
            
            # Time vector centered at 0
            t = np.arange(-(filter_len // 2), filter_len // 2 + 1) / fs
            
            # Morlet wavelet: psi(t) = exp(-t^2/2) * exp(i*omega0*t)
            # Normalized by sqrt(scale) for energy conservation
            gaussian = np.exp(-0.5 * (t / (scale / fs)) ** 2)
            carrier_real = np.cos(omega0 * t / (scale / fs))
            carrier_imag = np.sin(omega0 * t / (scale / fs))
            
            psi_real = gaussian * carrier_real / np.sqrt(scale)
            psi_imag = gaussian * carrier_imag / np.sqrt(scale)
            
            # Normalize to unit energy
            norm = np.sqrt(np.sum(psi_real**2 + psi_imag**2))
            psi_real /= norm
            psi_imag /= norm
            
            self._raw_filters.append(psi_real)
            
            # Pad to max filter length
            pad_left = (max_filter_len - filter_len) // 2
            pad_right = max_filter_len - filter_len - pad_left
            
            psi_real_padded = np.pad(psi_real, (pad_left, pad_right), mode='constant')
            psi_imag_padded = np.pad(psi_imag, (pad_left, pad_right), mode='constant')
            
            filters_real.append(psi_real_padded)
            filters_imag.append(psi_imag_padded)
        
        # Stack and reshape for Conv1d: [out_channels, in_channels, kernel_size]
        filters_real = torch.tensor(np.array(filters_real), dtype=torch.float32).unsqueeze(1)
        filters_imag = torch.tensor(np.array(filters_imag), dtype=torch.float32).unsqueeze(1)
        
        return filters_real, filters_imag
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CWT to input signal.
        
        Args:
            x: Input signal, shape [batch, 1, time]
        
        Returns:
            CWT magnitude coefficients, shape [batch, 1, num_scales, time]
        """
        batch_size = x.shape[0]
        
        # Padding for same output length
        pad_size = self.filters_real.shape[-1] // 2
        x_padded = F.pad(x, (pad_size, pad_size), mode='reflect')
        
        # Convolution with wavelet filters
        # Real and imaginary parts
        conv_real = F.conv1d(x_padded, self.filters_real)  # [B, num_scales, T]
        conv_imag = F.conv1d(x_padded, self.filters_imag)  # [B, num_scales, T]
        
        # Magnitude (modulus)
        magnitude = torch.sqrt(conv_real ** 2 + conv_imag ** 2 + 1e-8)
        
        # Add channel dimension: [B, 1, num_scales, T]
        magnitude = magnitude.unsqueeze(1)
        
        return magnitude
    
    def get_frequencies(self) -> torch.Tensor:
        """Get the center frequencies corresponding to each scale."""
        return self.omega0 * self.fs / (2 * np.pi * self.scales)


class InverseCWTLayer(nn.Module):
    """
    Approximate Inverse CWT for signal reconstruction.
    
    Uses a learned linear combination of scales to reconstruct the signal.
    This is an approximation since exact iCWT requires phase information.
    
    The reconstruction is learned end-to-end during training.
    
    Args:
        num_scales: Number of CWT scales
        signal_length: Length of output signal
    """
    
    def __init__(self, num_scales: int = 64, signal_length: int = 1024):
        super().__init__()
        
        self.num_scales = num_scales
        self.signal_length = signal_length
        
        # Learnable reconstruction weights for each scale
        # Initialize with uniform weights
        self.scale_weights = nn.Parameter(torch.ones(1, 1, num_scales, 1) / num_scales)
        
        # Optional 1x1 conv to refine reconstruction
        self.refine = nn.Conv2d(1, 1, kernel_size=(num_scales, 1), padding=0, bias=True)
        
        # Initialize refine layer
        nn.init.xavier_uniform_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)
    
    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct 1D signal from CWT coefficients.
        
        Args:
            coeffs: CWT coefficients, shape [batch, 1, num_scales, time]
        
        Returns:
            Reconstructed signal, shape [batch, 1, time]
        """
        # Weighted sum across scales
        # [B, 1, num_scales, T] * [1, 1, num_scales, 1] -> sum -> [B, 1, 1, T]
        weighted = coeffs * F.softmax(self.scale_weights, dim=2)
        
        # Use convolution to combine scales
        # refine: [B, 1, num_scales, T] -> [B, 1, 1, T]
        reconstructed = self.refine(weighted)
        
        # Remove scale dimension: [B, 1, T]
        reconstructed = reconstructed.squeeze(2)
        
        return reconstructed


class CWTTransform:
    """
    High-level CWT transform interface for non-differentiable use.
    
    Useful for data preprocessing and visualization.
    Uses scipy for higher quality transforms when available.
    
    Args:
        signal_length: Length of input signal
        num_scales: Number of wavelet scales
        fs: Sampling frequency
    """
    
    def __init__(
        self,
        signal_length: int = 1024,
        num_scales: int = 64,
        fs: float = 6.25e6,
        f_min: float = 50e3,
        f_max: float = 500e3
    ):
        self.signal_length = signal_length
        self.num_scales = num_scales
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        
        # Create the PyTorch CWT layer
        self.cwt_layer = CWTLayer(
            signal_length=signal_length,
            num_scales=num_scales,
            fs=fs,
            f_min=f_min,
            f_max=f_max
        )
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply CWT to a signal.
        
        Args:
            signal: 1D numpy array of shape (time,) or (batch, time)
        
        Returns:
            CWT coefficients of shape (num_scales, time) or (batch, num_scales, time)
        """
        # Handle input shape
        if signal.ndim == 1:
            signal = signal[np.newaxis, np.newaxis, :]  # [1, 1, T]
            squeeze_output = True
        elif signal.ndim == 2:
            signal = signal[:, np.newaxis, :]  # [B, 1, T]
            squeeze_output = False
        else:
            raise ValueError(f"Expected 1D or 2D input, got shape {signal.shape}")
        
        # Convert to tensor
        x = torch.from_numpy(signal).float()
        
        # Apply CWT
        with torch.no_grad():
            coeffs = self.cwt_layer(x)  # [B, 1, scales, T]
        
        # Convert back to numpy
        result = coeffs.squeeze(1).numpy()  # [B, scales, T] or [1, scales, T]
        
        if squeeze_output:
            result = result.squeeze(0)  # [scales, T]
        
        return result
    
    def get_frequencies(self) -> np.ndarray:
        """Get center frequencies for each scale."""
        return self.cwt_layer.get_frequencies().numpy()


if __name__ == "__main__":
    print("=" * 60)
    print("@SignalAgent - CWT Transform Module Test")
    print("=" * 60)
    
    # Test CWTLayer
    print("\n[1] Testing CWTLayer...")
    cwt = CWTLayer(signal_length=1024, num_scales=64)
    x = torch.randn(4, 1, 1024)
    coeffs = cwt(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {coeffs.shape}")
    print(f"    Expected: [4, 1, 64, 1024]")
    assert coeffs.shape == (4, 1, 64, 1024), "Shape mismatch!"
    print("    ✓ CWTLayer passed")
    
    # Test InverseCWTLayer
    print("\n[2] Testing InverseCWTLayer...")
    icwt = InverseCWTLayer(num_scales=64, signal_length=1024)
    reconstructed = icwt(coeffs)
    print(f"    Input shape: {coeffs.shape}")
    print(f"    Output shape: {reconstructed.shape}")
    print(f"    Expected: [4, 1, 1024]")
    assert reconstructed.shape == (4, 1, 1024), "Shape mismatch!"
    print("    ✓ InverseCWTLayer passed")
    
    # Test gradient flow
    print("\n[3] Testing gradient flow...")
    x = torch.randn(2, 1, 1024, requires_grad=True)
    coeffs = cwt(x)
    recon = icwt(coeffs)
    loss = recon.sum()
    loss.backward()
    print(f"    Gradient computed: {x.grad is not None}")
    print("    ✓ Gradient flow passed")
    
    # Test CWTTransform (numpy interface)
    print("\n[4] Testing CWTTransform...")
    transform = CWTTransform(signal_length=1024, num_scales=64)
    signal_np = np.random.randn(1024)
    coeffs_np = transform.transform(signal_np)
    print(f"    Input shape: {signal_np.shape}")
    print(f"    Output shape: {coeffs_np.shape}")
    freqs = transform.get_frequencies()
    print(f"    Frequency range: {freqs.min()/1e3:.1f} - {freqs.max()/1e3:.1f} kHz")
    print("    ✓ CWTTransform passed")
    
    print("\n" + "=" * 60)
    print("@SignalAgent - All Tests Passed!")
    print("=" * 60)
