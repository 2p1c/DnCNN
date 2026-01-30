"""
@ModelAgent - DnCNN-ResNet: 2D CNN with Residual Learning for CWT Denoising
============================================================================

Architecture: DnCNN with ResNet blocks for learning noise patterns in 
time-frequency domain (CWT coefficients).

Network Topology:
    Input (1, T) → CWT → (1, Scales, T) → ResNet DnCNN → Noise CWT → 
    Denoise in TF domain → iCWT → Output (1, T)

Key Features:
- End-to-end training with CWT/iCWT layers
- ResNet blocks for deeper networks without vanishing gradients
- Residual learning: predict noise, then subtract from input
- CPU-optimized with reduced channel counts
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cwt_layer import CWTLayer, InverseCWTLayer


class ResBlock2D(nn.Module):
    """
    2D Residual Block for processing time-frequency representations.
    
    Structure:
        x → Conv2d → BN → ReLU → Conv2d → BN → (+x) → ReLU
    
    Args:
        channels: Number of input/output channels (must be same for skip connection)
        kernel_size: Convolution kernel size (default: 3)
    """
    
    def __init__(self, channels: int = 64, kernel_size: int = 3):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor, shape [B, C, H, W]
        
        Returns:
            Output tensor, shape [B, C, H, W]
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Skip connection
        out = self.relu(out)
        
        return out


class DnCNN_CWT(nn.Module):
    """
    DnCNN with CWT feature extraction for 1D signal denoising.
    
    Key insight: CWT magnitude loses phase information, making perfect
    reconstruction impossible. Instead, we use CWT as a feature extractor
    and directly output the 1D denoised signal.
    
    Architecture:
        1. CWT: Extract time-frequency features (no reconstruction needed)
        2. 2D ResNet: Process TF features to predict noise pattern
        3. 1D Decoder: Convert TF noise pattern to 1D noise signal
        4. Residual: Clean = Noisy - Noise
    
    Args:
        signal_length: Length of 1D input signal (default: 1024)
        num_scales: Number of CWT scales (frequency resolution, default: 64)
        channels: Number of feature channels (default: 64)
        num_blocks: Number of ResNet blocks (default: 8)
        fs: Sampling frequency for CWT (default: 6.25 MHz)
        f_min: Minimum frequency of interest (default: 50 kHz)
        f_max: Maximum frequency of interest (default: 500 kHz)
    """
    
    def __init__(
        self,
        signal_length: int = 1024,
        num_scales: int = 64,
        channels: int = 64,
        num_blocks: int = 8,
        fs: float = 6.25e6,
        f_min: float = 50e3,
        f_max: float = 500e3
    ):
        super().__init__()
        
        self.signal_length = signal_length
        self.num_scales = num_scales
        self.channels = channels
        self.num_blocks = num_blocks
        
        # ============ CWT FEATURE EXTRACTOR ============
        self.cwt = CWTLayer(
            signal_length=signal_length,
            num_scales=num_scales,
            fs=fs,
            f_min=f_min,
            f_max=f_max
        )
        
        # Normalize CWT output
        self.cwt_norm = nn.InstanceNorm2d(1, affine=True)
        
        # ============ 2D ENCODER (TF Domain) ============
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # ResNet blocks for TF feature processing
        self.res_blocks = nn.ModuleList([
            ResBlock2D(channels=channels, kernel_size=3)
            for _ in range(num_blocks)
        ])
        
        # ============ 1D DECODER (Direct Noise Prediction) ============
        # Collapse frequency dimension and output 1D noise
        # [B, channels, scales, T] → [B, channels * scales, T] → [B, 1, T]
        self.freq_collapse = nn.Conv2d(channels, channels, kernel_size=(num_scales, 1), padding=0)
        
        self.noise_decoder = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, 1, kernel_size=3, padding=1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Xavier initialization for final output layer (direct prediction)
        final_conv = self.noise_decoder[-1]
        nn.init.xavier_uniform_(final_conv.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Extract TF features, directly predict clean signal.
        
        Note: Changed from residual learning to direct prediction because
        laser and piezoelectric signals may have significant waveform differences,
        not just noise differences.
        
        Args:
            x: Noisy input signal, shape [batch, 1, length]
        
        Returns:
            Denoised signal, shape [batch, 1, length]
        """
        # Step 1: CWT feature extraction
        # [B, 1, T] → [B, 1, Scales, T]
        cwt_features = self.cwt(x)
        cwt_features = self.cwt_norm(cwt_features)
        
        # Step 2: 2D Encoder for TF features
        # [B, 1, Scales, T] → [B, channels, Scales, T]
        features = self.encoder(cwt_features)
        
        # Step 3: ResNet blocks
        for res_block in self.res_blocks:
            features = res_block(features)
        
        # Step 4: Collapse frequency dimension
        # [B, channels, Scales, T] → [B, channels, 1, T]
        features = self.freq_collapse(features)
        
        # [B, channels, 1, T] → [B, channels, T]
        features = features.squeeze(2)
        
        # Step 5: Predict clean signal directly
        # [B, channels, T] → [B, 1, T]
        clean_pred = self.noise_decoder(features)
        
        # Step 6: Add skip connection from input for stability
        # This helps the network learn residual when signals are similar
        output = clean_pred + 0.1 * x
        
        return output
    
    def get_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted noise component."""
        cwt_features = self.cwt(x)
        cwt_features = self.cwt_norm(cwt_features)
        features = self.encoder(cwt_features)
        for res_block in self.res_blocks:
            features = res_block(features)
        features = self.freq_collapse(features).squeeze(2)
        noise = self.noise_decoder(features)
        return noise
    



class DnCNN_CWT_Light(nn.Module):
    """
    Lightweight version of DnCNN_CWT for faster CPU training.
    
    Reduces computational cost by:
    - Using fewer channels (32 instead of 64)
    - Using fewer ResBlocks (4 instead of 8)
    - Using smaller CWT scales (32 instead of 64)
    
    Args:
        signal_length: Length of 1D input signal (default: 1024)
        num_scales: Number of CWT scales (default: 32)
        channels: Number of feature channels (default: 32)
        num_blocks: Number of ResNet blocks (default: 4)
    """
    
    def __init__(
        self,
        signal_length: int = 1024,
        num_scales: int = 32,
        channels: int = 32,
        num_blocks: int = 4,
        fs: float = 6.25e6,
        f_min: float = 50e3,
        f_max: float = 500e3
    ):
        super().__init__()
        
        # Use the full DnCNN_CWT with reduced parameters
        self.model = DnCNN_CWT(
            signal_length=signal_length,
            num_scales=num_scales,
            channels=channels,
            num_blocks=num_blocks,
            fs=fs,
            f_min=f_min,
            f_max=f_max
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_noise(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_noise(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("@ModelAgent - DnCNN-ResNet CWT Architecture Test")
    print("=" * 60)
    
    # Test ResBlock2D
    print("\n[1] Testing ResBlock2D...")
    res_block = ResBlock2D(channels=64)
    x = torch.randn(4, 64, 32, 128)
    y = res_block(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {y.shape}")
    assert x.shape == y.shape, "ResBlock should preserve shape"
    print("    ✓ ResBlock2D passed")
    
    # Test DnCNN_CWT
    print("\n[2] Testing DnCNN_CWT (full model)...")
    model = DnCNN_CWT(
        signal_length=1024,
        num_scales=64,
        channels=64,
        num_blocks=8
    )
    print(f"    Trainable parameters: {count_parameters(model):,}")
    
    x = torch.randn(4, 1, 1024)
    with torch.no_grad():
        y = model(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {y.shape}")
    assert y.shape == x.shape, "Output should match input shape"
    print("    ✓ DnCNN_CWT passed")
    
    # Test gradient flow
    print("\n[3] Testing gradient flow...")
    x = torch.randn(2, 1, 1024, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(f"    Gradient exists: {x.grad is not None}")
    print(f"    Gradient shape: {x.grad.shape}")
    print("    ✓ Gradient flow passed")
    
    # Test noise extraction
    print("\n[4] Testing noise extraction...")
    x = torch.randn(2, 1, 1024)
    with torch.no_grad():
        noise = model.get_noise(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Noise shape: {noise.shape}")
    print("    ✓ Noise extraction passed")
    
    # Test lightweight model
    print("\n[5] Testing DnCNN_CWT_Light...")
    light_model = DnCNN_CWT_Light(signal_length=1024)
    print(f"    Trainable parameters: {count_parameters(light_model):,}")
    x = torch.randn(4, 1, 1024)
    with torch.no_grad():
        y = light_model(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {y.shape}")
    print("    ✓ DnCNN_CWT_Light passed")
    
    print("\n" + "=" * 60)
    print("@ModelAgent - All Tests Passed!")
    print("=" * 60)
