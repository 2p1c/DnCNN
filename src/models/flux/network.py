"""
@Antigravity.Flux - ResNet-backed 2D U-Net for Time-Frequency Processing
========================================================================

A 2D Convolutional Neural Network operating on CWT spectrograms.
Input: (Batch, 2, Scales, Time) where 2 = [Real, Imag]
Output: (Batch, 2, Scales, Time)

Architecture: Encoder-Bottleneck-Decoder with Skip Connections.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """
    Residual Block for 2D feature maps.
    
    Structure: Conv → BN → ReLU → Conv → BN + Skip → ReLU
    
    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size (default: 3)
    """
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Skip connection
        out = self.relu(out)
        
        return out


class EncoderBlock(nn.Module):
    """
    Encoder block: Conv(stride=2 for downsampling) → ResBlock.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        num_res_blocks: Number of residual blocks (default: 2)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2
    ):
        super().__init__()
        
        # Downsampling convolution
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResBlock2D(out_channels) for _ in range(num_res_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.res_blocks(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample → Concat Skip → Conv → ResBlock.
    
    Args:
        in_channels: Input channels (from previous decoder layer)
        skip_channels: Skip connection channels (from encoder)
        out_channels: Output channels
        num_res_blocks: Number of residual blocks (default: 2)
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_res_blocks: int = 2
    ):
        super().__init__()
        
        # Upsampling via transposed convolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        
        # Combine skip connection
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResBlock2D(out_channels) for _ in range(num_res_blocks)
        ])
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch due to pooling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.combine(x)
        x = self.res_blocks(x)
        
        return x


class FluxUNet(nn.Module):
    """
    ResNet-backed 2D U-Net for time-frequency domain denoising.
    
    Processes CWT spectrograms with 2 channels (Real/Imag).
    Uses skip connections to preserve high-frequency details.
    
    Architecture:
        Encoder: 2→64→128→256 with downsampling
        Bottleneck: 256→256
        Decoder: 256→128→64→2 with upsampling + skip connections
        
    Args:
        in_channels: Input channels (default: 2 for Real/Imag)
        base_channels: Base channel count (default: 64)
        num_res_blocks: Residual blocks per encoder/decoder stage
        
    Example:
        >>> model = FluxUNet()
        >>> x = torch.randn(4, 2, 64, 1024)  # [B, C, H, W]
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([4, 2, 64, 1024])
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_res_blocks: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        c = base_channels  # Shorthand
        
        # ============ INPUT PROJECTION ============
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        
        # ============ ENCODER ============
        self.enc1 = EncoderBlock(c, c, num_res_blocks)       # 64 → 64, /2
        self.enc2 = EncoderBlock(c, c * 2, num_res_blocks)   # 64 → 128, /2
        self.enc3 = EncoderBlock(c * 2, c * 4, num_res_blocks)  # 128 → 256, /2
        
        # ============ BOTTLENECK ============
        self.bottleneck = nn.Sequential(
            ResBlock2D(c * 4),
            ResBlock2D(c * 4)
        )
        
        # ============ DECODER ============
        self.dec3 = DecoderBlock(c * 4, c * 4, c * 2, num_res_blocks)  # 256+256 → 128
        self.dec2 = DecoderBlock(c * 2, c * 2, c, num_res_blocks)      # 128+128 → 64
        self.dec1 = DecoderBlock(c, c, c, num_res_blocks)              # 64+64 → 64
        
        # ============ OUTPUT PROJECTION ============
        self.output_proj = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, in_channels, kernel_size=1, bias=True)  # 64 → 2
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor, shape [B, 2, H, W]
            
        Returns:
            Output tensor, shape [B, 2, H, W]
        """
        # Input projection
        x0 = self.input_proj(x)  # [B, 64, H, W]
        
        # Encoder path (save for skip connections)
        x1 = self.enc1(x0)  # [B, 64, H/2, W/2]
        x2 = self.enc2(x1)  # [B, 128, H/4, W/4]
        x3 = self.enc3(x2)  # [B, 256, H/8, W/8]
        
        # Bottleneck
        x_bottle = self.bottleneck(x3)  # [B, 256, H/8, W/8]
        
        # Decoder path with skip connections
        d3 = self.dec3(x_bottle, x3)  # [B, 128, H/4, W/4]
        d2 = self.dec2(d3, x2)        # [B, 64, H/2, W/2]
        d1 = self.dec1(d2, x1)        # [B, 64, H, W]
        
        # Handle final size to match input
        if d1.shape[2:] != x0.shape[2:]:
            d1 = F.interpolate(d1, size=x0.shape[2:], mode='bilinear', align_corners=False)
        
        # Output projection
        out = self.output_proj(d1)  # [B, 2, H, W]
        
        return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("@Antigravity.Flux - FluxUNet Architecture Test")
    print("=" * 60)
    
    # Create model
    model = FluxUNet(in_channels=2, base_channels=64)
    print(f"\n[OK] Model created")
    print(f"    Trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass with typical CWT output shape
    batch_size = 4
    num_scales = 64
    time_length = 1024
    
    x = torch.randn(batch_size, 2, num_scales, time_length)
    print(f"\\n[OK] Input shape: {x.shape}")
    print(f"    Expected: [B, 2, Scales, Time] = [{batch_size}, 2, {num_scales}, {time_length}]")
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    print(f"\n[OK] Output shape: {y.shape}")
    print(f"    Expected: {x.shape}")
    
    # Verify output shape matches input
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    print(f"\n[OK] Shape verification: PASS")
    
    # Print architecture summary
    print(f"\n" + "-" * 40)
    print("Architecture Summary:")
    print("-" * 40)
    for name, module in model.named_children():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module}")
        else:
            print(f"  {name}: {module.__class__.__name__}")
    
    print("\n" + "=" * 60)
    print("@Antigravity.Flux - Test Complete")
    print("=" * 60)
