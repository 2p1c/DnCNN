"""
@ModelAgent - DenoiseNet: 1D CNN Autoencoder for UGW Denoising
==============================================================

Architecture: Residual learning strategy where the model predicts noise,
and the clean signal is obtained by subtracting noise from input.

Network Topology:
    Input (1, 1024) → Encoder → Bottleneck → Decoder → Noise (1, 1024)
    Clean = Input - Noise
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv1d -> BatchNorm -> ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride (default: 1)
        padding: Padding mode (default: 'same')
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = 'same'
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, in_channels, length]
        x = self.conv(x)   # -> [batch, out_channels, length]
        x = self.bn(x)     # -> [batch, out_channels, length]
        x = self.relu(x)   # -> [batch, out_channels, length]
        return x


class DenoiseNet(nn.Module):
    """
    1D Convolutional Autoencoder with residual learning for denoising.
    
    The model learns to predict the noise component, which is then
    subtracted from the input to obtain the denoised signal.
    
    Architecture:
        Encoder: Conv1d(1→64, k=7) → MaxPool(2) → Conv1d(64→128, k=5) → MaxPool(2)
        Decoder: ConvTranspose1d(128→64) → ConvTranspose1d(64→1)
    
    Args:
        in_channels: Input channels (default: 1 for single signal)
        hidden_channels: List of hidden channel sizes (default: [64, 128])
        kernel_sizes: List of kernel sizes for encoder (default: [7, 5])
    
    Example:
        >>> model = DenoiseNet()
        >>> x = torch.randn(16, 1, 1024)  # [batch, channel, length]
        >>> denoised = model(x)
        >>> print(denoised.shape)  # torch.Size([16, 1, 1024])
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: List[int] = None,
        kernel_sizes: List[int] = None
    ):
        super().__init__()
        
        # Default architecture
        if hidden_channels is None:
            hidden_channels = [64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5]
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        
        # ============ ENCODER ============
        # Build encoder layers dynamically
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # First layer: input -> hidden[0]
        self.encoders.append(ConvBlock(in_channels, hidden_channels[0], kernel_sizes[0]))
        self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        # Subsequent layers: hidden[i-1] -> hidden[i]
        for i in range(1, self.num_layers):
            k_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            self.encoders.append(ConvBlock(hidden_channels[i-1], hidden_channels[i], k_size))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        # ============ BOTTLENECK ============
        self.bottleneck = ConvBlock(hidden_channels[-1], hidden_channels[-1], 3)
        
        # ============ DECODER ============
        # Build decoder layers dynamically (reverse order)
        self.decoders = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()
        self.decoder_relus = nn.ModuleList()
        
        # Reverse decoder: hidden[-1] -> ... -> hidden[0]
        for i in range(self.num_layers - 1, 0, -1):
            self.decoders.append(
                nn.ConvTranspose1d(
                    hidden_channels[i], hidden_channels[i-1],
                    kernel_size=4, stride=2, padding=1
                )
            )
            self.decoder_bns.append(nn.BatchNorm1d(hidden_channels[i-1]))
            self.decoder_relus.append(nn.ReLU(inplace=True))
        
        # Final layer: hidden[0] -> output
        self.final_layer = nn.ConvTranspose1d(
            hidden_channels[0], in_channels,
            kernel_size=4, stride=2, padding=1
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward pass (dynamic layers)."""
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            x = pool(x)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder forward pass (dynamic layers)."""
        x = self.bottleneck(x)
        
        # Upsample through decoder layers
        for decoder, bn, relu in zip(self.decoders, self.decoder_bns, self.decoder_relus):
            x = decoder(x)
            x = bn(x)
            x = relu(x)
        
        # Final output layer
        x = self.final_layer(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual learning.
        
        The model predicts noise, then subtracts it from input.
        
        Args:
            x: Noisy input signal, shape [batch, 1, length]
        
        Returns:
            Denoised signal, shape [batch, 1, length]
        """
        # Encode and decode to get noise estimate
        latent = self.encode(x)
        noise = self.decode(latent)
        
        # Residual learning: clean = noisy - noise
        denoised = x - noise
        
        return denoised
    
    def get_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Get the estimated noise component."""
        latent = self.encode(x)
        noise = self.decode(latent)
        return noise


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("@ModelAgent - DenoiseNet Architecture Test")
    print("=" * 60)
    
    # Create model
    model = DenoiseNet()
    print(f"\n[✓] Model created")
    print(f"    Trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 1, 1024)
    print(f"\n[✓] Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        denoised = model(x)
        noise = model.get_noise(x)
    
    print(f"[✓] Denoised output shape: {denoised.shape}")
    print(f"[✓] Noise estimate shape: {noise.shape}")
    
    # Verify residual learning
    residual_check = torch.allclose(x - noise, denoised)
    print(f"[✓] Residual learning check: {'PASS' if residual_check else 'FAIL'}")
    
    # Print architecture summary
    print(f"\n" + "-" * 40)
    print("Architecture Summary:")
    print("-" * 40)
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")
    
    print("\n" + "=" * 60)
    print("@ModelAgent - Test Complete")
    print("=" * 60)
