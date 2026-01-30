"""
@Antigravity.Flux - Loss Functions for Time-Frequency Domain Training
=====================================================================

Combined loss functions for training on CWT spectrograms:
- MSELoss: Pixel-wise L2 loss in time-frequency domain
- SpectralConvergenceLoss: Frequency-domain fidelity
- SSIMLoss: Structural similarity for spectrogram quality
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss for frequency-domain fidelity.
    
    Measures the normalized difference in frequency content:
        L_sc = ||FFT(pred) - FFT(target)||_F / ||FFT(target)||_F
        
    This ensures the model preserves spectral characteristics.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral convergence loss.
        
        Args:
            pred: Predicted tensor, shape [B, C, H, W]
            target: Target tensor, shape [B, C, H, W]
            
        Returns:
            Scalar loss value
        """
        # 2D FFT on spatial dimensions
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Compute magnitude spectra
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Spectral convergence: normalized L2 error
        numerator = torch.norm(pred_mag - target_mag, p='fro')
        denominator = torch.norm(target_mag, p='fro') + self.eps
        
        return numerator / denominator


class LogMagnitudeLoss(nn.Module):
    """
    Log-Magnitude Loss for spectrogram comparison.
    
    Operates on log-scale magnitudes which is more perceptually relevant
    and handles large dynamic range in spectrograms.
    
        L_lm = ||log(|pred| + eps) - log(|target| + eps)||_1
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute log-magnitude loss.
        
        Args:
            pred: Predicted tensor, shape [B, C, H, W]
            target: Target tensor, shape [B, C, H, W]
            
        Returns:
            Scalar loss value
        """
        # Compute magnitude (for 2-channel Real/Imag input)
        if pred.shape[1] == 2:
            pred_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + self.eps)
            target_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + self.eps)
        else:
            pred_mag = torch.abs(pred)
            target_mag = torch.abs(target)
        
        # Log-scale comparison
        pred_log = torch.log(pred_mag + self.eps)
        target_log = torch.log(target_mag + self.eps)
        
        return F.l1_loss(pred_log, target_log)


class PhaseLoss(nn.Module):
    """
    Phase Consistency Loss for preserving phase information.
    
    Critical for accurate ICWT reconstruction.
        L_phase = 1 - cos(angle(pred) - angle(target))
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute phase loss.
        
        Args:
            pred: Predicted tensor, shape [B, 2, H, W] (Real/Imag)
            target: Target tensor, shape [B, 2, H, W]
            
        Returns:
            Scalar loss value
        """
        # Extract phase from Real/Imag representation
        pred_phase = torch.atan2(pred[:, 1], pred[:, 0])
        target_phase = torch.atan2(target[:, 1], target[:, 0])
        
        # Phase difference
        phase_diff = pred_phase - target_phase
        
        # Cosine distance (1 - cos for loss)
        loss = 1 - torch.cos(phase_diff)
        
        return loss.mean()


class CombinedTimeFreqLoss(nn.Module):
    """
    Combined Loss for Time-Frequency Domain Training.
    
    Combines multiple loss terms:
        L_total = α * L_mse + β * L_spectral + γ * L_phase
        
    Args:
        alpha: Weight for MSE loss (default: 1.0)
        beta: Weight for spectral convergence loss (default: 0.5)
        gamma: Weight for phase loss (default: 0.1)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralConvergenceLoss()
        self.phase_loss = PhaseLoss()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted tensor, shape [B, 2, H, W]
            target: Target tensor, shape [B, 2, H, W]
            return_components: If True, return dict with individual losses
            
        Returns:
            Total loss (or dict if return_components=True)
        """
        # Individual losses
        l_mse = self.mse_loss(pred, target)
        l_spectral = self.spectral_loss(pred, target)
        l_phase = self.phase_loss(pred, target)
        
        # Combined
        total = self.alpha * l_mse + self.beta * l_spectral + self.gamma * l_phase
        
        if return_components:
            return {
                'total': total,
                'mse': l_mse,
                'spectral': l_spectral,
                'phase': l_phase
            }
        
        return total


class TimeDomainLoss(nn.Module):
    """
    Time-Domain Loss for evaluating reconstructed 1D signals.
    
    Used during validation to assess final reconstruction quality
    after inverse CWT.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(
        self,
        pred_signal: torch.Tensor,
        target_signal: torch.Tensor
    ) -> dict:
        """
        Compute time-domain metrics.
        
        Args:
            pred_signal: Reconstructed signal, shape [B, 1, L]
            target_signal: Target signal, shape [B, 1, L]
            
        Returns:
            Dict with MSE and SNR metrics
        """
        mse = self.mse(pred_signal, target_signal)
        
        # SNR computation
        signal_power = torch.mean(target_signal ** 2)
        noise_power = torch.mean((target_signal - pred_signal) ** 2)
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-12))
        
        return {
            'mse': mse,
            'snr_db': snr_db
        }


if __name__ == "__main__":
    print("=" * 60)
    print("@Antigravity.Flux - Loss Functions Test")
    print("=" * 60)
    
    # Create test tensors
    batch_size = 4
    pred = torch.randn(batch_size, 2, 64, 1024)
    target = pred + 0.1 * torch.randn_like(pred)  # Small perturbation
    
    print(f"\n[✓] Test tensors created: {pred.shape}")
    
    # Test individual losses
    print("\n[Testing Individual Losses]")
    
    spectral_loss = SpectralConvergenceLoss()
    l_spectral = spectral_loss(pred, target)
    print(f"    SpectralConvergenceLoss: {l_spectral.item():.6f}")
    
    log_mag_loss = LogMagnitudeLoss()
    l_logmag = log_mag_loss(pred, target)
    print(f"    LogMagnitudeLoss: {l_logmag.item():.6f}")
    
    phase_loss = PhaseLoss()
    l_phase = phase_loss(pred, target)
    print(f"    PhaseLoss: {l_phase.item():.6f}")
    
    # Test combined loss
    print("\n[Testing Combined Loss]")
    combined_loss = CombinedTimeFreqLoss(alpha=1.0, beta=0.5, gamma=0.1)
    losses = combined_loss(pred, target, return_components=True)
    
    for name, value in losses.items():
        print(f"    {name}: {value.item():.6f}")
    
    print("\n" + "=" * 60)
    print("@Antigravity.Flux - Test Complete")
    print("=" * 60)
