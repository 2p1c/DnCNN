"""
@Antigravity.Core - Main Gravity Loop (Training & Inference)
============================================================

End-to-end training pipeline for Signal-to-Image-to-Signal Autoencoder:
1. Lift Off: 1D Signal → 2D CWT Spectrogram
2. Process: 2D U-Net denoising in time-frequency domain
3. Re-Entry: 2D Spectrogram → 1D Reconstructed Signal

Philosophy: "Zero-Boilerplate, Dimensional-Fluidity"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for direct script execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Optional, Tuple, List
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Antigravity modules
# Antigravity modules
from src.core.transmuters import DimensionalTransmuter, TransmuterMetadata
from src.models.flux import FluxUNet, CombinedTimeFreqLoss, TimeDomainLoss, count_parameters
from src.data import UGWDenoiseDataset, create_train_val_split
from src.core.visualize import plot_training_history, compute_metrics


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class GravityLoopTrainer:
    """
    Trainer for Signal-to-Image-to-Signal Autoencoder.
    
    Manages the full training loop with CWT preprocessing,
    2D U-Net forward pass, and optional ICWT reconstruction for validation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        transmuter: DimensionalTransmuter,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        apply_compression: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            model: FluxUNet model
            transmuter: DimensionalTransmuter for CWT/ICWT
            criterion: Loss function (CombinedTimeFreqLoss)
            optimizer: PyTorch optimizer
            device: Training device ('cpu' or 'cuda')
            apply_compression: Whether to apply log compression to CWT
        """
        self.model = model.to(device)
        self.transmuter = transmuter
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.apply_compression = apply_compression
        
    def _prepare_batch(
        self,
        noisy_1d: torch.Tensor,
        clean_1d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, TransmuterMetadata]:
        """
        Transform 1D signals to 2D spectrograms.
        
        Args:
            noisy_1d: Noisy signals, shape [B, 1, L]
            clean_1d: Clean signals, shape [B, 1, L]
            
        Returns:
            noisy_2d: CWT of noisy, shape [B, 2, S, L]
            clean_2d: CWT of clean, shape [B, 2, S, L]
            metadata: TransmuterMetadata for reconstruction
        """
        # Convert to numpy for CWT
        noisy_np = noisy_1d.cpu().numpy()
        clean_np = clean_1d.cpu().numpy()
        
        # Lift off to 2D
        noisy_2d, meta = self.transmuter.lift_off(noisy_np, apply_compression=self.apply_compression)
        clean_2d, _ = self.transmuter.lift_off(clean_np, apply_compression=self.apply_compression)
        
        # Convert back to torch
        noisy_2d = torch.from_numpy(noisy_2d).to(self.device)
        clean_2d = torch.from_numpy(clean_2d).to(self.device)
        
        return noisy_2d, clean_2d, meta
    
    def train_one_epoch(self, dataloader: DataLoader) -> dict:
        """
        Train for one epoch.
        
        Returns:
            Dict with average loss values
        """
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_spectral = 0.0
        num_batches = 0
        
        for noisy, clean in dataloader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Transform to 2D
            noisy_2d, clean_2d, _ = self._prepare_batch(noisy, clean)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_2d = self.model(noisy_2d)
            
            # Compute loss
            losses = self.criterion(pred_2d, clean_2d, return_components=True)
            loss = losses['total']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            total_spectral += losses['spectral'].item()
            num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'spectral': total_spectral / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> dict:
        """
        Validate model on validation set.
        
        Returns:
            Dict with average loss and SNR metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        snr_values = []
        num_batches = 0
        
        with torch.no_grad():
            for noisy, clean in dataloader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Transform to 2D
                noisy_2d, clean_2d, meta = self._prepare_batch(noisy, clean)
                
                # Forward pass
                pred_2d = self.model(noisy_2d)
                
                # Compute 2D loss
                loss = self.criterion(pred_2d, clean_2d)
                total_loss += loss.item()
                
                # Reconstruct 1D for SNR computation
                pred_np = pred_2d.cpu().numpy()
                pred_1d = self.transmuter.re_entry(pred_np, meta, decompress=self.apply_compression)
                pred_1d_torch = torch.from_numpy(pred_1d).to(self.device)
                
                # Compute SNR
                signal_power = torch.mean(clean ** 2)
                noise_power = torch.mean((clean - pred_1d_torch) ** 2)
                snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-12))
                snr_values.append(snr_db.item())
                
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'snr_db': np.mean(snr_values)
        }


def run_gravity_loop(
    config: dict,
    overfit_test: bool = False,
    num_samples: Optional[int] = None,
    epochs: Optional[int] = None
):
    """
    Main training loop for Signal-to-Image-to-Signal Autoencoder.
    
    Args:
        config: Configuration dictionary
        overfit_test: If True, run overfit test on small subset
        num_samples: Number of samples for overfit test
        epochs: Override number of epochs
    """
    device = config['training']['device']
    print(f"\n[AntigravityCore] Device: {device}")
    
    seed_everything(42)
    
    # ============ TRANSMUTER SETUP ============
    print("\n[AntigravityCore] Initializing DimensionalTransmuter...")
    transmuter = DimensionalTransmuter(
        wavelet='cmor1.5-1.0',
        num_scales=64,
        fs=float(config['signal']['sampling_rate']),
        freq_range=(
            float(config['signal'].get('bandpass_low', 50e3)),
            float(config['signal'].get('bandpass_high', 500e3))
        )
    )
    print(f"    Wavelet: cmor1.5-1.0")
    print(f"    Scales: 64")
    print(f"    Frequency range: {transmuter.freq_range}")
    
    # ============ DATASET SETUP ============
    print("\n[AntigravityCore] Loading dataset...")
    dataset = UGWDenoiseDataset(
        noisy_path=config['data']['noisy_path'],
        target_path=config['data']['target_path'],
        noisy_grid_size=config['data']['noisy_grid_size'],
        target_grid_size=config['data']['target_grid_size'],
        target_length=config['signal']['target_length'],
        normalization_type=config['data'].get('normalization_type', 'instance')
    )
    
    # Overfit test mode
    if overfit_test:
        num_samples = num_samples or config['overfit_test']['num_samples']
        epochs = epochs or config['overfit_test']['epochs']
        
        print(f"\n[AntigravityCore] [TEST] OVERFIT TEST MODE")
        print(f"    Samples: {num_samples}")
        print(f"    Epochs: {epochs}")
        
        indices = list(range(min(num_samples, len(dataset))))
        subset = Subset(dataset, indices)
        train_loader = DataLoader(
            subset,
            batch_size=min(config['training']['batch_size'], num_samples),
            shuffle=True,
            num_workers=config['training']['num_workers']
        )
        val_loader = None
    else:
        epochs = epochs or config['training']['epochs']
        train_idx, val_idx = create_train_val_split(len(dataset), val_ratio=0.2)
        
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers']
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers']
        )
        
        print(f"\n[AntigravityCore] Training samples: {len(train_idx)}")
        print(f"[AntigravityCore] Validation samples: {len(val_idx)}")
    
    # ============ MODEL SETUP ============
    print("\n[AntigravityCore] Creating FluxUNet model...")
    model = FluxUNet(
        in_channels=2,  # Real/Imag
        base_channels=64,
        num_res_blocks=2
    )
    print(f"    Trainable parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = CombinedTimeFreqLoss(alpha=1.0, beta=0.5, gamma=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create trainer
    trainer = GravityLoopTrainer(
        model=model,
        transmuter=transmuter,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        apply_compression=True
    )
    
    # ============ TRAINING LOOP ============
    print("\n[AntigravityCore] Starting Gravity Loop training...")
    print("-" * 60)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    checkpoint_dir = Path("checkpoints/gravity_loop")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = trainer.train_one_epoch(train_loader)
        train_losses.append(train_metrics['total'])
        
        # Validate
        if val_loader:
            val_metrics = trainer.validate(val_loader)
            val_losses.append(val_metrics['loss'])
            current_loss = val_metrics['loss']
            log_str = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_metrics['total']:.6f} (MSE: {train_metrics['mse']:.6f}) | "
                f"Val: {val_metrics['loss']:.6f} | SNR: {val_metrics['snr_db']:.2f} dB"
            )
        else:
            current_loss = train_metrics['total']
            log_str = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_metrics['total']:.6f} (MSE: {train_metrics['mse']:.6f})"
            )
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'transmuter_config': {
                    'wavelet': transmuter.wavelet,
                    'num_scales': transmuter.num_scales,
                    'fs': transmuter.fs,
                    'freq_range': transmuter.freq_range
                },
                'norm_params': dataset.get_normalization_params()  # For inference denormalization
            }, checkpoint_dir / "best_model.pt")
            log_str += " ✓ (saved)"
        
        print(log_str)
    
    print("-" * 60)
    print(f"\n[AntigravityCore] Training complete!")
    print(f"    Best loss: {best_loss:.6f}")
    print(f"    Model saved: {checkpoint_dir / 'best_model.pt'}")
    
    # Save training history
    plot_training_history(
        train_losses,
        val_losses if val_losses else None,
        title="Gravity Loop Training History",
        save_path=checkpoint_dir / "training_history.png"
    )
    
    # Overfit test verification
    if overfit_test:
        if train_losses[-1] < 0.1:
            print("\n[AntigravityCore] [PASS] OVERFIT TEST PASSED!")
        else:
            print(f"\n[AntigravityCore] [WARN] OVERFIT TEST WARNING: Final loss {train_losses[-1]:.6f}")
    
    return model, trainer, train_losses, val_losses


def run_sanity_check():
    """
    Run transmuter sanity check: lift_off → re_entry without neural network.
    """
    from src.core.transmuters import sanity_check
    mse, snr_db, correlation = sanity_check()
    return mse < 1e-2  # Relaxed threshold for CWT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal-to-Image-to-Signal Autoencoder Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--overfit-test", action="store_true",
                        help="Run overfit test on small subset")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples for overfit test")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run transmuter sanity check only")
    
    args = parser.parse_args()
    
    if args.sanity_check:
        print("\n[AntigravityCore] Running sanity check...")
        passed = run_sanity_check()
        if passed:
            print("\n[PASS] Sanity check PASSED!")
        else:
            print("\n[WARN] Sanity check WARNING: MSE threshold not met")
    else:
        config = load_config(Path(args.config))
        run_gravity_loop(
            config,
            overfit_test=args.overfit_test,
            num_samples=args.samples,
            epochs=args.epochs
        )
