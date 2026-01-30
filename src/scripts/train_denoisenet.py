"""
@Antigravity.ResidualDnCNN - Residual Learning Training Script
==============================================================

Residual learning approach for laser ultrasonic denoising:
  - Network learns to predict NOISE (laser - piezoelectric)
  - Denoised = Laser - Predicted_Noise
  - Loss computed on noise estimation OR denoised signal

Usage:
    uv run python src/main_residual_dncnn.py --overfit-test --samples 50 --epochs 50
    uv run python src/main_residual_dncnn.py --epochs 100
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path for direct script execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import UGWDenoiseDataset, create_train_val_split
from src.models.denoisenet_1d import DenoiseNet
from src.models.denoisenet_1d.network import count_parameters
from src.core.visualize import plot_training_history


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualTrainer:
    """
    Trainer for residual learning DnCNN.
    
    Key difference from direct mapping:
      - Network predicts: noise = input - target
      - Loss: MSE(predicted_noise, actual_noise)
      - Output: denoised = input - predicted_noise
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch with residual learning."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for noisy_1d, clean_1d in train_loader:
            # noisy_1d: laser signal (low SNR)
            # clean_1d: piezoelectric signal (high SNR, as ground truth)
            noisy_1d = noisy_1d.to(self.device)  # [B, 1, L]
            clean_1d = clean_1d.to(self.device)  # [B, 1, L]
            
            # Compute actual noise (what network should predict)
            actual_noise = noisy_1d - clean_1d  # [B, 1, L]
            
            # Get predicted noise from model
            predicted_noise = self.model.get_noise(noisy_1d)  # [B, 1, L]
            
            # Loss: how well does network predict the noise?
            loss = self.criterion(predicted_noise, actual_noise)
            
            # Also compute MSE on denoised signal for monitoring
            denoised = noisy_1d - predicted_noise
            mse = torch.mean((denoised - clean_1d) ** 2).item()
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse
            num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'mse': total_mse / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation with SNR computation."""
        self.model.eval()
        total_loss = 0.0
        total_snr = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for noisy_1d, clean_1d in val_loader:
                noisy_1d = noisy_1d.to(self.device)
                clean_1d = clean_1d.to(self.device)
                
                # Predict noise and compute denoised
                predicted_noise = self.model.get_noise(noisy_1d)
                denoised = noisy_1d - predicted_noise
                
                # Noise loss
                actual_noise = noisy_1d - clean_1d
                loss = self.criterion(predicted_noise, actual_noise)
                
                # SNR improvement
                signal_power = torch.mean(clean_1d ** 2, dim=-1)
                error_power = torch.mean((denoised - clean_1d) ** 2, dim=-1)
                snr_db = 10 * torch.log10(signal_power / (error_power + 1e-10))
                
                total_loss += loss.item() * noisy_1d.size(0)
                total_snr += snr_db.sum().item()
                num_samples += noisy_1d.size(0)
        
        return {
            'loss': total_loss / num_samples,
            'snr_db': total_snr / num_samples
        }


def main(
    config_path: str = 'configs/default.yaml',
    epochs: Optional[int] = None,
    overfit_test: bool = False,
    num_samples: Optional[int] = None,
    sanity_check: bool = False
):
    """Main training function."""
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = config['training'].get('device', 'cpu')
    print(f"\n[ResidualDnCNN] Device: {device}")
    
    seed_everything(42)
    
    # ============ DATASET SETUP ============
    print("\n[ResidualDnCNN] Loading dataset...")
    dataset = UGWDenoiseDataset(
        noisy_path=config['data']['noisy_path'],
        target_path=config['data']['target_path'],
        noisy_grid_size=config['data']['noisy_grid_size'],
        target_grid_size=config['data']['target_grid_size'],
        target_length=config['signal']['target_length'],
        normalization_type=config['data'].get('normalization_type', 'bilateral')
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/residual_dncnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Overfit test mode
    if overfit_test:
        num_samples = num_samples or config['overfit_test']['num_samples']
        epochs = epochs or config['overfit_test']['epochs']
        
        print(f"\n[ResidualDnCNN] [TEST] OVERFIT TEST MODE")
        print(f"    Samples: {num_samples}")
        print(f"    Epochs: {epochs}")
        
        indices = list(range(min(num_samples, len(dataset))))
        subset = Subset(dataset, indices)
        train_loader = DataLoader(
            subset,
            batch_size=min(config['training']['batch_size'], num_samples),
            shuffle=True,
            num_workers=0
        )
        val_loader = None
    else:
        epochs = epochs or config['training']['epochs']
        train_idx, val_idx = create_train_val_split(len(dataset))
        
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 0)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training'].get('num_workers', 0)
        )
        
        print(f"    Train samples: {len(train_idx)}")
        print(f"    Val samples: {len(val_idx)}")
    
    # ============ MODEL SETUP ============
    print("\n[ResidualDnCNN] Creating DenoiseNet model...")
    model = DenoiseNet(
        in_channels=config['model'].get('in_channels', 1),
        hidden_channels=config['model'].get('hidden_channels', [64, 128]),
        kernel_sizes=config['model'].get('kernel_sizes', [7, 5])
    )
    print(f"    Trainable parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Create trainer
    trainer = ResidualTrainer(model, criterion, optimizer, device)
    
    # ============ TRAINING LOOP ============
    print(f"\n[ResidualDnCNN] Starting training...")
    print("-" * 60)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        train_losses.append(train_metrics['total'])
        
        # Validate (if validation loader exists)
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
                'norm_params': dataset.get_normalization_params(),
                'model_type': 'residual_dncnn'
            }, checkpoint_dir / "best_model.pt")
            log_str += " [saved]"
        
        print(log_str)
    
    print("-" * 60)
    print(f"\n[ResidualDnCNN] Training complete!")
    print(f"    Best loss: {best_loss:.6f}")
    print(f"    Model saved: {checkpoint_dir / 'best_model.pt'}")
    
    # Save training history
    plot_training_history(
        train_losses,
        val_losses if val_losses else None,
        title="Residual DnCNN Training History",
        save_path=checkpoint_dir / "training_history.png"
    )
    print(f"[VizAgent] Figure saved: {checkpoint_dir / 'training_history.png'}")
    
    if overfit_test and best_loss > 0.1:
        print(f"\n[ResidualDnCNN] [WARN] OVERFIT TEST WARNING: Final loss {best_loss:.6f}")
    
    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual DnCNN Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--overfit-test', action='store_true',
                        help='Run overfit test on small subset')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples for overfit test')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity check only')
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        epochs=args.epochs,
        overfit_test=args.overfit_test,
        num_samples=args.samples,
        sanity_check=args.sanity_check
    )
