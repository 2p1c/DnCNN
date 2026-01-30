"""
@AntigravityCore - Training Script for UGW Denoising
====================================================

MVP training loop with:
- Overfit test capability (verify convergence on small subset)
- Automatic model checkpointing
- CPU-optimized training
"""

import argparse
from pathlib import Path
from typing import Optional
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import UGWDenoiseDataset, create_train_val_split
from src.model import DenoiseNet, count_parameters
from src.dncnn_resnet import DnCNN_CWT, DnCNN_CWT_Light
from src.visualize import plot_training_history, plot_signal_comparison, plot_psd_comparison, compute_metrics


def create_model(config: dict, device: str) -> nn.Module:
    """
    Create model based on configuration.
    
    Supports:
    - 'denoise_net': Original 1D CNN autoencoder
    - 'dncnn_cwt': DnCNN with CWT transform (2D ResNet)
    - 'dncnn_cwt_light': Lightweight version for faster CPU training
    """
    model_type = config['model'].get('type', 'denoise_net')
    
    if model_type == 'denoise_net':
        # Original 1D model
        model = DenoiseNet(
            in_channels=config['model']['in_channels'],
            hidden_channels=config['model']['hidden_channels'],
            kernel_sizes=config['model']['kernel_sizes']
        )
        print(f"    Model type: DenoiseNet (1D CNN)")
        
    elif model_type == 'dncnn_cwt':
        # New 2D DnCNN with CWT
        cwt_config = config.get('cwt', {})
        model = DnCNN_CWT(
            signal_length=config['signal']['target_length'],
            num_scales=cwt_config.get('num_scales', 64),
            channels=config['model'].get('channels', 64),
            num_blocks=config['model'].get('num_blocks', 8),
            fs=config['signal']['sampling_rate'],
            f_min=cwt_config.get('f_min', 50e3),
            f_max=cwt_config.get('f_max', 500e3)
        )
        print(f"    Model type: DnCNN_CWT (2D ResNet with CWT)")
        print(f"    CWT scales: {cwt_config.get('num_scales', 64)}")
        print(f"    ResNet blocks: {config['model'].get('num_blocks', 8)}")
        
    elif model_type == 'dncnn_cwt_light':
        # Lightweight version for faster CPU training
        cwt_config = config.get('cwt', {})
        model = DnCNN_CWT_Light(
            signal_length=config['signal']['target_length'],
            num_scales=cwt_config.get('num_scales', 32),
            channels=config['model'].get('channels', 32),
            num_blocks=config['model'].get('num_blocks', 4),
            fs=config['signal']['sampling_rate'],
            f_min=cwt_config.get('f_min', 50e3),
            f_max=cwt_config.get('f_max', 500e3)
        )
        print(f"    Model type: DnCNN_CWT_Light (CPU-optimized)")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Follows best practices for PyTorch reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for noisy, clean in dataloader:
        noisy = noisy.to(device)  # [B, 1, 1024]
        clean = clean.to(device)  # [B, 1, 1024]
        
        optimizer.zero_grad()
        
        # Forward pass
        denoised = model(noisy)  # [B, 1, 1024]
        
        # Time-domain MSE Loss
        mse_loss = criterion(denoised, clean)
        
        # Frequency-domain Loss (normalized by signal length)
        # FFT output is O(N), so we normalize by sqrt(N) for energy conservation
        signal_length = denoised.shape[-1]
        fft_pred = torch.fft.rfft(denoised, dim=-1) / signal_length
        fft_target = torch.fft.rfft(clean, dim=-1) / signal_length
        freq_loss = nn.functional.mse_loss(torch.abs(fft_pred), torch.abs(fft_target))
        
        # Combined loss with balanced weighting
        loss = mse_loss + 0.1 * freq_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Validate model on validation set.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            denoised = model(noisy)
            loss = criterion(denoised, clean)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def run_training(
    config: dict,
    overfit_test: bool = False,
    num_samples: Optional[int] = None,
    epochs: Optional[int] = None
):
    """
    Main training loop.
    
    Args:
        config: Configuration dictionary
        overfit_test: If True, run overfit test on small subset
        num_samples: Number of samples for overfit test
        epochs: Override number of epochs
    """
    # Device setup (CPU only for this project)
    device = config['training']['device']
    print(f"\n[AntigravityCore] Device: {device}")
    
    # Set seed for reproducibility
    seed_everything(42)
    
    # Load dataset
    print("\n[AntigravityCore] Loading dataset...")
    dataset = UGWDenoiseDataset(
        noisy_path=config['data']['noisy_path'],
        target_path=config['data']['target_path'],
        noisy_grid_size=config['data']['noisy_grid_size'],
        target_grid_size=config['data']['target_grid_size'],
        target_length=config['signal']['target_length'],
        normalization_type=config['data'].get('normalization_type', 'global')
    )
    
    # Overfit test mode: use small subset
    if overfit_test:
        num_samples = num_samples or config['overfit_test']['num_samples']
        epochs = epochs or config['overfit_test']['epochs']
        
        print(f"\n[AntigravityCore] 🧪 OVERFIT TEST MODE")
        print(f"    Samples: {num_samples}")
        print(f"    Epochs: {epochs}")
        
        # Use first N samples for overfit test
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
        
        # Train/val split
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
    
    # Create model
    print("\n[AntigravityCore] Creating model...")
    model = create_model(config, device)
    print(f"    Trainable parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=1e-4  # L2 regularization
    )
    
    # Learning rate scheduler: Cosine annealing for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=1e-5
    )
    
    # Training loop
    print("\n[AntigravityCore] Starting training...")
    print("-" * 50)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            current_loss = val_loss
            log_str = f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        else:
            current_loss = train_loss
            log_str = f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f}"
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'norm_params': dataset.get_normalization_params()
            }, checkpoint_path)
            log_str += " ✓ (saved)"
        
        print(log_str)
    
    print("-" * 50)
    print(f"\n[AntigravityCore] Training complete!")
    print(f"    Best loss: {best_loss:.6f}")
    print(f"    Model saved: {checkpoint_dir / 'best_model.pt'}")
    
    # Save training history plot
    fig = plot_training_history(
        train_losses,
        val_losses if val_losses else None,
        title="Training History",
        save_path=checkpoint_dir / "training_history.png"
    )
    
    # Overfit test success check
    if overfit_test:
        if train_losses[-1] < 0.01:
            print("\n[AntigravityCore] ✅ OVERFIT TEST PASSED!")
            print("    Model can successfully overfit on small subset.")
        else:
            print("\n[AntigravityCore] ⚠️ OVERFIT TEST WARNING")
            print(f"    Final loss {train_losses[-1]:.6f} > 0.01 threshold")
            print("    Model may have issues with architecture or data.")
    
    # ============ AUTO VISUALIZATION ============
    print("\n[AntigravityCore] Generating visualization...")
    visualize_samples(model, dataset, device, checkpoint_dir, num_samples=3)
    
    return model, train_losses, val_losses


def visualize_samples(
    model: nn.Module,
    dataset: UGWDenoiseDataset,
    device: str,
    save_dir: Path,
    num_samples: int = 3
):
    """
    Visualize denoising results on random samples.
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: Device for inference
        save_dir: Directory to save figures
        num_samples: Number of samples to visualize
    """
    import random
    model.eval()
    
    # Select random sample indices
    total_samples = len(dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"    Visualizing {len(sample_indices)} samples...")
    
    for i, sample_idx in enumerate(sample_indices):
        # Get sample
        noisy, clean = dataset[sample_idx]
        noisy = noisy.unsqueeze(0).to(device)  # [1, 1, 1024]
        clean = clean.unsqueeze(0).to(device)
        
        with torch.no_grad():
            denoised = model(noisy)
        
        # Convert to numpy
        raw_np = noisy.squeeze().cpu().numpy()
        gt_np = clean.squeeze().cpu().numpy()
        dn_np = denoised.squeeze().cpu().numpy()
        
        # Get spatial coordinates
        row, col = dataset.get_spatial_coords(sample_idx)
        
        # Plot comparison
        fig = plot_signal_comparison(
            raw_np, gt_np, dn_np,
            fs=dataset.fs,
            title=f"Denoising Result - Position ({row}, {col})",
            save_path=save_dir / f"sample_{i+1}_pos_{row}_{col}_comparison.png"
        )
        plt.close(fig)
        
        # Plot PSD
        fig2 = plot_psd_comparison(
            raw_np, gt_np, dn_np,
            fs=dataset.fs,
            title=f"PSD Analysis - Position ({row}, {col})",
            save_path=save_dir / f"sample_{i+1}_pos_{row}_{col}_psd.png"
        )
        plt.close(fig2)
        
        # Compute and print metrics
        metrics = compute_metrics(gt_np, dn_np)
        print(f"    Sample {i+1} ({row}, {col}): SNR={metrics['snr_db']:.2f}dB, Corr={metrics['correlation']:.4f}")
    
    print(f"    Figures saved to: {save_dir}")


def demo_inference(model: nn.Module, dataset: UGWDenoiseDataset, device: str, sample_idx: int = 0):
    """
    Run inference on a sample and visualize results.
    """
    model.eval()
    
    # Get sample
    noisy, clean = dataset[sample_idx]
    noisy = noisy.unsqueeze(0).to(device)  # [1, 1, 1024]
    clean = clean.unsqueeze(0).to(device)
    
    with torch.no_grad():
        denoised = model(noisy)
    
    # Convert to numpy
    raw_np = noisy.squeeze().cpu().numpy()
    gt_np = clean.squeeze().cpu().numpy()
    dn_np = denoised.squeeze().cpu().numpy()
    
    # Plot comparison
    row, col = dataset.get_spatial_coords(sample_idx)
    fig = plot_signal_comparison(
        raw_np, gt_np, dn_np,
        fs=dataset.fs,
        title=f"Denoising Result - Position ({row}, {col})",
        save_path=Path("checkpoints") / f"sample_{sample_idx}_comparison.png"
    )
    
    # Plot PSD
    fig2 = plot_psd_comparison(
        raw_np, gt_np, dn_np,
        fs=dataset.fs,
        title=f"PSD Analysis - Position ({row}, {col})",
        save_path=Path("checkpoints") / f"sample_{sample_idx}_psd.png"
    )
    
    # Compute metrics
    metrics = compute_metrics(gt_np, dn_np)
    print("\n[VizAgent] Denoising Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenoiseNet for UGW denoising")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--overfit-test", action="store_true",
                        help="Run overfit test on small subset")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples for overfit test")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo inference after training")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    
    # Run training
    model, train_losses, val_losses = run_training(
        config,
        overfit_test=args.overfit_test,
        num_samples=args.samples,
        epochs=args.epochs
    )
    
    # Demo inference
    if args.demo:
        print("\n[AntigravityCore] Running demo inference...")
        dataset = UGWDenoiseDataset(
            noisy_path=config['data']['noisy_path'],
            target_path=config['data']['target_path'],
            noisy_grid_size=config['data']['noisy_grid_size'],
            target_grid_size=config['data']['target_grid_size'],
            target_length=config['signal']['target_length'],
            normalize=True
        )
        demo_inference(model, dataset, config['training']['device'], sample_idx=0)
