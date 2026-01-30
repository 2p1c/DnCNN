"""
@DataAgent - PyTorch Dataset for UGW Denoising
==============================================

Custom Dataset class that creates (noisy, clean) pairs for training.
"""

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .loader import (
    load_mat_file,
    interpolate_target_grid,
    normalize_signal,
    pad_or_crop_signal
)


class UGWDenoiseDataset(Dataset):
    """
    PyTorch Dataset for Ultrasonic Guided Wave denoising.
    
    Creates (noisy_signal, clean_signal) pairs where:
    - noisy_signal: Single-shot from input grid (e.g., 41×41)
    - clean_signal: Target signal interpolated to match input grid size
    
    Supports flexible grid sizes:
    - Upsampling: 11×11 target → 41×41 input grid
    - Downsampling: 51×51 target → 41×41 input grid
    
    Each sample is a 1D time-series with shape (1, target_length).
    
    Args:
        noisy_path: Path to noisy data .mat file
        target_path: Path to target (clean) data .mat file
        noisy_grid_size: Grid size of noisy data (default: 41)
        target_grid_size: Grid size of target data (default: 51)
        target_length: Fixed time-series length (default: 1024)
        normalize: Whether to normalize signals (default: True)
        indices: Optional list of spatial indices to use (for train/val split)
    
    Example:
        >>> # 51×51 piezoelectric target → 41×41 input
        >>> dataset = UGWDenoiseDataset(
        ...     noisy_path='raw/41_41.mat',
        ...     target_path='yadian200k/51_51.mat',
        ...     noisy_grid_size=41,
        ...     target_grid_size=51
        ... )
        >>> noisy, clean = dataset[0]
        >>> print(noisy.shape)  # torch.Size([1, 1024])
    """
    
    def __init__(
        self,
        noisy_path: Path | str,
        target_path: Path | str,
        noisy_grid_size: int = 41,
        target_grid_size: int = 51,
        target_length: int = 1024,
        normalization_type: str = 'instance',
        indices: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.target_length = target_length
        self.normalization_type = normalization_type
        self.normalize = True if normalization_type else False
        # Load noisy data (Input)
        print(f"[DataAgent] Loading noisy (input) data from: {noisy_path}")
        # First load with actual grid size to get raw data
        raw_noisy_data = load_mat_file(noisy_path, n_points=noisy_grid_size)
        raw_noisy_xyt = raw_noisy_data['data_xyt']
        self.fs = raw_noisy_data['fs']
        self.time = raw_noisy_data['time']
        
        # Load target data (Label)
        print(f"[DataAgent] Loading target (label) data from: {target_path}")
        raw_target_data = load_mat_file(target_path, n_points=target_grid_size)
        raw_target_xyt = raw_target_data['data_xyt']
        
        # Grid Alignment Logic
        # We need both input and target to be on the SAME grid for point-wise training
        # Strategy: Always align to the LARGER grid to preserve high-res details
        
        if noisy_grid_size < target_grid_size:
            # Case: Inverse Mapping (11x11 Input -> 51x51 Target)
            # Upsample Input to match Target
            print(f"[DataAgent] Upsampling Input {noisy_grid_size}×{noisy_grid_size} → "
                  f"{target_grid_size}×{target_grid_size} (bicubic)...")
            self.noisy_xyt = interpolate_target_grid(
                raw_noisy_xyt, 
                (target_grid_size, target_grid_size)
            )
            self.clean_xyt = raw_target_xyt
            # Update grid size for spatial indexing
            self.current_grid_size = target_grid_size
            
        elif noisy_grid_size > target_grid_size:
            # Case: Denoising High-Res (41x41 Input -> 11x11 Target)
            # Upsample Target to match Input
            print(f"[DataAgent] Upsampling Target {target_grid_size}×{target_grid_size} → "
                  f"{noisy_grid_size}×{noisy_grid_size} (bicubic)...")
            self.noisy_xyt = raw_noisy_xyt
            self.clean_xyt = interpolate_target_grid(
                raw_target_xyt,
                (noisy_grid_size, noisy_grid_size)
            )
            self.current_grid_size = noisy_grid_size
            
        else:
            # Case: Same grid size
            print(f"[DataAgent] Grids aligned ({noisy_grid_size}×{noisy_grid_size})")
            self.noisy_xyt = raw_noisy_xyt
            self.clean_xyt = raw_target_xyt
            self.current_grid_size = noisy_grid_size
        
        # Flatten spatial dimensions for indexing
        # Shape: (grid*grid, n_time)
        self.noisy_flat = self.noisy_xyt.reshape(-1, self.noisy_xyt.shape[-1])
        self.clean_flat = self.clean_xyt.reshape(-1, self.clean_xyt.shape[-1])
        
        # Apply time-series length adjustment
        self.noisy_flat = pad_or_crop_signal(self.noisy_flat, target_length)
        self.clean_flat = pad_or_crop_signal(self.clean_flat, target_length)
        
        # Set indices (for train/val split)
        total_samples = self.noisy_flat.shape[0]
        self.indices = indices if indices is not None else list(range(total_samples))
        
        # Pre-compute normalization parameters
        # Always compute global statistics for both datasets (for bilateral normalization)
        self.noisy_mean = self.noisy_flat.mean()
        self.noisy_std = self.noisy_flat.std()
        self.clean_mean = self.clean_flat.mean()
        self.clean_std = self.clean_flat.std()
        
        if normalization_type == 'global' or normalization_type == 'bilateral':
            print(f"[DataAgent] Input Stats  - Mean: {self.noisy_mean:.2e}, Std: {self.noisy_std:.2e}")
            print(f"[DataAgent] Target Stats - Mean: {self.clean_mean:.2e}, Std: {self.clean_std:.2e}")
        
        print(f"[DataAgent] Dataset ready: {len(self.indices)} samples, "
              f"time-series length: {target_length}, norm: {normalization_type}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a (noisy, clean) signal pair.
        
        Returns:
            Tuple of (noisy_signal, clean_signal), each with shape (1, target_length)
        """
        # Map to actual spatial index
        spatial_idx = self.indices[idx]
        
        # Get signals
        noisy = self.noisy_flat[spatial_idx].copy()  # (target_length,)
        clean = self.clean_flat[spatial_idx].copy()  # (target_length,)
        
        # Normalize based on strategy
        if self.normalization_type == 'global':
            # Same stats for both (legacy mode)
            noisy = (noisy - self.noisy_mean) / (self.noisy_std + 1e-8)
            clean = (clean - self.clean_mean) / (self.clean_std + 1e-8)
        elif self.normalization_type == 'bilateral':
            # Bilateral: each signal normalized by GLOBAL stats of its own dataset
            # This way network learns shape mapping, not amplitude scaling
            noisy = (noisy - self.noisy_mean) / (self.noisy_std + 1e-8)
            clean = (clean - self.clean_mean) / (self.clean_std + 1e-8)
        elif self.normalization_type == 'instance':
            # Instance normalization: (x - u) / s per sample
            noisy = (noisy - noisy.mean()) / (noisy.std() + 1e-20)
            clean = (clean - clean.mean()) / (clean.std() + 1e-20)
        
        # Convert to torch tensors with channel dimension
        # Shape: (1, target_length) for Conv1d input
        noisy_tensor = torch.from_numpy(noisy).float().unsqueeze(0)
        clean_tensor = torch.from_numpy(clean).float().unsqueeze(0)
        
        return noisy_tensor, clean_tensor
    
    def get_spatial_coords(self, idx: int) -> Tuple[int, int]:
        """
        Get the (row, col) spatial coordinates for a sample index.
        
        Useful for visualization.
        """
        spatial_idx = self.indices[idx]
        row = spatial_idx // self.current_grid_size
        col = spatial_idx % self.current_grid_size
        return row, col
    
    def get_normalization_params(self) -> dict:
        """
        Get normalization parameters for denormalization during inference.
        
        For bilateral normalization:
          - Inference input: (x - input_mean) / input_std
          - Inference output: pred * target_std + target_mean
        """
        return {
            'type': self.normalization_type,
            'input_mean': float(self.noisy_mean),
            'input_std': float(self.noisy_std),
            'target_mean': float(self.clean_mean),
            'target_std': float(self.clean_std)
        }


def create_train_val_split(
    total_samples: int,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Create train/validation index split.
    
    Args:
        total_samples: Total number of samples (41*41 = 1681)
        val_ratio: Fraction for validation (default: 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices)
    """
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    val_size = int(total_samples * val_ratio)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    
    return train_indices, val_indices


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print("@DataAgent - Dataset Test")
    print("=" * 60)
    
    # Test with actual data
    noisy_path = Path(r"E:\数据\260126\DnCNN\raw\41_41.mat")
    averaged_path = Path(r"E:\数据\260126\DnCNN\averaged\11_11.mat")
    
    if noisy_path.exists() and averaged_path.exists():
        # Create dataset
        dataset = UGWDenoiseDataset(noisy_path, averaged_path)
        
        # Test single sample
        noisy, clean = dataset[0]
        print(f"\n[✓] Sample shape: noisy={noisy.shape}, clean={clean.shape}")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        batch_noisy, batch_clean = next(iter(loader))
        print(f"[✓] Batch shape: noisy={batch_noisy.shape}, clean={batch_clean.shape}")
        
        # Test train/val split
        train_idx, val_idx = create_train_val_split(len(dataset))
        print(f"[✓] Train/Val split: {len(train_idx)}/{len(val_idx)} samples")
    else:
        print("[!] Data files not found. Please check paths.")
    
    print("\n" + "=" * 60)
    print("@DataAgent - Test Complete")
    print("=" * 60)
