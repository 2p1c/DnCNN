"""
@Antigravity.Transmuters - Wavelet Transform Module
===================================================

Provides dimensional lifting (1D → 2D) and re-entry (2D → 1D)
using Continuous Wavelet Transform.
"""

from .wavelet_core import DimensionalTransmuter, TransmuterMetadata, sanity_check

__all__ = [
    'DimensionalTransmuter',
    'TransmuterMetadata',
    'sanity_check'
]
