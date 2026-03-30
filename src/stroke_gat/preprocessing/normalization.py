"""Intensity normalization methods for multi-modal MRI."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range using min-max scaling.

    Args:
        data: Input array.

    Returns:
        Normalized float32 array.
    """
    data = data.astype(np.float32)
    vmin, vmax = data.min(), data.max()
    if vmax - vmin > 1e-6:
        return (data - vmin) / (vmax - vmin)
    return np.zeros_like(data)


def z_score_normalize(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalize image intensities.

    Args:
        data: Input image array.
        mask: Optional brain mask. If provided, statistics are computed
              only within the mask.

    Returns:
        Z-score normalized float32 array.
    """
    data = data.astype(np.float32)
    if mask is not None:
        brain_voxels = data[mask > 0]
        mean = brain_voxels.mean()
        std = brain_voxels.std()
    else:
        mean = data.mean()
        std = data.std()

    if std < 1e-6:
        return np.zeros_like(data)
    return (data - mean) / std


def percentile_normalize(
    data: np.ndarray, low: float = 1.0, high: float = 99.0
) -> np.ndarray:
    """Clip and normalize to percentile range.

    Args:
        data: Input array.
        low: Lower percentile for clipping.
        high: Upper percentile for clipping.

    Returns:
        Clipped and normalized float32 array in [0, 1].
    """
    data = data.astype(np.float32)
    p_low = np.percentile(data, low)
    p_high = np.percentile(data, high)
    data = np.clip(data, p_low, p_high)
    if p_high - p_low > 1e-6:
        return (data - p_low) / (p_high - p_low)
    return np.zeros_like(data)
