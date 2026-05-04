"""Image transformation utilities: normalization, resampling, downsampling."""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def normalize_image(data: np.ndarray) -> np.ndarray:
    """Min-max normalize an image to [0, 1].

    Args:
        data: Input image array.

    Returns:
        Normalized float32 array.
    """
    data = data.astype(np.float32)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val > 1e-6:
        return (data - min_val) / (max_val - min_val)
    return data - min_val


def resample_to_reference(
    source: nib.Nifti1Image,
    reference: nib.Nifti1Image,
    interpolation: str = "linear",
) -> nib.Nifti1Image:
    """Resample a source image to the voxel grid of a reference image.

    Args:
        source: Source NIfTI image to resample.
        reference: Reference NIfTI image defining the target grid.
        interpolation: Interpolation method ('linear', 'nearest', 'continuous').

    Returns:
        Resampled NIfTI image aligned to the reference grid.
    """
    return resample_to_img(source, reference, interpolation=interpolation)


def downsample_mri(data: np.ndarray, scale_factor: float) -> np.ndarray:
    """Downsample a 3D volume by a given scale factor using linear interpolation.

    Args:
        data: 3D image array.
        scale_factor: Scaling factor (e.g., 0.5 for half resolution).

    Returns:
        Downsampled array.
    """
    zoom_factors = [scale_factor] * data.ndim
    return zoom(data, zoom_factors, order=1).astype(data.dtype)


def compute_brain_mask(data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Compute a simple brain mask by thresholding normalized intensity.

    Args:
        data: Normalized 3D brain volume.
        threshold: Intensity threshold for mask.

    Returns:
        Boolean mask array.
    """
    return data > threshold
