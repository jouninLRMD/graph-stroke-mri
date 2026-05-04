"""NIfTI file I/O utilities for loading and saving brain MRI volumes."""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI image from disk.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Loaded NIfTI image.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    return nib.load(str(path))


def load_nifti_data(
    path: str | Path, dtype: type = np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI data array and affine matrix.

    Args:
        path: Path to NIfTI file.
        dtype: Data type for the returned array.

    Returns:
        Tuple of (data_array, affine_matrix).
    """
    img = load_nifti(path)
    data = img.get_fdata(dtype=dtype)
    return data, img.affine


def handle_4d_volume(data: np.ndarray) -> np.ndarray:
    """Collapse a 4D volume to 3D by averaging along the 4th dimension.

    Some DWI-derived maps (ADC, TRACE) may be stored as 4D volumes.
    This function averages across the last axis to produce a 3D volume.

    Args:
        data: 3D or 4D numpy array.

    Returns:
        3D numpy array.
    """
    if data.ndim == 4:
        logger.info("Collapsing 4D volume %s to 3D by averaging axis=3", data.shape)
        return np.mean(data, axis=3).astype(data.dtype)
    return data


def save_compressed_nifti(
    data: np.ndarray, affine: np.ndarray, path: str | Path
) -> None:
    """Save a NIfTI image with gzip compression.

    Args:
        data: 3D numpy array of voxel values.
        affine: 4x4 affine transformation matrix.
        path: Output path (should end in .nii.gz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.Nifti1Image(data, affine)

    if str(path).endswith(".nii.gz"):
        # Save uncompressed first, then compress
        uncompressed = path.with_suffix("").with_suffix(".nii")
        nib.save(img, str(uncompressed))
        with open(uncompressed, "rb") as f_in, gzip.open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        uncompressed.unlink()
    else:
        nib.save(img, str(path))

    logger.info("Saved NIfTI to %s", path)
