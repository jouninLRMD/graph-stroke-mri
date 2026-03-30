"""Atlas registration and inter-modal alignment utilities."""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

logger = logging.getLogger(__name__)


def register_atlas_to_subject(
    atlas_img: nib.Nifti1Image,
    reference_img: nib.Nifti1Image,
) -> np.ndarray:
    """Register an atlas image to a subject's reference space.

    Uses nearest-neighbor interpolation to preserve integer labels.

    Args:
        atlas_img: Atlas NIfTI image (e.g., ArterialAtlas136).
        reference_img: Reference NIfTI image (typically T1w).

    Returns:
        Integer atlas data array in the reference image's voxel grid.
    """
    registered = resample_to_img(atlas_img, reference_img, interpolation="nearest")
    atlas_data = registered.get_fdata(dtype=np.float32).astype(np.int16)
    logger.info(
        "Registered atlas to subject space: %s, %d unique regions",
        atlas_data.shape,
        len(np.unique(atlas_data)),
    )
    return atlas_data


def register_modalities_to_reference(
    modalities: dict[str, nib.Nifti1Image],
    reference_img: nib.Nifti1Image,
    exclude: str = "T1",
) -> dict[str, nib.Nifti1Image]:
    """Align all modality images to a common reference space.

    Args:
        modalities: Dict of modality_name -> NIfTI image.
        reference_img: Reference image (e.g., T1w) to align to.
        exclude: Modality name that is already in the reference space.

    Returns:
        Dict of modality_name -> registered NIfTI image.
    """
    registered = {}
    for name, img in modalities.items():
        if name == exclude:
            registered[name] = img
        else:
            registered[name] = resample_to_img(img, reference_img, interpolation="linear")
            logger.debug("Registered %s to reference space", name)
    return registered


def register_masks_to_reference(
    masks: dict[str, np.ndarray],
    mask_affine: np.ndarray,
    reference_img: nib.Nifti1Image,
) -> dict[str, np.ndarray]:
    """Register binary masks to the reference image space.

    Uses nearest-neighbor interpolation to preserve binary values.

    Args:
        masks: Dict of mask_name -> binary numpy array.
        mask_affine: Affine matrix of the mask source space.
        reference_img: Reference NIfTI image for target space.

    Returns:
        Dict of mask_name -> registered binary numpy array.
    """
    registered = {}
    for name, mask_data in masks.items():
        if mask_data.size == 0:
            registered[name] = np.zeros(reference_img.shape[:3], dtype=np.uint8)
            continue
        mask_img = nib.Nifti1Image(mask_data.astype(np.float32), mask_affine)
        reg = resample_to_img(mask_img, reference_img, interpolation="nearest")
        registered[name] = (reg.get_fdata() > 0).astype(np.uint8)
    return registered
