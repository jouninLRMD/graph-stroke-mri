"""DataService: central data access layer for SOOP MRI data.

Supports two data layouts:
1. Raw BIDS: sub-XXX/anat/sub-XXX_T1w.nii.gz, etc.
2. SOOP normalized flat: wsub-XXX_FLAIR.nii.gz, bwsrsub-XXX_lesion.nii.gz
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from stroke_gat.config import PathsConfig
from stroke_gat.data.nifti_io import handle_4d_volume, load_nifti, load_nifti_data
from stroke_gat.data.transforms import normalize_image, resample_to_reference

logger = logging.getLogger(__name__)

# Strict BIDS subject directory pattern: rejects "derivatives", "code",
# "sourcedata", "stimuli", "phenotype", participants.tsv, etc.
_SUBJECT_DIR_PATTERN = re.compile(r"^sub-[A-Za-z0-9]+$")


class DataService:
    """Stateless service for accessing SOOP MRI data.

    Provides a unified interface for loading multi-modal MRI data from
    either raw BIDS format or the pre-processed SOOP normalized layout.
    """

    def __init__(self, config: PathsConfig):
        self._config = config
        self._atlas_cache: Optional[tuple[np.ndarray, nib.Nifti1Image]] = None
        self._atlas_labels_cache: Optional[dict[int, str]] = None

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def discover_subjects_bids(self) -> list[str]:
        """Scan raw BIDS data directory and return sorted subject IDs.

        Only directories matching ``^sub-[A-Za-z0-9]+$`` are returned, so
        non-subject BIDS top-level entries (``derivatives/``, ``code/``,
        ``sourcedata/``, ``stimuli/``, ``phenotype/``, files like
        ``participants.tsv`` or ``dataset_description.json``) are excluded.
        """
        raw_path = Path(self._config.raw_bids)
        if not raw_path.exists():
            logger.warning("BIDS directory not found: %s", raw_path)
            return []

        subjects = sorted(
            d.name
            for d in raw_path.iterdir()
            if d.is_dir() and _SUBJECT_DIR_PATTERN.match(d.name)
        )
        logger.info("Discovered %d BIDS subjects", len(subjects))
        return subjects

    def discover_subjects_soop(self) -> list[str]:
        """Scan SOOP normalized directory for subject IDs from lesion masks."""
        soop_path = Path(self._config.soop_normalized)
        if not soop_path.exists():
            logger.warning("SOOP directory not found: %s", soop_path)
            return []

        pattern = re.compile(r"bwsrsub-(\d+)_lesion\.nii\.gz")
        subject_ids = sorted(
            {m.group(1) for f in soop_path.glob("bwsrsub-*_lesion.nii.gz") if (m := pattern.match(f.name))}
        )
        logger.info("Discovered %d SOOP subjects", len(subject_ids))
        return subject_ids

    # ------------------------------------------------------------------
    # BIDS data loading
    # ------------------------------------------------------------------

    def load_subject_modalities(
        self, subject_id: str
    ) -> dict[str, nib.Nifti1Image]:
        """Load all available MRI modalities for a BIDS subject.

        Args:
            subject_id: Subject directory name (e.g., 'sub-1000').

        Returns:
            Dict mapping modality name to NIfTI image.
            Keys: 'T1', 'FLAIR', 'ADC', 'TRACE' (present if file exists).
        """
        base = Path(self._config.raw_bids) / subject_id
        modality_paths = {
            "T1": base / "anat" / f"{subject_id}_T1w.nii.gz",
            "FLAIR": base / "anat" / f"{subject_id}_FLAIR.nii.gz",
            "ADC": base / "dwi" / f"{subject_id}_rec-ADC_dwi.nii.gz",
            "TRACE": base / "dwi" / f"{subject_id}_rec-TRACE_dwi.nii.gz",
        }

        modalities = {}
        missing: list[str] = []
        for name, path in modality_paths.items():
            if path.exists():
                modalities[name] = load_nifti(path)
                logger.debug("Loaded %s for %s", name, subject_id)
            else:
                missing.append(name)
                logger.debug("%s not found for %s: %s", name, subject_id, path)

        if missing:
            # Single structured line so validate_dataset.py can grep / parse it.
            logger.warning(
                "subject=%s missing_modalities=%s present_modalities=%s",
                subject_id,
                ",".join(missing) if missing else "-",
                ",".join(sorted(modalities.keys())) if modalities else "-",
            )

        return modalities

    def load_subject_masks(self, subject_id: str) -> dict[str, np.ndarray]:
        """Load lesion masks for a BIDS subject.

        Args:
            subject_id: Subject directory name (e.g., 'sub-1000').

        Returns:
            Dict with keys 'General', 'Acute', 'Chronic'. Missing masks are zero arrays.
        """
        deriv = Path(self._config.raw_bids) / "derivatives" / "lesion_masks" / subject_id / "dwi"
        mask_paths = {
            "General": deriv / f"{subject_id}_space-TRACE_desc-lesion_mask.nii.gz",
            "Acute": deriv / f"{subject_id}_space-TRACE_desc-lesionAcute_mask.nii.gz",
            "Chronic": deriv / f"{subject_id}_space-TRACE_desc-lesionChronic_mask.nii.gz",
        }

        masks: dict[str, np.ndarray] = {}
        ref_shape: Optional[tuple] = None

        for name, path in mask_paths.items():
            if path.exists():
                data, _ = load_nifti_data(path, dtype=np.float32)
                masks[name] = (data > 0).astype(np.uint8)
                ref_shape = data.shape
            else:
                logger.debug("%s mask not found for %s", name, subject_id)

        # Fill missing masks with zeros
        for name in mask_paths:
            if name not in masks:
                if ref_shape is not None:
                    masks[name] = np.zeros(ref_shape, dtype=np.uint8)
                else:
                    masks[name] = np.array([], dtype=np.uint8)

        return masks

    # ------------------------------------------------------------------
    # SOOP normalized data loading
    # ------------------------------------------------------------------

    def load_soop_normalized(
        self, subject_id: str
    ) -> tuple[nib.Nifti1Image, np.ndarray]:
        """Load warped FLAIR and lesion mask from SOOP normalized layout.

        Args:
            subject_id: Numeric subject ID (e.g., '1000').

        Returns:
            Tuple of (FLAIR NIfTI image, binary lesion mask array).
        """
        soop = Path(self._config.soop_normalized)
        flair_path = soop / f"wsub-{subject_id}_FLAIR.nii.gz"
        lesion_path = soop / f"bwsrsub-{subject_id}_lesion.nii.gz"

        flair_img = load_nifti(flair_path)
        lesion_data, _ = load_nifti_data(lesion_path, dtype=np.float32)
        lesion_mask = (lesion_data > 0).astype(np.uint8)

        return flair_img, lesion_mask

    # ------------------------------------------------------------------
    # Atlas loading
    # ------------------------------------------------------------------

    def load_atlas(self) -> tuple[np.ndarray, nib.Nifti1Image]:
        """Load the ArterialAtlas136 volume. Cached after first call.

        Returns:
            Tuple of (atlas_data as integer array, atlas NIfTI image).
        """
        if self._atlas_cache is not None:
            return self._atlas_cache

        atlas_img = load_nifti(self._config.atlas_file)
        atlas_data = atlas_img.get_fdata(dtype=np.float32).astype(np.int16)
        self._atlas_cache = (atlas_data, atlas_img)
        logger.info(
            "Loaded atlas with %d regions, shape %s",
            len(np.unique(atlas_data)),
            atlas_data.shape,
        )
        return self._atlas_cache

    def load_atlas_labels(self) -> dict[int, str]:
        """Load atlas region label mapping from text file.

        Expected format per line: index|abbreviation|full_name|group

        Returns:
            Dict mapping region index to full name string.
        """
        if self._atlas_labels_cache is not None:
            return self._atlas_labels_cache

        labels = {}
        path = Path(self._config.atlas_labels_file)
        for line in path.read_text().strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                idx = int(parts[0])
                full_name = parts[2]
                labels[idx] = full_name
        self._atlas_labels_cache = labels
        logger.info("Loaded %d atlas labels", len(labels))
        return self._atlas_labels_cache

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def preprocess_subject(
        self, subject_id: str
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """Load and preprocess all modalities for a BIDS subject.

        Performs: load -> 4D collapse -> normalization -> co-registration to T1.

        Args:
            subject_id: BIDS subject ID (e.g., 'sub-1000').

        Returns:
            Tuple of (modality_data_dict, mask_dict, registered_atlas_data).
        """
        raw_modalities = self.load_subject_modalities(subject_id)
        masks = self.load_subject_masks(subject_id)
        atlas_data, atlas_img = self.load_atlas()

        if "T1" not in raw_modalities:
            raise ValueError(f"T1 modality required for subject {subject_id}")

        t1_img = raw_modalities["T1"]

        # Register atlas to T1 space
        registered_atlas = resample_to_reference(atlas_img, t1_img, interpolation="nearest")
        atlas_aligned = registered_atlas.get_fdata(dtype=np.float32).astype(np.int16)

        # Process each modality
        modality_data: dict[str, np.ndarray] = {}
        for name, img in raw_modalities.items():
            data = img.get_fdata(dtype=np.float32)
            data = handle_4d_volume(data)

            if name != "T1":
                resampled = resample_to_reference(
                    nib.Nifti1Image(data, img.affine), t1_img, interpolation="linear"
                )
                data = resampled.get_fdata(dtype=np.float32)

            modality_data[name] = normalize_image(data)

        # Register masks to T1 space
        registered_masks: dict[str, np.ndarray] = {}
        for mname, mdata in masks.items():
            if mdata.size == 0:
                registered_masks[mname] = np.zeros(t1_img.shape[:3], dtype=np.uint8)
                continue
            # Find affine from the mask's source (assume same as TRACE/ADC)
            trace_key = "TRACE" if "TRACE" in raw_modalities else "ADC"
            if trace_key in raw_modalities:
                mask_img = nib.Nifti1Image(mdata.astype(np.float32), raw_modalities[trace_key].affine)
                reg = resample_to_reference(mask_img, t1_img, interpolation="nearest")
                registered_masks[mname] = (reg.get_fdata() > 0).astype(np.uint8)
            else:
                registered_masks[mname] = mdata

        return modality_data, registered_masks, atlas_aligned

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def get_subject_metadata(
        subject_id: str, masks: dict[str, np.ndarray]
    ) -> dict:
        """Derive stroke status and lesion level from mask data.

        Args:
            subject_id: Subject identifier.
            masks: Dict with 'General', 'Acute', 'Chronic' mask arrays.

        Returns:
            Metadata dict with subject, stroke_status, lesion_level keys.
        """
        has_acute = bool(np.any(masks.get("Acute", np.array([])) > 0))
        has_chronic = bool(np.any(masks.get("Chronic", np.array([])) > 0))
        has_general = bool(np.any(masks.get("General", np.array([])) > 0))

        if has_acute and has_chronic:
            lesion_level = "both"
        elif has_acute:
            lesion_level = "acute"
        elif has_chronic:
            lesion_level = "chronic"
        else:
            lesion_level = "none"

        stroke_status = "stroke" if (has_general or has_acute or has_chronic) else "non-stroke"

        return {
            "subject": subject_id,
            "stroke_status": stroke_status,
            "lesion_level": lesion_level,
            "has_acute": has_acute,
            "has_chronic": has_chronic,
        }
