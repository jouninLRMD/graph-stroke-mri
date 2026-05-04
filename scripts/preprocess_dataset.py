#!/usr/bin/env python
"""Preprocess raw BIDS data: register modalities to T1, normalize, create brain masks.

Usage:
    python scripts/preprocess_dataset.py --config configs/default.yaml
    python scripts/preprocess_dataset.py --config configs/default.yaml --subjects sub-0001 sub-0002
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stroke_gat.config import load_config
from stroke_gat.data.service import DataService
from stroke_gat.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess BIDS dataset for graph construction.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Specific subject IDs to process.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    config = load_config(args.config)
    service = DataService(config.paths)

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = service.discover_subjects_bids()

    if not subjects:
        logger.error("No subjects found. Check your paths config.")
        return

    logger.info("Preprocessing %d subjects", len(subjects))

    output_dir = Path(config.paths.output_preprocessed)
    output_dir.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    for i, subject_id in enumerate(subjects):
        logger.info("[%d/%d] Processing %s", i + 1, len(subjects), subject_id)
        try:
            modalities, masks, atlas = service.preprocess_subject(subject_id)

            # Save preprocessed data
            subject_dir = output_dir / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)

            import nibabel as nib
            import numpy as np

            for mod_name, img in modalities.items():
                nib.save(img, str(subject_dir / f"{mod_name}.nii.gz"))

            for mask_name, mask_data in masks.items():
                nib.save(
                    nib.Nifti1Image(mask_data.astype(np.float32), modalities["T1"].affine),
                    str(subject_dir / f"{mask_name}_mask.nii.gz"),
                )

            logger.info("  Saved preprocessed data to %s", subject_dir)
            success += 1

        except Exception:
            logger.exception("  Failed to process %s", subject_id)
            failed += 1

    logger.info("Done: %d succeeded, %d failed out of %d", success, failed, len(subjects))


if __name__ == "__main__":
    main()
