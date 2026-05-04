#!/usr/bin/env python
"""Generate supervoxel graphs from preprocessed MRI data.

Runs SLIC supervoxel segmentation followed by graph construction
for each subject, saving PyG Data objects as .pt files.

Usage:
    python scripts/generate_graphs.py --config configs/default.yaml
    python scripts/generate_graphs.py --config configs/default.yaml --subjects sub-0001
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from stroke_gat.config import load_config
from stroke_gat.data.service import DataService
from stroke_gat.graph.builder import GraphBuilder
from stroke_gat.utils.logging import setup_logging
from stroke_gat.utils.memory import log_memory

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate supervoxel graphs from MRI data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Specific subject IDs.")
    parser.add_argument("--use-preprocessed", action="store_true", help="Use preprocessed data instead of raw BIDS.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    config = load_config(args.config)
    service = DataService(config.paths)

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    elif args.use_preprocessed:
        preprocessed_dir = Path(config.paths.output_preprocessed)
        subjects = [d.name for d in preprocessed_dir.iterdir() if d.is_dir()]
    else:
        subjects = service.discover_subjects_bids()

    if not subjects:
        logger.error("No subjects found.")
        return

    output_dir = Path(config.paths.output_graphs)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = GraphBuilder(slic_config=config.slic, graph_config=config.graph)

    logger.info("Generating graphs for %d subjects", len(subjects))
    success, failed = 0, 0
    metadata_rows = []

    for i, subject_id in enumerate(subjects):
        output_path = output_dir / f"{subject_id}_supervoxel_graph.pt"
        if output_path.exists():
            logger.info("[%d/%d] %s already exists, skipping", i + 1, len(subjects), subject_id)
            success += 1
            continue

        logger.info("[%d/%d] Building graph for %s", i + 1, len(subjects), subject_id)
        log_memory()

        try:
            if args.use_preprocessed:
                import nibabel as nib
                import numpy as np

                subj_dir = Path(config.paths.output_preprocessed) / subject_id
                modalities = {}
                for mod in ["T1", "FLAIR", "ADC", "TRACE"]:
                    mod_path = subj_dir / f"{mod}.nii.gz"
                    if mod_path.exists():
                        modalities[mod] = nib.load(str(mod_path))

                masks = {}
                for mask_type in ["acute", "chronic", "combined"]:
                    mask_path = subj_dir / f"{mask_type}_mask.nii.gz"
                    if mask_path.exists():
                        masks[mask_type] = nib.load(str(mask_path)).get_fdata().astype(bool)

                atlas, _ = service.load_atlas()
            else:
                modalities, masks, atlas = service.preprocess_subject(subject_id)

            metadata = service.get_subject_metadata(subject_id, masks)
            graph_data, supervoxel_labels = builder.build(
                modalities=modalities,
                masks=masks,
                atlas=atlas,
                subject_id=subject_id,
                metadata=metadata,
            )

            torch.save(graph_data, str(output_path))
            logger.info("  Saved graph: %d nodes, %d edges -> %s",
                        graph_data.num_nodes, graph_data.num_edges, output_path)

            metadata_rows.append(metadata)
            success += 1

        except Exception:
            logger.exception("  Failed to build graph for %s", subject_id)
            failed += 1

    # Save metadata CSV
    if metadata_rows:
        import pandas as pd

        df = pd.DataFrame(metadata_rows)
        csv_path = output_dir.parent / "subjects_metadata.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved metadata to %s", csv_path)

    logger.info("Done: %d succeeded, %d failed out of %d", success, failed, len(subjects))


if __name__ == "__main__":
    main()
