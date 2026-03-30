"""Post-processing for SLIC supervoxel labels: connectivity and statistics."""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def enforce_connectivity(
    labels: np.ndarray,
    min_size: int = 20,
) -> np.ndarray:
    """Enforce spatial connectivity and merge small clusters.

    Ensures every supervoxel forms a single connected component.
    Fragments smaller than min_size are reassigned to their nearest
    larger neighbor.

    Args:
        labels: 3D integer label array from SLIC.
        min_size: Minimum number of voxels for a valid supervoxel.

    Returns:
        Cleaned label array with connected, adequately-sized supervoxels.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # exclude background

    new_labels = labels.copy()
    next_label = int(unique_labels.max()) + 1

    for lbl in unique_labels:
        mask = labels == lbl
        connected, n_components = ndimage.label(mask)

        if n_components <= 1:
            continue

        # Keep the largest component, relabel others
        component_sizes = ndimage.sum(mask, connected, range(1, n_components + 1))
        largest = np.argmax(component_sizes) + 1

        for comp_id in range(1, n_components + 1):
            if comp_id == largest:
                continue
            fragment_mask = connected == comp_id
            if component_sizes[comp_id - 1] < min_size:
                # Merge into nearest neighbor by dilation
                _merge_fragment(new_labels, fragment_mask, lbl)
            else:
                # Assign new unique label
                new_labels[fragment_mask] = next_label
                next_label += 1

    # Second pass: merge any remaining small clusters
    final_labels = new_labels.copy()
    for lbl in np.unique(final_labels):
        if lbl == 0:
            continue
        count = np.sum(final_labels == lbl)
        if count < min_size:
            mask = final_labels == lbl
            _merge_fragment(final_labels, mask, lbl)

    # Re-index labels to be contiguous starting from 1
    unique_final = np.unique(final_labels)
    unique_final = unique_final[unique_final > 0]
    remap = np.zeros(int(final_labels.max()) + 1, dtype=np.int32)
    for new_idx, old_lbl in enumerate(unique_final, start=1):
        remap[old_lbl] = new_idx
    result = remap[final_labels]

    n_sv = len(unique_final)
    logger.info("Post-processing: %d supervoxels after connectivity enforcement", n_sv)
    return result


def _merge_fragment(labels: np.ndarray, fragment_mask: np.ndarray, original_label: int) -> None:
    """Merge a small fragment into its nearest spatial neighbor.

    Uses dilation to find adjacent labels and picks the most common one.
    """
    dilated = ndimage.binary_dilation(fragment_mask, iterations=2)
    border = dilated & ~fragment_mask & (labels > 0) & (labels != original_label)

    if np.any(border):
        neighbor_labels = labels[border]
        # Pick most frequent neighbor
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        best_neighbor = unique[np.argmax(counts)]
        labels[fragment_mask] = best_neighbor
    # If no neighbor found, leave as is (edge case)


def compute_supervoxel_stats(
    labels: np.ndarray,
    atlas: np.ndarray | None = None,
) -> dict:
    """Compute quality metrics for supervoxel segmentation.

    Args:
        labels: 3D integer supervoxel label array.
        atlas: Optional atlas label array for computing region adherence.

    Returns:
        Dict with keys: num_supervoxels, mean_size, std_size,
        min_size, max_size, region_adherence (if atlas provided).
    """
    unique = np.unique(labels)
    unique = unique[unique > 0]
    sizes = np.array([np.sum(labels == lbl) for lbl in unique])

    stats = {
        "num_supervoxels": len(unique),
        "mean_size": float(np.mean(sizes)),
        "std_size": float(np.std(sizes)),
        "min_size": int(np.min(sizes)),
        "max_size": int(np.max(sizes)),
    }

    if atlas is not None:
        adherent = 0
        for lbl in unique:
            mask = labels == lbl
            atlas_vals = atlas[mask]
            if atlas_vals.size > 0:
                dominant = np.bincount(atlas_vals.astype(int)).argmax()
                fraction_dominant = np.sum(atlas_vals == dominant) / atlas_vals.size
                if fraction_dominant >= 0.9:
                    adherent += 1
        stats["region_adherence"] = adherent / len(unique) if len(unique) > 0 else 0.0

    return stats
