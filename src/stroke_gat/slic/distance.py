"""Distance functions for the anatomically-constrained 3D SLIC algorithm.

Implements the distance metric from Equation 5 of the paper:
  D*_k(v) = d_intensity + (m/S) * d_spatial + lambda * Delta(A(v), A(s_k))

Where:
  - d_intensity: multi-modal weighted intensity distance (Eq. 6)
  - d_spatial: normalized Euclidean distance in 3D
  - Delta: atlas boundary penalty (0 if same region, 1 if different)
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def multimodal_intensity_distance(
    voxel_intensities: np.ndarray,
    center_intensities: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute weighted multi-modal intensity distance (Eq. 6).

    d_intensity = sqrt(sum_m w_m * (I^m_v - I^m_center)^2)

    Args:
        voxel_intensities: Intensity values at voxel for each modality (M,).
        center_intensities: Intensity values at cluster center (M,).
        weights: Per-modality weights (M,), should sum to 1.

    Returns:
        Scalar distance value.
    """
    dist_sq = 0.0
    for m in range(voxel_intensities.shape[0]):
        diff = voxel_intensities[m] - center_intensities[m]
        dist_sq += weights[m] * diff * diff
    return np.sqrt(dist_sq)


@njit(cache=True)
def spatial_distance(
    voxel_coord: np.ndarray,
    center_coord: np.ndarray,
) -> float:
    """Compute Euclidean spatial distance between voxel and cluster center.

    Args:
        voxel_coord: 3D coordinates of the voxel (3,).
        center_coord: 3D coordinates of the cluster center (3,).

    Returns:
        Euclidean distance.
    """
    dist_sq = 0.0
    for i in range(3):
        diff = voxel_coord[i] - center_coord[i]
        dist_sq += diff * diff
    return np.sqrt(dist_sq)


@njit(cache=True)
def atlas_penalty(voxel_atlas: int, center_atlas: int) -> float:
    """Compute atlas boundary penalty (Eq. 5 Delta function).

    Delta(A(v), A(s_k)) = 0 if same atlas region, 1 if different.

    Args:
        voxel_atlas: Atlas region label of the voxel.
        center_atlas: Atlas region label of the cluster center.

    Returns:
        0.0 or 1.0.
    """
    if voxel_atlas == center_atlas:
        return 0.0
    return 1.0


@njit(cache=True)
def combined_distance(
    voxel_intensities: np.ndarray,
    center_intensities: np.ndarray,
    modality_weights: np.ndarray,
    voxel_coord: np.ndarray,
    center_coord: np.ndarray,
    grid_spacing: float,
    compactness: float,
    voxel_atlas: int,
    center_atlas: int,
    atlas_lambda: float,
) -> float:
    """Compute the full anatomically-constrained distance (Eq. 5).

    D*_k(v) = d_intensity + (compactness / S) * d_spatial + lambda * Delta(A(v), A(s_k))

    Args:
        voxel_intensities: Multi-modal intensity values at voxel (M,).
        center_intensities: Multi-modal intensity values at center (M,).
        modality_weights: Per-modality weights (M,).
        voxel_coord: 3D coordinates of voxel (3,).
        center_coord: 3D coordinates of cluster center (3,).
        grid_spacing: Expected supervoxel grid spacing S.
        compactness: Compactness parameter m controlling spatial weight.
        voxel_atlas: Atlas label of the voxel.
        center_atlas: Atlas label of the cluster center.
        atlas_lambda: Weight of the atlas boundary penalty.

    Returns:
        Combined distance value.
    """
    d_int = multimodal_intensity_distance(voxel_intensities, center_intensities, modality_weights)
    d_sp = spatial_distance(voxel_coord, center_coord)
    d_atlas = atlas_penalty(voxel_atlas, center_atlas)

    return d_int + (compactness / grid_spacing) * d_sp + atlas_lambda * d_atlas


@njit(parallel=True, cache=True)
def assign_voxels_to_centers(
    modalities_stack: np.ndarray,
    atlas: np.ndarray,
    brain_mask: np.ndarray,
    centers_coords: np.ndarray,
    centers_intensities: np.ndarray,
    centers_atlas: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    modality_weights: np.ndarray,
    grid_spacing: float,
    compactness: float,
    atlas_lambda: float,
    search_radius: int,
) -> None:
    """Assign each voxel to the nearest cluster center within search radius.

    This is the inner loop of SLIC, parallelized with numba for performance.
    Modifies `labels` and `distances` arrays in-place.

    Args:
        modalities_stack: (M, D, H, W) array of modality volumes.
        atlas: (D, H, W) integer atlas labels.
        brain_mask: (D, H, W) boolean brain mask.
        centers_coords: (K, 3) float center coordinates.
        centers_intensities: (K, M) float center intensities.
        centers_atlas: (K,) int center atlas labels.
        labels: (D, H, W) output label array (modified in-place).
        distances: (D, H, W) output distance array (modified in-place).
        modality_weights: (M,) modality weights.
        grid_spacing: Expected supervoxel spacing S.
        compactness: Compactness parameter.
        atlas_lambda: Atlas penalty weight.
        search_radius: Half-width of the search region around each center.
    """
    n_modalities = modalities_stack.shape[0]
    depth, height, width = modalities_stack.shape[1], modalities_stack.shape[2], modalities_stack.shape[3]
    n_centers = centers_coords.shape[0]

    for k in prange(n_centers):
        cx = int(centers_coords[k, 0])
        cy = int(centers_coords[k, 1])
        cz = int(centers_coords[k, 2])

        # Search bounds
        x_lo = max(0, cx - search_radius)
        x_hi = min(depth, cx + search_radius + 1)
        y_lo = max(0, cy - search_radius)
        y_hi = min(height, cy + search_radius + 1)
        z_lo = max(0, cz - search_radius)
        z_hi = min(width, cz + search_radius + 1)

        center_int = centers_intensities[k]
        center_coord = centers_coords[k]
        center_atl = centers_atlas[k]

        for x in range(x_lo, x_hi):
            for y in range(y_lo, y_hi):
                for z in range(z_lo, z_hi):
                    if not brain_mask[x, y, z]:
                        continue

                    # Gather voxel intensities
                    voxel_int = np.empty(n_modalities, dtype=np.float64)
                    for m in range(n_modalities):
                        voxel_int[m] = modalities_stack[m, x, y, z]

                    voxel_coord = np.array([float(x), float(y), float(z)])

                    dist = combined_distance(
                        voxel_int,
                        center_int,
                        modality_weights,
                        voxel_coord,
                        center_coord,
                        grid_spacing,
                        compactness,
                        int(atlas[x, y, z]),
                        int(center_atl),
                        atlas_lambda,
                    )

                    if dist < distances[x, y, z]:
                        distances[x, y, z] = dist
                        labels[x, y, z] = k + 1  # 1-indexed labels
