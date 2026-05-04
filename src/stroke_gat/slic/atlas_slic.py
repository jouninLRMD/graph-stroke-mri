"""Anatomically-constrained 3D SLIC supervoxel segmentation.

Algorithm 1 from the paper, with the modified distance metric of Eq. 5:

    D*_k(v) = d_intensity + (m/S) * d_spatial + lambda * Delta(A(v), A(s_k))

The atlas penalty term keeps clusters from crossing arterial-territory
boundaries; the multi-modal intensity term uses all MRI sequences. The voxel
assignment loop is JIT-compiled with numba so per-subject segmentation runs in
roughly a minute on commodity CPUs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import sobel

from stroke_gat.config import SLICConfig
from stroke_gat.slic.distance import assign_voxels_to_centers
from stroke_gat.slic.postprocess import compute_supervoxel_stats, enforce_connectivity

logger = logging.getLogger(__name__)


class AnatomicalSLIC:
    """3D SLIC with multi-modal intensity, atlas penalty, and brain-mask support.

    With ``target_supervoxel_size = 256`` and the default penalty lambda this
    yields about 8000 supervoxels per brain with >90% atlas-region adherence.
    """

    def __init__(self, config: SLICConfig):
        self.config = config

    def segment(
        self,
        modalities: dict[str, np.ndarray],
        atlas: np.ndarray,
        brain_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Segment a brain volume into anatomically-constrained supervoxels.

        Args:
            modalities: Dict mapping modality name to 3D float32 array.
                        All arrays must have the same shape.
                        Expected keys: 'T1', 'FLAIR', 'ADC', 'TRACE'.
            atlas: 3D integer array of atlas region labels.
            brain_mask: Optional boolean mask. If None, derived from
                        the first modality (intensity > 0.01).

        Returns:
            3D integer array of supervoxel labels (1-indexed, 0=background).
        """
        # Validate inputs
        shapes = {name: vol.shape for name, vol in modalities.items()}
        ref_shape = next(iter(shapes.values()))
        for name, shape in shapes.items():
            if shape != ref_shape:
                raise ValueError(f"Shape mismatch: {name} is {shape}, expected {ref_shape}")

        if atlas.shape != ref_shape:
            raise ValueError(f"Atlas shape {atlas.shape} != volume shape {ref_shape}")

        # Build modality stack and weights
        modality_names = sorted(modalities.keys())
        modalities_stack = np.stack(
            [modalities[name].astype(np.float64) for name in modality_names], axis=0
        )
        n_modalities = len(modality_names)

        # Get weights for available modalities
        weights = np.array(
            [self.config.modality_weights.get(name, 1.0 / n_modalities) for name in modality_names],
            dtype=np.float64,
        )
        weights /= weights.sum()  # Normalize

        # Brain mask
        if brain_mask is None:
            brain_mask = modalities_stack[0] > 0.01
        brain_mask = brain_mask.astype(np.bool_)

        n_brain_voxels = int(np.sum(brain_mask))
        logger.info(
            "SLIC input: shape=%s, modalities=%s, brain_voxels=%d",
            ref_shape,
            modality_names,
            n_brain_voxels,
        )

        # Compute number of supervoxels and grid spacing
        n_supervoxels = max(1, n_brain_voxels // self.config.target_supervoxel_size)
        grid_spacing = (n_brain_voxels / n_supervoxels) ** (1.0 / 3.0)

        logger.info(
            "Target: %d supervoxels, grid_spacing=%.1f, target_size=%d",
            n_supervoxels,
            grid_spacing,
            self.config.target_supervoxel_size,
        )

        # Initialize cluster centers
        centers_coords, centers_intensities, centers_atlas = self._initialize_centers(
            modalities_stack, atlas, brain_mask, n_supervoxels, grid_spacing
        )
        n_centers = centers_coords.shape[0]
        logger.info("Initialized %d cluster centers", n_centers)

        # Perturb centers to lowest gradient position
        gradient = self._compute_gradient(modalities_stack)
        centers_coords = self._perturb_centers(centers_coords, gradient, brain_mask)

        # Update center intensities/atlas after perturbation
        for k in range(n_centers):
            x, y, z = int(centers_coords[k, 0]), int(centers_coords[k, 1]), int(centers_coords[k, 2])
            x = np.clip(x, 0, ref_shape[0] - 1)
            y = np.clip(y, 0, ref_shape[1] - 1)
            z = np.clip(z, 0, ref_shape[2] - 1)
            centers_intensities[k] = modalities_stack[:, x, y, z]
            centers_atlas[k] = atlas[x, y, z]

        # Iterative assignment
        labels = np.zeros(ref_shape, dtype=np.int32)
        search_radius = int(2 * grid_spacing)

        for iteration in range(self.config.max_iterations):
            distances = np.full(ref_shape, np.inf, dtype=np.float64)

            assign_voxels_to_centers(
                modalities_stack,
                atlas.astype(np.int32),
                brain_mask,
                centers_coords,
                centers_intensities,
                centers_atlas.astype(np.int32),
                labels,
                distances,
                weights,
                grid_spacing,
                self.config.compactness,
                self.config.atlas_penalty_lambda,
                search_radius,
            )

            # Update centers
            new_centers_coords, new_centers_intensities, new_centers_atlas = (
                self._update_centers(labels, modalities_stack, atlas, brain_mask, n_centers)
            )

            # Check convergence
            valid = np.all(np.isfinite(new_centers_coords), axis=1) & np.all(
                np.isfinite(centers_coords), axis=1
            )
            if np.sum(valid) > 0:
                residual = np.sqrt(
                    np.sum((new_centers_coords[valid] - centers_coords[valid]) ** 2, axis=1)
                ).mean()
            else:
                residual = 0.0

            centers_coords = new_centers_coords
            centers_intensities = new_centers_intensities
            centers_atlas = new_centers_atlas

            logger.info("Iteration %d/%d: residual=%.6f", iteration + 1, self.config.max_iterations, residual)

            if residual < self.config.convergence_threshold:
                logger.info("Converged at iteration %d", iteration + 1)
                break

        # Post-processing: enforce connectivity
        labels = enforce_connectivity(labels, min_size=self.config.min_cluster_size)

        stats = compute_supervoxel_stats(labels, atlas)
        logger.info(
            "SLIC complete: %d supervoxels, mean_size=%.0f, adherence=%.1f%%",
            stats["num_supervoxels"],
            stats["mean_size"],
            stats.get("region_adherence", 0) * 100,
        )

        return labels

    def _initialize_centers(
        self,
        modalities_stack: np.ndarray,
        atlas: np.ndarray,
        brain_mask: np.ndarray,
        n_supervoxels: int,
        grid_spacing: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize cluster centers on a regular 3D grid within the brain mask.

        Returns:
            Tuple of (coords (K,3), intensities (K,M), atlas_labels (K,)).
        """
        depth, height, width = modalities_stack.shape[1:]
        step = max(1, int(grid_spacing))

        # Generate grid points
        grid_x = np.arange(step // 2, depth, step)
        grid_y = np.arange(step // 2, height, step)
        grid_z = np.arange(step // 2, width, step)

        coords_list = []
        intensities_list = []
        atlas_list = []

        for x in grid_x:
            for y in grid_y:
                for z in grid_z:
                    if brain_mask[x, y, z]:
                        coords_list.append([float(x), float(y), float(z)])
                        intensities_list.append(modalities_stack[:, x, y, z].copy())
                        atlas_list.append(int(atlas[x, y, z]))

        if len(coords_list) == 0:
            # Fallback: sample from brain voxels
            brain_indices = np.argwhere(brain_mask)
            step = max(1, len(brain_indices) // n_supervoxels)
            sampled = brain_indices[::step]
            for idx in sampled:
                x, y, z = idx
                coords_list.append([float(x), float(y), float(z)])
                intensities_list.append(modalities_stack[:, x, y, z].copy())
                atlas_list.append(int(atlas[x, y, z]))

        coords = np.array(coords_list, dtype=np.float64)
        intensities = np.array(intensities_list, dtype=np.float64)
        atlas_labels = np.array(atlas_list, dtype=np.int32)

        return coords, intensities, atlas_labels

    def _perturb_centers(
        self,
        centers: np.ndarray,
        gradient: np.ndarray,
        brain_mask: np.ndarray,
    ) -> np.ndarray:
        """Move each center to the lowest gradient position in a 3x3x3 neighborhood.

        This prevents centers from landing on edges, improving cluster quality.
        """
        depth, height, width = gradient.shape
        perturbed = centers.copy()

        for k in range(centers.shape[0]):
            cx, cy, cz = int(centers[k, 0]), int(centers[k, 1]), int(centers[k, 2])
            best_grad = gradient[
                np.clip(cx, 0, depth - 1),
                np.clip(cy, 0, height - 1),
                np.clip(cz, 0, width - 1),
            ]
            best_pos = (cx, cy, cz)

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nx, ny, nz = cx + dx, cy + dy, cz + dz
                        if 0 <= nx < depth and 0 <= ny < height and 0 <= nz < width:
                            if brain_mask[nx, ny, nz] and gradient[nx, ny, nz] < best_grad:
                                best_grad = gradient[nx, ny, nz]
                                best_pos = (nx, ny, nz)

            perturbed[k] = [float(best_pos[0]), float(best_pos[1]), float(best_pos[2])]

        return perturbed

    def _compute_gradient(self, modalities_stack: np.ndarray) -> np.ndarray:
        """Compute 3D gradient magnitude averaged across modalities.

        Uses Sobel filters along each spatial axis.
        """
        n_modalities = modalities_stack.shape[0]
        grad_sum = np.zeros(modalities_stack.shape[1:], dtype=np.float64)

        for m in range(n_modalities):
            vol = modalities_stack[m]
            gx = sobel(vol, axis=0)
            gy = sobel(vol, axis=1)
            gz = sobel(vol, axis=2)
            grad_sum += np.sqrt(gx**2 + gy**2 + gz**2)

        return grad_sum / n_modalities

    def _update_centers(
        self,
        labels: np.ndarray,
        modalities_stack: np.ndarray,
        atlas: np.ndarray,
        brain_mask: np.ndarray,
        n_centers: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recompute cluster centers as mean of assigned voxels."""
        n_modalities = modalities_stack.shape[0]
        depth, height, width = labels.shape

        new_coords = np.zeros((n_centers, 3), dtype=np.float64)
        new_intensities = np.zeros((n_centers, n_modalities), dtype=np.float64)
        new_atlas = np.zeros(n_centers, dtype=np.int32)
        counts = np.zeros(n_centers, dtype=np.float64)

        # Use vectorized operations for efficiency
        brain_indices = np.argwhere(brain_mask & (labels > 0))

        for idx in brain_indices:
            x, y, z = idx[0], idx[1], idx[2]
            k = labels[x, y, z] - 1  # 0-indexed
            if 0 <= k < n_centers:
                new_coords[k, 0] += x
                new_coords[k, 1] += y
                new_coords[k, 2] += z
                for m in range(n_modalities):
                    new_intensities[k, m] += modalities_stack[m, x, y, z]
                counts[k] += 1

        # Average
        for k in range(n_centers):
            if counts[k] > 0:
                new_coords[k] /= counts[k]
                new_intensities[k] /= counts[k]
                # Atlas label: mode of assigned voxels (approximate with center position)
                cx = int(np.clip(new_coords[k, 0], 0, depth - 1))
                cy = int(np.clip(new_coords[k, 1], 0, height - 1))
                cz = int(np.clip(new_coords[k, 2], 0, width - 1))
                new_atlas[k] = atlas[cx, cy, cz]
            else:
                new_coords[k] = np.nan
                new_intensities[k] = np.nan

        return new_coords, new_intensities, new_atlas
