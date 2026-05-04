"""Node feature computation for supervoxel-based brain graphs.

Computes per-supervoxel features using GPU-accelerated scatter operations:
- Per modality (T1, FLAIR, ADC, TRACE): mean, std, min, max => 16 dims
- 3D centroid (x, y, z) normalized => 3 dims
- Volume (voxel count, log-normalized) => 1 dim
- Atlas region one-hot (dominant label) => num_atlas_regions dims
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
import torch_scatter

logger = logging.getLogger(__name__)


class NodeFeatureComputer:
    """Computes per-supervoxel node features and labels."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")

    def compute(
        self,
        modalities: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        atlas: np.ndarray,
        supervoxel_labels: np.ndarray,
        batch_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute node features, lesion labels, and parcellation labels.

        Args:
            modalities: Dict of modality_name -> 3D float array.
            masks: Dict with 'General', 'Acute', 'Chronic' -> 3D uint8 array.
            atlas: 3D integer atlas label array.
            supervoxel_labels: 3D integer supervoxel label array (1-indexed).
            batch_size: Number of supervoxels per batch for memory management.

        Returns:
            Tuple of:
                node_features: (N, F) tensor of node features
                lesion_labels: (N,) tensor of lesion class per node (0, 1, or 2)
                parcellation_labels: (N,) tensor of dominant atlas region per node
        """
        # Flatten all arrays
        sv_flat = torch.tensor(supervoxel_labels.ravel(), dtype=torch.int64)
        unique_svs, inverse = torch.unique(sv_flat, sorted=True, return_inverse=True)
        # Filter out background (label 0)
        valid_mask = sv_flat > 0
        inverse_valid = inverse.clone()
        # Re-index to exclude background
        if unique_svs[0] == 0:
            unique_svs = unique_svs[1:]
            inverse_valid = inverse - 1
            valid_mask = sv_flat > 0

        num_svs = unique_svs.size(0)
        logger.info("Computing features for %d supervoxels", num_svs)

        # Flatten modalities
        modality_names = sorted(modalities.keys())
        mod_flat = {
            name: torch.tensor(modalities[name].ravel(), dtype=torch.float32)
            for name in modality_names
        }

        # Flatten masks
        general = torch.tensor(masks.get("General", np.zeros_like(atlas)).ravel(), dtype=torch.float32)
        acute = torch.tensor(masks.get("Acute", np.zeros_like(atlas)).ravel(), dtype=torch.float32)
        chronic = torch.tensor(masks.get("Chronic", np.zeros_like(atlas)).ravel(), dtype=torch.float32)
        atlas_flat = torch.tensor(atlas.ravel(), dtype=torch.int64)

        # Voxel coordinates
        coords = np.indices(supervoxel_labels.shape).reshape(3, -1).T.astype(np.float32)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        # Process in batches
        all_features = []
        all_lesion_labels = []
        all_parcellation = []

        for batch_start in range(0, num_svs, batch_size):
            batch_end = min(batch_start + batch_size, num_svs)

            # Mask for voxels in this batch of supervoxels
            batch_mask = valid_mask & (inverse_valid >= batch_start) & (inverse_valid < batch_end)
            batch_inv = (inverse_valid[batch_mask] - batch_start).to(torch.int64).to(self.device)

            batch_n = batch_end - batch_start
            features_list = []

            # Intensity statistics per modality: mean, std, min, max
            for name in modality_names:
                mod_data = mod_flat[name][batch_mask].to(self.device)

                mean = torch_scatter.scatter_mean(mod_data, batch_inv, dim=0, dim_size=batch_n)
                mean_expanded = mean[batch_inv]
                var = torch_scatter.scatter_mean(
                    (mod_data - mean_expanded) ** 2, batch_inv, dim=0, dim_size=batch_n
                )
                std = torch.sqrt(var + 1e-6)
                min_val = torch_scatter.scatter_min(mod_data, batch_inv, dim=0, dim_size=batch_n)[0]
                max_val = torch_scatter.scatter_max(mod_data, batch_inv, dim=0, dim_size=batch_n)[0]

                features_list.append(torch.stack([mean, std, min_val, max_val], dim=1).cpu())
                del mod_data

            # Centroid coordinates (normalized)
            batch_coords = coords_tensor[batch_mask].to(self.device)
            centroid_x = torch_scatter.scatter_mean(batch_coords[:, 0], batch_inv, dim=0, dim_size=batch_n)
            centroid_y = torch_scatter.scatter_mean(batch_coords[:, 1], batch_inv, dim=0, dim_size=batch_n)
            centroid_z = torch_scatter.scatter_mean(batch_coords[:, 2], batch_inv, dim=0, dim_size=batch_n)

            # Normalize centroids to [0, 1]
            shape = torch.tensor(supervoxel_labels.shape, dtype=torch.float32, device=self.device)
            centroid_x = centroid_x / shape[0]
            centroid_y = centroid_y / shape[1]
            centroid_z = centroid_z / shape[2]

            features_list.append(torch.stack([centroid_x, centroid_y, centroid_z], dim=1).cpu())
            del batch_coords

            # Volume (log-normalized voxel count)
            ones = torch.ones(batch_inv.size(0), device=self.device)
            volumes = torch_scatter.scatter_add(ones, batch_inv, dim=0, dim_size=batch_n)
            log_volumes = torch.log1p(volumes).unsqueeze(1)
            features_list.append(log_volumes.cpu())

            # Atlas region one-hot (dominant label per supervoxel)
            batch_atlas = atlas_flat[batch_mask].to(self.device)
            num_atlas_regions = int(atlas_flat.max().item()) + 1
            atlas_onehot = torch.nn.functional.one_hot(batch_atlas, num_classes=num_atlas_regions).float()
            atlas_counts = torch_scatter.scatter_add(atlas_onehot, batch_inv.unsqueeze(1).expand(-1, num_atlas_regions), dim=0, dim_size=batch_n)
            dominant_region = atlas_counts.argmax(dim=1)

            # One-hot encode dominant region
            dominant_onehot = torch.nn.functional.one_hot(dominant_region, num_classes=num_atlas_regions).float()
            features_list.append(dominant_onehot.cpu())
            all_parcellation.append(dominant_region.cpu())
            del batch_atlas, atlas_onehot, atlas_counts

            # Concatenate all features for this batch
            batch_features = torch.cat(features_list, dim=1)
            all_features.append(batch_features)

            # Lesion labels: 0=no lesion, 1=acute, 2=chronic
            batch_general = general[batch_mask].to(self.device)
            batch_acute = acute[batch_mask].to(self.device)
            batch_chronic = chronic[batch_mask].to(self.device)

            gen_max = torch_scatter.scatter_max(batch_general, batch_inv, dim=0, dim_size=batch_n)[0]
            acu_max = torch_scatter.scatter_max(batch_acute, batch_inv, dim=0, dim_size=batch_n)[0]
            chr_max = torch_scatter.scatter_max(batch_chronic, batch_inv, dim=0, dim_size=batch_n)[0]
            del gen_max  # Computed for symmetry / future use; not part of label rule.

            # Per-supervoxel label rule (matches docs/DATA_LAYOUT.md, Section 4):
            #   if any voxel in the supervoxel is in the acute mask     -> 1 (acute)
            #   elif any voxel is in the chronic mask                   -> 2 (chronic)
            #   else                                                    -> 0 (no lesion / control)
            # Acute wins on overlap (more clinically urgent).
            ones = torch.ones(batch_n, dtype=torch.long, device=self.device)
            twos = torch.full((batch_n,), 2, dtype=torch.long, device=self.device)
            zeros = torch.zeros(batch_n, dtype=torch.long, device=self.device)
            labels = torch.where(
                acu_max > 0,
                ones,
                torch.where(chr_max > 0, twos, zeros),
            )

            all_lesion_labels.append(labels.cpu())

            del batch_general, batch_acute, batch_chronic
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        node_features = torch.cat(all_features, dim=0)
        lesion_labels = torch.cat(all_lesion_labels, dim=0)
        parcellation_labels = torch.cat(all_parcellation, dim=0)

        logger.info("Node features shape: %s", node_features.shape)

        gc.collect()
        return node_features, lesion_labels, parcellation_labels
