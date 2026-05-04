"""Edge computation for supervoxel brain graphs.

Creates edges between spatially adjacent supervoxels using 26-connectivity
and computes edge weights using the 4-component formula (Eq. 7):

w_ij = alpha * sim_intensity(i,j) + beta * sim_atlas(i,j) +
       gamma * sim_spatial(i,j) + delta * P_co_occurrence(i,j)
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch

from stroke_gat.config import GraphConfig

logger = logging.getLogger(__name__)


class EdgeComputer:
    """Computes edges and edge weights from supervoxel adjacency."""

    # 26-connectivity offsets (all neighbors in 3D grid)
    OFFSETS_26 = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(self, config: GraphConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")

    def compute(
        self,
        supervoxel_labels: np.ndarray,
        node_features: torch.Tensor,
        parcellation_labels: torch.Tensor,
        lesion_labels: torch.Tensor,
        batch_size: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute edges, attributes, and weights between adjacent supervoxels.

        Args:
            supervoxel_labels: 3D integer array of supervoxel labels (1-indexed).
            node_features: (N, F) tensor of node features.
            parcellation_labels: (N,) tensor of atlas region per node.
            lesion_labels: (N,) tensor of lesion class per node.
            batch_size: Number of offset directions per batch.

        Returns:
            Tuple of:
                edge_index: (2, E) long tensor of edges
                edge_attr: (E, D) edge attribute tensor
                edge_weight: (E,) float tensor of edge weights
        """
        shape = supervoxel_labels.shape
        sv_3d = torch.tensor(supervoxel_labels, dtype=torch.int32, device=self.device)

        # Build label-to-index mapping
        unique_labels = torch.unique(sv_3d)
        unique_labels = unique_labels[unique_labels > 0]
        num_nodes = unique_labels.size(0)

        max_label = int(sv_3d.max().item())
        label_to_idx = torch.full((max_label + 1,), -1, dtype=torch.int32, device=self.device)
        label_to_idx[unique_labels.long()] = torch.arange(num_nodes, dtype=torch.int32, device=self.device)

        # Find edges by checking shifted adjacency
        edge_set: set[tuple[int, int]] = set()
        offsets = self.OFFSETS_26

        for batch_start in range(0, len(offsets), batch_size):
            batch_offsets = offsets[batch_start : batch_start + batch_size]

            for dx, dy, dz in batch_offsets:
                shifted = torch.roll(sv_3d, shifts=(dx, dy, dz), dims=(0, 1, 2))

                # Mask: different labels, both non-zero
                mask = (sv_3d != shifted) & (sv_3d > 0) & (shifted > 0)

                src_labels = sv_3d[mask]
                dst_labels = shifted[mask]

                src_idx = label_to_idx[src_labels.long()]
                dst_idx = label_to_idx[dst_labels.long()]

                # Filter valid indices
                valid = (src_idx >= 0) & (dst_idx >= 0)
                src_idx = src_idx[valid].cpu()
                dst_idx = dst_idx[valid].cpu()

                for i in range(src_idx.size(0)):
                    s, d = int(src_idx[i].item()), int(dst_idx[i].item())
                    if s < d:
                        edge_set.add((s, d))
                    elif d < s:
                        edge_set.add((d, s))

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        del sv_3d

        if not edge_set:
            logger.warning("No edges found!")
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
            )

        # Convert to tensors (undirected: add both directions)
        edges = list(edge_set)
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)

        # Make bidirectional
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )

        logger.info("Found %d undirected edges (%d directed)", len(edges), edge_index.size(1))

        # Compute edge weights using the 4-component formula
        edge_weight = self._compute_edge_weights(
            edge_index, node_features, parcellation_labels, lesion_labels
        )

        # Edge attributes: [same_atlas, intensity_sim, spatial_dist]
        edge_attr = self._compute_edge_attributes(
            edge_index, node_features, parcellation_labels
        )

        gc.collect()
        return edge_index, edge_attr, edge_weight

    def _compute_edge_weights(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        parcellation_labels: torch.Tensor,
        lesion_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge weights using 4-component formula (Eq. 7).

        w_ij = alpha * sim_intensity + beta * sim_atlas + gamma * sim_spatial + delta * P_co_occurrence
        """
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.size(0)

        # Number of modalities (4 stats per modality, starting features)
        # Features layout: [mod1_mean, mod1_std, mod1_min, mod1_max, ..., cx, cy, cz, vol, atlas_onehot...]
        # Mean intensities are at indices 0, 4, 8, 12 (for 4 modalities)
        n_mods = 4  # T1, FLAIR, ADC, TRACE
        mean_indices = [i * 4 for i in range(min(n_mods, node_features.size(1) // 4))]

        # 1. Intensity similarity (cosine similarity of mean intensity vectors)
        if mean_indices:
            src_means = node_features[src][:, mean_indices]
            dst_means = node_features[dst][:, mean_indices]
            cos_sim = torch.nn.functional.cosine_similarity(src_means, dst_means, dim=1)
            sim_intensity = (cos_sim + 1) / 2  # Map [-1, 1] to [0, 1]
        else:
            sim_intensity = torch.ones(num_edges)

        # 2. Atlas similarity
        src_atlas = parcellation_labels[src]
        dst_atlas = parcellation_labels[dst]
        sim_atlas = (src_atlas == dst_atlas).float()

        # 3. Spatial proximity (Gaussian kernel on centroid distance)
        # Centroids are at indices [n_mods*4, n_mods*4+1, n_mods*4+2]
        centroid_start = min(n_mods * 4, node_features.size(1) - 3)
        src_centroids = node_features[src][:, centroid_start:centroid_start + 3]
        dst_centroids = node_features[dst][:, centroid_start:centroid_start + 3]
        spatial_dist = torch.norm(src_centroids - dst_centroids, dim=1)
        sim_spatial = torch.exp(-spatial_dist ** 2 / (2 * self.config.spatial_sigma ** 2))

        # 4. Lesion co-occurrence prior
        src_lesion = lesion_labels[src]
        dst_lesion = lesion_labels[dst]
        # Simple co-occurrence: same lesion type gets higher weight
        co_occurrence = (src_lesion == dst_lesion).float() * 0.7 + 0.3

        # Combine with weights
        edge_weight = (
            self.config.edge_weight_alpha * sim_intensity
            + self.config.edge_weight_beta * sim_atlas
            + self.config.edge_weight_gamma * sim_spatial
            + self.config.edge_weight_delta * co_occurrence
        )

        return edge_weight

    def _compute_edge_attributes(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        parcellation_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge attribute vector for each edge.

        Returns (E, 3) tensor with [same_atlas, intensity_sim, spatial_dist].
        """
        src, dst = edge_index[0], edge_index[1]

        # Same atlas region
        same_atlas = (parcellation_labels[src] == parcellation_labels[dst]).float().unsqueeze(1)

        # Intensity similarity
        n_mods = 4
        mean_indices = [i * 4 for i in range(min(n_mods, node_features.size(1) // 4))]
        if mean_indices:
            src_means = node_features[src][:, mean_indices]
            dst_means = node_features[dst][:, mean_indices]
            intensity_sim = torch.nn.functional.cosine_similarity(src_means, dst_means, dim=1)
            intensity_sim = ((intensity_sim + 1) / 2).unsqueeze(1)
        else:
            intensity_sim = torch.ones(src.size(0), 1)

        # Spatial distance
        centroid_start = min(n_mods * 4, node_features.size(1) - 3)
        src_c = node_features[src][:, centroid_start:centroid_start + 3]
        dst_c = node_features[dst][:, centroid_start:centroid_start + 3]
        spatial_dist = torch.norm(src_c - dst_c, dim=1).unsqueeze(1)

        return torch.cat([same_atlas, intensity_sim, spatial_dist], dim=1)
