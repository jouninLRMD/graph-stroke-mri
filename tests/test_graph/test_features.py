"""Tests for stroke_gat.graph.features -- NodeFeatureComputer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stroke_gat.graph.features import NodeFeatureComputer


@pytest.fixture
def simple_supervoxel_labels(synthetic_brain_mask) -> np.ndarray:
    """Create a simple supervoxel label volume with ~4 supervoxels from the brain mask."""
    labels = np.zeros((32, 32, 32), dtype=np.int32)
    # Assign 4 quadrants within the brain mask
    labels[:16, :16, :] = 1
    labels[:16, 16:, :] = 2
    labels[16:, :16, :] = 3
    labels[16:, 16:, :] = 4
    # Zero out voxels outside brain mask
    labels[~synthetic_brain_mask] = 0
    return labels


class TestNodeFeatureComputer:
    """Tests for per-supervoxel feature computation."""

    def test_feature_dimension(
        self,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
        simple_supervoxel_labels,
    ):
        """Output features should have expected number of columns.

        Expected dims:
          4 modalities * 4 stats = 16
          + 3 centroid
          + 1 volume
          + num_atlas_regions one-hot
        """
        computer = NodeFeatureComputer(device=torch.device("cpu"))
        features, labels, parc = computer.compute(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            supervoxel_labels=simple_supervoxel_labels,
        )

        num_atlas_regions = int(synthetic_atlas.max()) + 1  # 0..4 => 5
        expected_cols = 4 * 4 + 3 + 1 + num_atlas_regions  # 16 + 3 + 1 + 5 = 25
        assert features.shape[1] == expected_cols

    def test_feature_values_in_range(
        self,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
        simple_supervoxel_labels,
    ):
        """Mean intensity stats should be within input range [0, 1]."""
        computer = NodeFeatureComputer(device=torch.device("cpu"))
        features, _, _ = computer.compute(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            supervoxel_labels=simple_supervoxel_labels,
        )

        # First 4 columns are mean intensities for each modality
        # (columns 0, 4, 8, 12 in the 4-stat-per-modality layout)
        for mod_idx in range(4):
            col = mod_idx * 4  # mean column for each modality
            means = features[:, col]
            assert torch.all(means >= 0.0), f"Negative mean found for modality {mod_idx}"
            assert torch.all(means <= 1.0), f"Mean > 1 found for modality {mod_idx}"

    def test_centroid_computation(
        self,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
        simple_supervoxel_labels,
    ):
        """Centroids (normalized to [0,1]) should be within [0, 1]."""
        computer = NodeFeatureComputer(device=torch.device("cpu"))
        features, _, _ = computer.compute(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            supervoxel_labels=simple_supervoxel_labels,
        )

        # Centroid columns come after 4 modalities * 4 stats = 16 columns
        centroid_cols = features[:, 16:19]
        assert torch.all(centroid_cols >= 0.0), "Centroid coordinate < 0"
        assert torch.all(centroid_cols <= 1.0), "Centroid coordinate > 1"

    def test_atlas_one_hot_sums_to_one(
        self,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
        simple_supervoxel_labels,
    ):
        """The one-hot atlas region encoding should sum to 1 for each node."""
        computer = NodeFeatureComputer(device=torch.device("cpu"))
        features, _, _ = computer.compute(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            supervoxel_labels=simple_supervoxel_labels,
        )

        num_atlas_regions = int(synthetic_atlas.max()) + 1
        # Atlas one-hot starts after 16 (intensity) + 3 (centroid) + 1 (volume) = 20
        atlas_start = 20
        atlas_end = atlas_start + num_atlas_regions
        atlas_onehot = features[:, atlas_start:atlas_end]

        row_sums = atlas_onehot.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums)), (
            f"Atlas one-hot rows do not sum to 1: {row_sums}"
        )
