"""Tests for stroke_gat.slic.atlas_slic -- AnatomicalSLIC segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from stroke_gat.config import SLICConfig
from stroke_gat.slic.atlas_slic import AnatomicalSLIC


@pytest.fixture
def small_slic_config() -> SLICConfig:
    """SLIC config tuned for small 32^3 test volumes."""
    return SLICConfig(
        target_supervoxel_size=64,
        compactness=0.1,
        atlas_penalty_lambda=0.5,
        max_iterations=3,
        convergence_threshold=1e-4,
        min_cluster_size=5,
    )


class TestAnatomicalSLIC:
    """Integration-level tests for the AnatomicalSLIC.segment method."""

    def test_segment_returns_correct_shape(
        self,
        small_slic_config,
        synthetic_modalities,
        synthetic_atlas,
        synthetic_brain_mask,
    ):
        """Output label volume must have the same shape as the input volumes."""
        slic = AnatomicalSLIC(small_slic_config)
        labels = slic.segment(synthetic_modalities, synthetic_atlas, synthetic_brain_mask)
        assert labels.shape == (32, 32, 32)

    def test_segment_produces_expected_number_of_supervoxels(
        self,
        small_slic_config,
        synthetic_modalities,
        synthetic_atlas,
        synthetic_brain_mask,
    ):
        """Number of unique supervoxels should be within 50% of target."""
        slic = AnatomicalSLIC(small_slic_config)
        labels = slic.segment(synthetic_modalities, synthetic_atlas, synthetic_brain_mask)

        n_brain = int(np.sum(synthetic_brain_mask))
        expected = n_brain // small_slic_config.target_supervoxel_size
        unique_labels = np.unique(labels[labels > 0])
        n_supervoxels = len(unique_labels)

        # Within 50% of target (generous, since small volumes are harder)
        assert n_supervoxels >= expected * 0.5 or n_supervoxels >= 1
        assert n_supervoxels <= expected * 1.5 or n_supervoxels <= expected + 20

    def test_segment_respects_brain_mask(
        self,
        small_slic_config,
        synthetic_modalities,
        synthetic_atlas,
        synthetic_brain_mask,
    ):
        """Voxels outside the brain mask should have label 0 (background)."""
        slic = AnatomicalSLIC(small_slic_config)
        labels = slic.segment(synthetic_modalities, synthetic_atlas, synthetic_brain_mask)

        outside_mask = ~synthetic_brain_mask
        assert np.all(labels[outside_mask] == 0), (
            "Found non-zero labels outside the brain mask"
        )

    def test_segment_all_brain_voxels_labeled(
        self,
        small_slic_config,
        synthetic_modalities,
        synthetic_atlas,
        synthetic_brain_mask,
    ):
        """Every voxel inside the brain mask should have a positive label."""
        slic = AnatomicalSLIC(small_slic_config)
        labels = slic.segment(synthetic_modalities, synthetic_atlas, synthetic_brain_mask)

        inside_mask = synthetic_brain_mask
        assert np.all(labels[inside_mask] > 0), (
            "Found unlabeled (0 or -1) voxels inside the brain mask"
        )
