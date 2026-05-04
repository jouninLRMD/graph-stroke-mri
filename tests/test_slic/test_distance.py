"""Tests for stroke_gat.slic.distance -- numba distance functions."""

from __future__ import annotations

import numpy as np
import pytest

from stroke_gat.slic.distance import (
    atlas_penalty,
    combined_distance,
    multimodal_intensity_distance,
    spatial_distance,
)


# ---------------------------------------------------------------------------
# combined_distance
# ---------------------------------------------------------------------------

class TestCombinedDistance:
    """Tests for the full anatomically-constrained distance (Eq. 5)."""

    @staticmethod
    def _default_args(**overrides):
        """Return default kwargs for combined_distance, with optional overrides."""
        defaults = dict(
            voxel_intensities=np.array([0.5, 0.5, 0.5, 0.5]),
            center_intensities=np.array([0.5, 0.5, 0.5, 0.5]),
            modality_weights=np.array([0.25, 0.25, 0.25, 0.25]),
            voxel_coord=np.array([10.0, 10.0, 10.0]),
            center_coord=np.array([10.0, 10.0, 10.0]),
            grid_spacing=6.0,
            compactness=0.1,
            voxel_atlas=1,
            center_atlas=1,
            atlas_lambda=0.5,
        )
        defaults.update(overrides)
        return defaults

    def test_combined_distance_same_center_is_zero(self):
        """When voxel == center in all dimensions, distance should be 0."""
        dist = combined_distance(**self._default_args())
        assert dist == pytest.approx(0.0, abs=1e-12)

    def test_combined_distance_increases_with_spatial_distance(self):
        """Moving the voxel farther from the center should increase distance."""
        near = combined_distance(
            **self._default_args(voxel_coord=np.array([11.0, 10.0, 10.0]))
        )
        far = combined_distance(
            **self._default_args(voxel_coord=np.array([20.0, 10.0, 10.0]))
        )
        assert far > near > 0.0

    @pytest.mark.parametrize(
        "voxel_atlas,center_atlas,expected",
        [
            (1, 1, 0.0),
            (2, 2, 0.0),
            (0, 0, 0.0),
        ],
    )
    def test_atlas_penalty_same_region_is_zero(self, voxel_atlas, center_atlas, expected):
        """When voxel and center share the same atlas label, penalty is 0."""
        assert atlas_penalty(voxel_atlas, center_atlas) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "voxel_atlas,center_atlas",
        [(1, 2), (3, 0), (0, 5)],
    )
    def test_atlas_penalty_different_region_adds_lambda(self, voxel_atlas, center_atlas):
        """When atlas labels differ, combined distance includes lambda * 1.0."""
        atlas_lambda = 0.5
        dist_same = combined_distance(
            **self._default_args(voxel_atlas=1, center_atlas=1, atlas_lambda=atlas_lambda)
        )
        dist_diff = combined_distance(
            **self._default_args(
                voxel_atlas=voxel_atlas,
                center_atlas=center_atlas,
                atlas_lambda=atlas_lambda,
            )
        )
        assert dist_diff == pytest.approx(dist_same + atlas_lambda, abs=1e-12)

    def test_modality_weights_affect_distance(self):
        """Changing modality weights should change the resulting distance."""
        common = dict(
            voxel_intensities=np.array([1.0, 0.0, 0.0, 0.0]),
            center_intensities=np.array([0.0, 0.0, 0.0, 0.0]),
            voxel_coord=np.array([10.0, 10.0, 10.0]),
            center_coord=np.array([10.0, 10.0, 10.0]),
            grid_spacing=6.0,
            compactness=0.1,
            voxel_atlas=1,
            center_atlas=1,
            atlas_lambda=0.5,
        )
        # All weight on first modality
        dist_heavy = combined_distance(
            modality_weights=np.array([1.0, 0.0, 0.0, 0.0]), **common
        )
        # Equal weights
        dist_equal = combined_distance(
            modality_weights=np.array([0.25, 0.25, 0.25, 0.25]), **common
        )
        # First modality intensity differs by 1.0:
        #   heavy: sqrt(1.0 * 1^2) = 1.0
        #   equal: sqrt(0.25 * 1^2) = 0.5
        assert dist_heavy == pytest.approx(1.0, abs=1e-6)
        assert dist_equal == pytest.approx(0.5, abs=1e-6)
        assert dist_heavy > dist_equal


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

class TestMultimodalIntensityDistance:
    def test_same_intensities_zero(self):
        v = np.array([0.5, 0.3, 0.7, 0.1])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert multimodal_intensity_distance(v, v, w) == pytest.approx(0.0)

    def test_known_value(self):
        v = np.array([1.0, 0.0])
        c = np.array([0.0, 0.0])
        w = np.array([1.0, 0.0])
        assert multimodal_intensity_distance(v, c, w) == pytest.approx(1.0)


class TestSpatialDistance:
    def test_same_point_zero(self):
        p = np.array([5.0, 5.0, 5.0])
        assert spatial_distance(p, p) == pytest.approx(0.0)

    def test_unit_distance(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert spatial_distance(a, b) == pytest.approx(1.0)
