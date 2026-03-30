"""Tests for stroke_gat.graph.builder -- GraphBuilder."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from stroke_gat.config import GraphConfig, SLICConfig
from stroke_gat.graph.builder import GraphBuilder


@pytest.fixture
def small_graph_config() -> GraphConfig:
    return GraphConfig(batch_size_nodes=512, batch_size_edges=4)


@pytest.fixture
def small_slic_config() -> SLICConfig:
    return SLICConfig(
        target_supervoxel_size=64,
        max_iterations=2,
        min_cluster_size=5,
    )


class TestGraphBuilder:
    """Tests for the full graph construction pipeline."""

    def test_build_returns_data_object(
        self,
        small_slic_config,
        small_graph_config,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
    ):
        """GraphBuilder.build should return a PyG Data object."""
        builder = GraphBuilder(
            slic_config=small_slic_config,
            graph_config=small_graph_config,
            device=torch.device("cpu"),
        )
        data, sv_labels = builder.build(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            subject_id="test_sub",
        )
        assert isinstance(data, Data)

    def test_graph_has_required_attributes(
        self,
        small_slic_config,
        small_graph_config,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
    ):
        """The resulting Data object must have x, edge_index, and y."""
        builder = GraphBuilder(
            slic_config=small_slic_config,
            graph_config=small_graph_config,
            device=torch.device("cpu"),
        )
        data, _ = builder.build(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            subject_id="test_sub",
        )
        assert data.x is not None, "Missing node features (x)"
        assert data.edge_index is not None, "Missing edge_index"
        assert data.y is not None, "Missing node labels (y)"

    def test_edge_index_valid(
        self,
        small_slic_config,
        small_graph_config,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
    ):
        """All edge indices must be in [0, num_nodes)."""
        builder = GraphBuilder(
            slic_config=small_slic_config,
            graph_config=small_graph_config,
            device=torch.device("cpu"),
        )
        data, _ = builder.build(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            subject_id="test_sub",
        )
        num_nodes = data.x.size(0)
        assert torch.all(data.edge_index >= 0)
        assert torch.all(data.edge_index < num_nodes)

    def test_graph_is_undirected(
        self,
        small_slic_config,
        small_graph_config,
        synthetic_modalities,
        synthetic_masks,
        synthetic_atlas,
    ):
        """For every edge (i,j), the reverse edge (j,i) should also exist."""
        builder = GraphBuilder(
            slic_config=small_slic_config,
            graph_config=small_graph_config,
            device=torch.device("cpu"),
        )
        data, _ = builder.build(
            modalities=synthetic_modalities,
            masks=synthetic_masks,
            atlas=synthetic_atlas,
            subject_id="test_sub",
        )
        ei = data.edge_index
        # Build set of edges
        edge_set = set()
        for col in range(ei.size(1)):
            edge_set.add((ei[0, col].item(), ei[1, col].item()))

        for col in range(ei.size(1)):
            src, dst = ei[0, col].item(), ei[1, col].item()
            assert (dst, src) in edge_set, (
                f"Edge ({src},{dst}) exists but reverse ({dst},{src}) does not"
            )
