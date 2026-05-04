"""Tests for stroke_gat.data.soop_dataset -- SOOPGraphDataset."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from torch_geometric.data import Data

from stroke_gat.data.soop_dataset import SOOPGraphDataset


@pytest.fixture
def tmp_graph_files(tmp_path):
    """Create temporary .pt graph files for testing the dataset."""
    file_paths = []
    for i in range(5):
        data = Data(
            x=torch.randn(10 + i, 20),
            edge_index=torch.stack([
                torch.randint(0, 10 + i, (30,)),
                torch.randint(0, 10 + i, (30,)),
            ]),
            y=torch.randint(0, 3, (10 + i,)),
        )
        data.subject_id = f"sub_{i:03d}"
        data.lesion_level = "acute" if i % 2 == 0 else "chronic"
        fpath = tmp_path / f"sub_{i:03d}_supervoxel_graph.pt"
        torch.save(data, str(fpath))
        file_paths.append(str(fpath))
    return file_paths


class TestSOOPGraphDataset:
    """Tests for the SOOPGraphDataset wrapper."""

    def test_soop_dataset_length(self, tmp_graph_files):
        """Dataset length should match the number of provided files."""
        dataset = SOOPGraphDataset(tmp_graph_files)
        assert len(dataset) == len(tmp_graph_files)

    def test_soop_dataset_get_returns_data(self, tmp_graph_files):
        """dataset.get(idx) should return a torch_geometric Data object."""
        dataset = SOOPGraphDataset(tmp_graph_files)
        for idx in range(len(dataset)):
            item = dataset.get(idx)
            assert isinstance(item, Data)
            assert item.x is not None
            assert item.edge_index is not None
            assert item.y is not None

    @pytest.mark.parametrize("idx", [0, 2, 4])
    def test_soop_dataset_specific_indices(self, tmp_graph_files, idx):
        """Spot-check that specific indices load successfully."""
        dataset = SOOPGraphDataset(tmp_graph_files)
        item = dataset.get(idx)
        assert isinstance(item, Data)
        assert item.x.size(0) == 10 + idx  # Matches our fixture construction

    def test_soop_dataset_empty_list(self):
        """An empty file list should produce a zero-length dataset."""
        dataset = SOOPGraphDataset([])
        assert len(dataset) == 0

    def test_soop_dataset_file_paths_property(self, tmp_graph_files):
        """file_paths property should return the list of file paths."""
        dataset = SOOPGraphDataset(tmp_graph_files)
        assert dataset.file_paths == tmp_graph_files
