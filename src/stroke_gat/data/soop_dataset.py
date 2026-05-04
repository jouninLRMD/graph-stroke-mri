"""PyTorch Geometric Dataset for pre-computed stroke brain graphs."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset

logger = logging.getLogger(__name__)


class SOOPGraphDataset(Dataset):
    """Dataset that loads pre-computed graph .pt files for stroke brain MRI.

    Each file contains a torch_geometric.data.Data object with:
        - x: Node features (N, F)
        - edge_index: Edge connectivity (2, E)
        - edge_attr: Edge attributes (E, D)
        - edge_weight: Edge weights (E,)
        - y: Node-level lesion labels (N,) with values in {0, 1, 2}
        - parcellation_labels: Atlas region per node (N,)
        - lesion_level: Graph-level lesion type string
        - stroke_status: Graph-level stroke/non-stroke string
    """

    def __init__(
        self,
        file_list: list[str | Path],
        transform=None,
        pre_transform=None,
    ):
        """Initialize dataset from a list of .pt file paths.

        Args:
            file_list: Paths to pre-computed graph files.
            transform: Optional runtime transform.
            pre_transform: Optional pre-processing transform.
        """
        self._file_list = [str(p) for p in file_list]
        self._transform = transform
        self._pre_transform = pre_transform

    def len(self) -> int:
        return len(self._file_list)

    def get(self, idx: int) -> Data:
        data = torch.load(self._file_list[idx], weights_only=False)

        if self._pre_transform is not None:
            data = self._pre_transform(data)
        if self._transform is not None:
            data = self._transform(data)

        return data

    @property
    def file_paths(self) -> list[str]:
        return list(self._file_list)
