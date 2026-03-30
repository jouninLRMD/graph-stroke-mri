"""GraphBuilder: orchestrates the full graph construction pipeline.

Pipeline:
1. Run AnatomicalSLIC to get supervoxel labels
2. Compute node features (intensity stats + centroid + atlas encoding + volume)
3. Build adjacency via 26-connectivity
4. Compute edge weights using 4-component formula
5. Compute connectivity probability vectors
6. Append connectivity probs to node features
7. Package into torch_geometric.data.Data
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from stroke_gat.config import GraphConfig, SLICConfig
from stroke_gat.graph.connectivity import compute_connectivity_probabilities
from stroke_gat.graph.edges import EdgeComputer
from stroke_gat.graph.features import NodeFeatureComputer
from stroke_gat.slic.atlas_slic import AnatomicalSLIC
from stroke_gat.utils.memory import clear_gpu_memory, log_memory

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Orchestrates the full graph construction pipeline for a single subject."""

    def __init__(
        self,
        slic_config: SLICConfig,
        graph_config: GraphConfig,
        device: torch.device | None = None,
    ):
        self.slic_config = slic_config
        self.graph_config = graph_config
        self.device = device or torch.device("cpu")
        self.slic = AnatomicalSLIC(slic_config)
        self.feature_computer = NodeFeatureComputer(device=self.device)
        self.edge_computer = EdgeComputer(graph_config, device=self.device)

    def build(
        self,
        modalities: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        atlas: np.ndarray,
        subject_id: str,
        metadata: dict | None = None,
    ) -> tuple[Data, np.ndarray]:
        """Build a complete graph for a single subject.

        Args:
            modalities: Dict of modality_name -> 3D float32 array.
            masks: Dict with 'General', 'Acute', 'Chronic' mask arrays.
            atlas: 3D integer atlas label array.
            subject_id: Subject identifier string.
            metadata: Optional dict with stroke_status, lesion_level.

        Returns:
            Tuple of (PyG Data object, supervoxel_labels_3d array).
        """
        log_memory(f"graph_build_start:{subject_id}")

        # Step 1: Supervoxel segmentation
        logger.info("Step 1: Running anatomical SLIC for %s", subject_id)
        supervoxel_labels = self.slic.segment(modalities, atlas)
        log_memory(f"after_slic:{subject_id}")

        # Step 2: Compute node features
        logger.info("Step 2: Computing node features for %s", subject_id)
        node_features, lesion_labels, parcellation_labels = self.feature_computer.compute(
            modalities=modalities,
            masks=masks,
            atlas=atlas,
            supervoxel_labels=supervoxel_labels,
            batch_size=self.graph_config.batch_size_nodes,
        )
        log_memory(f"after_features:{subject_id}")

        # Step 3: Compute edges and weights
        logger.info("Step 3: Computing edges for %s", subject_id)
        edge_index, edge_attr, edge_weight = self.edge_computer.compute(
            supervoxel_labels=supervoxel_labels,
            node_features=node_features,
            parcellation_labels=parcellation_labels,
            lesion_labels=lesion_labels,
            batch_size=self.graph_config.batch_size_edges,
        )
        log_memory(f"after_edges:{subject_id}")

        # Step 4: Compute connectivity probabilities
        logger.info("Step 4: Computing connectivity probabilities for %s", subject_id)
        conn_probs = compute_connectivity_probabilities(
            num_nodes=node_features.size(0),
            edge_index=edge_index,
            lesion_labels=lesion_labels,
            num_lesion_types=self.graph_config.num_lesion_types,
        )

        # Step 5: Append connectivity probs to node features
        node_features = torch.cat([node_features, conn_probs], dim=1)
        logger.info("Final node features shape: %s", node_features.shape)

        # Step 6: Package into Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            y=lesion_labels,
            parcellation_labels=parcellation_labels,
            connectivity_probs=conn_probs,
        )

        # Add metadata
        if metadata:
            data.lesion_level = metadata.get("lesion_level", "unknown")
            data.stroke_status = metadata.get("stroke_status", "unknown")
        data.subject_id = subject_id

        logger.info(
            "Graph for %s: %d nodes, %d edges, feature_dim=%d",
            subject_id,
            data.num_nodes,
            data.num_edges,
            data.x.size(1),
        )

        clear_gpu_memory()
        return data, supervoxel_labels

    def build_and_save(
        self,
        modalities: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        atlas: np.ndarray,
        subject_id: str,
        output_path: str | Path,
        metadata: dict | None = None,
    ) -> Data:
        """Build graph and save to disk.

        Args:
            modalities: Dict of modality_name -> 3D float32 array.
            masks: Dict with 'General', 'Acute', 'Chronic' mask arrays.
            atlas: 3D integer atlas label array.
            subject_id: Subject identifier.
            output_path: Path to save the .pt file.
            metadata: Optional metadata dict.

        Returns:
            The constructed Data object.
        """
        data, _ = self.build(modalities, masks, atlas, subject_id, metadata)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(output_path))
        logger.info("Saved graph to %s", output_path)

        return data
