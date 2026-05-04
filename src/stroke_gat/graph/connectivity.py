"""Probabilistic connectivity feature computation for graph nodes.

Computes per-node probability vectors [p0, p1, p2] where:
  p0 = proportion of neighbors with no lesion (class 0)
  p1 = proportion of neighbors with acute lesion (class 1)
  p2 = proportion of neighbors with chronic lesion (class 2)

These features capture the local lesion distribution in each node's
spatial neighborhood, providing contextual information for the GAT
attention mechanism and the PAA module.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def compute_connectivity_probabilities(
    num_nodes: int,
    edge_index: torch.Tensor,
    lesion_labels: torch.Tensor,
    num_lesion_types: int = 3,
) -> torch.Tensor:
    """Compute probabilistic connectivity feature vectors per node.

    For each node, counts the lesion type distribution among its
    neighbors and normalizes to a probability distribution.

    Args:
        num_nodes: Total number of nodes in the graph.
        edge_index: (2, E) edge index tensor.
        lesion_labels: (N,) integer tensor of lesion labels per node.
        num_lesion_types: Number of distinct lesion types (default 3).

    Returns:
        (N, num_lesion_types) tensor of neighbor lesion type probabilities.
    """
    src, dst = edge_index[0], edge_index[1]

    # Count neighbor lesion types for each node
    counts = torch.zeros(num_nodes, num_lesion_types, dtype=torch.float32)

    # For each edge (src -> dst), the neighbor of src is dst
    dst_onehot = torch.nn.functional.one_hot(
        lesion_labels[dst].long(), num_classes=num_lesion_types
    ).float()
    counts.index_add_(0, src, dst_onehot)

    # For each edge (src -> dst), the neighbor of dst is src
    src_onehot = torch.nn.functional.one_hot(
        lesion_labels[src].long(), num_classes=num_lesion_types
    ).float()
    counts.index_add_(0, dst, src_onehot)

    # Normalize to probabilities
    total = counts.sum(dim=1, keepdim=True)
    probabilities = counts / (total + 1e-6)

    logger.info(
        "Connectivity probabilities: shape=%s, mean_p0=%.3f, mean_p1=%.3f, mean_p2=%.3f",
        probabilities.shape,
        probabilities[:, 0].mean().item(),
        probabilities[:, 1].mean().item() if num_lesion_types > 1 else 0,
        probabilities[:, 2].mean().item() if num_lesion_types > 2 else 0,
    )

    return probabilities
