"""Probabilistic Attention Attribution (Eq. 12).

For an edge (i, j),

    A_ij = alpha_ij * (p_i^T . p_j)

where alpha_ij is the learned GAT attention coefficient and p_i, p_j are the
node-level connectivity probability vectors [p0, p1, p2] over the three
lesion classes. PAA re-weights attention by the lesion-type agreement of a
node's neighborhood, which gives a more clinically meaningful saliency map.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

logger = logging.getLogger(__name__)


class ProbabilisticAttentionAttribution(nn.Module):
    """PAA layer: rescales GAT attention by lesion-type neighborhood agreement.

    A high attribution A_ij means the model both attends to the edge and the
    two endpoints have similar lesion-type neighbourhoods.
    """

    def __init__(self, hidden_dim: int, num_lesion_types: int = 3):
        """Initialize PAA module.

        Args:
            hidden_dim: Hidden dimension of node embeddings.
            num_lesion_types: Number of lesion type categories (default 3).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lesion_types = num_lesion_types
        self.message_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        attention_weights: torch.Tensor,
        connectivity_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply PAA-modulated message passing.

        Args:
            x: (N, hidden_dim) node embeddings.
            edge_index: (2, E) edge connectivity.
            attention_weights: (E,) or (E, H) attention coefficients from GAT.
            connectivity_probs: (N, num_lesion_types) probability vectors.
                                If None, extracts from last dims of x.

        Returns:
            (N, hidden_dim) updated node embeddings with residual connection.
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # Get connectivity probability vectors
        if connectivity_probs is not None:
            p_src = connectivity_probs[src]  # (E, K)
            p_dst = connectivity_probs[dst]  # (E, K)
        else:
            p_src = x[src, -self.num_lesion_types:]
            p_dst = x[dst, -self.num_lesion_types:]

        # Compute probabilistic similarity: p_i^T . p_j
        prob_similarity = (p_src * p_dst).sum(dim=-1)  # (E,)

        # Modulate attention weights (Eq. 12)
        if attention_weights.dim() == 2:
            # Multi-head: average across heads first
            alpha = attention_weights.mean(dim=-1)  # (E,)
        else:
            alpha = attention_weights

        paa_weights = alpha * prob_similarity  # (E,)

        # Normalize per destination node
        paa_weights = softmax(paa_weights, dst, num_nodes=num_nodes)  # (E,)

        # Compute messages
        messages = self.message_proj(x[src])  # (E, hidden_dim)
        weighted_messages = messages * paa_weights.unsqueeze(-1)  # (E, hidden_dim)

        # Aggregate
        aggregated = scatter_add(weighted_messages, dst, dim=0, dim_size=num_nodes)  # (N, hidden_dim)

        # Gated residual connection
        gate_input = torch.cat([x, aggregated], dim=-1)  # (N, 2*hidden_dim)
        gate_value = torch.sigmoid(self.gate(gate_input))  # (N, 1)

        return x + gate_value * aggregated

    def compute_attribution_scores(
        self,
        attention_weights: torch.Tensor,
        connectivity_probs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-edge attribution scores for visualization.

        A_ij = alpha_ij * (p_i^T . p_j)

        Args:
            attention_weights: (E,) or (E, H) attention coefficients.
            connectivity_probs: (N, K) probability vectors.
            edge_index: (2, E) edge connectivity.

        Returns:
            (E,) tensor of attribution scores.
        """
        src, dst = edge_index[0], edge_index[1]

        p_src = connectivity_probs[src]
        p_dst = connectivity_probs[dst]
        prob_sim = (p_src * p_dst).sum(dim=-1)

        if attention_weights.dim() == 2:
            alpha = attention_weights.mean(dim=-1)
        else:
            alpha = attention_weights

        return alpha * prob_sim
