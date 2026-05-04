"""GAT classifier for supervoxel nodes.

Architecture from Section 3.3: three GATConv layers with 8 attention heads
each, hidden dimension 128 (16 per head), LeakyReLU(0.2), Dropout(0.1).
The first two layers concatenate heads (Eq. 10) and the last averages them
(Eq. 11). The PAA module sits between the second and third layers (Eq. 12).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from stroke_gat.config import ModelConfig
from stroke_gat.models.paa import ProbabilisticAttentionAttribution

logger = logging.getLogger(__name__)


class StrokeGAT(nn.Module):
    """GAT with optional PAA, classifying each supervoxel node into
    {no lesion (0), acute (1), chronic (2)}.
    """

    def __init__(
        self,
        in_channels: int,
        config: ModelConfig | None = None,
    ):
        """Initialize StrokeGAT.

        Args:
            in_channels: Dimension of input node features.
            config: Model configuration. If None, uses defaults.
        """
        super().__init__()
        cfg = config or ModelConfig()

        self.num_layers = cfg.num_layers
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.dropout = cfg.dropout
        self.num_classes = cfg.num_classes
        self.use_paa = cfg.use_paa

        head_dim = cfg.hidden_dim // cfg.num_heads

        # Build GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Layer 1: input -> hidden (multi-head, concatenated)
        self.convs.append(
            GATConv(
                in_channels,
                head_dim,
                heads=cfg.num_heads,
                dropout=cfg.dropout,
                negative_slope=cfg.negative_slope,
                concat=True,
            )
        )
        self.norms.append(nn.LayerNorm(cfg.hidden_dim))

        # Intermediate layers: hidden -> hidden (multi-head, concatenated)
        for _ in range(cfg.num_layers - 2):
            self.convs.append(
                GATConv(
                    cfg.hidden_dim,
                    head_dim,
                    heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    negative_slope=cfg.negative_slope,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))

        # Final layer: hidden -> num_classes (single head, averaged)
        self.convs.append(
            GATConv(
                cfg.hidden_dim,
                cfg.num_classes,
                heads=1,
                concat=False,
                dropout=cfg.dropout,
                negative_slope=cfg.negative_slope,
            )
        )

        # Input projection for residual when dimensions differ
        if in_channels != cfg.hidden_dim:
            self.input_proj = nn.Linear(in_channels, cfg.hidden_dim)
        else:
            self.input_proj = None

        # PAA module (sits before the final layer)
        if cfg.use_paa:
            self.paa = ProbabilisticAttentionAttribution(
                hidden_dim=cfg.hidden_dim,
                num_lesion_types=cfg.num_classes,
            )
        else:
            self.paa = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        connectivity_probs: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the GAT.

        Args:
            x: (N, in_channels) node features.
            edge_index: (2, E) edge connectivity.
            edge_weight: (E,) optional edge weights.
            connectivity_probs: (N, K) optional connectivity probability vectors
                                for PAA. Required if use_paa=True.
            return_attention: If True, also return attention weights from each layer.

        Returns:
            If return_attention=False: (N, num_classes) logits.
            If return_attention=True: tuple of (logits, list_of_attention_weights).
        """
        attention_weights_all = []

        # Process through intermediate GAT layers
        for i in range(len(self.convs) - 1):
            x_in = x

            out, (edge_idx, alpha) = self.convs[i](
                x, edge_index, return_attention_weights=True
            )
            if return_attention:
                attention_weights_all.append(alpha.detach())

            out = self.norms[i](out)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = F.dropout(out, p=self.dropout, training=self.training)

            # Residual connection (after first layer, project input if needed)
            if i == 0 and self.input_proj is not None:
                x_in = self.input_proj(x_in)
            if x_in.size(1) == out.size(1):
                out = out + x_in

            x = out

        # Apply PAA before final layer
        if self.paa is not None and attention_weights_all:
            last_alpha = attention_weights_all[-1]
            x = self.paa(x, edge_index, last_alpha, connectivity_probs)

        # Final classification layer
        out, (edge_idx, alpha) = self.convs[-1](
            x, edge_index, return_attention_weights=True
        )
        if return_attention:
            attention_weights_all.append(alpha.detach())

        if return_attention:
            return out, attention_weights_all
        return out

    def get_attention_maps(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        connectivity_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        """Run inference and return attention weights + PAA attribution.

        Args:
            x: Node features.
            edge_index: Edge connectivity.
            connectivity_probs: Optional connectivity probs for PAA.

        Returns:
            Tuple of (logits, attention_weights_per_layer, paa_attribution_scores).
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(
                x, edge_index, connectivity_probs=connectivity_probs, return_attention=True
            )

            paa_scores = None
            if self.paa is not None and connectivity_probs is not None and attn_weights:
                paa_scores = self.paa.compute_attribution_scores(
                    attn_weights[-2] if len(attn_weights) > 1 else attn_weights[-1],
                    connectivity_probs,
                    edge_index,
                )

        return logits, attn_weights, paa_scores
