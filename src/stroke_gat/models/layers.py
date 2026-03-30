"""Custom layers for the StrokeGAT architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATBlock(nn.Module):
    """A single GAT layer with optional normalization and residual connection.

    Consists of: GATConv -> LayerNorm -> LeakyReLU -> Dropout
    With optional residual connection when input/output dimensions match.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            negative_slope=negative_slope,
            concat=concat,
        )
        actual_out = out_channels * heads if concat else out_channels
        self.norm = nn.LayerNorm(actual_out)
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Residual projection if dimensions don't match
        self.residual = residual
        if residual and in_channels != actual_out:
            self.res_proj = nn.Linear(in_channels, actual_out)
        else:
            self.res_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: (N, in_channels) node features.
            edge_index: (2, E) edge connectivity.
            return_attention: If True, return attention weights.

        Returns:
            Tuple of (output features, attention_weights or None).
        """
        identity = x

        if return_attention:
            out, (edge_idx, alpha) = self.conv(x, edge_index, return_attention_weights=True)
        else:
            out = self.conv(x, edge_index)
            alpha = None

        out = self.norm(out)
        out = F.leaky_relu(out, negative_slope=self.negative_slope)
        out = F.dropout(out, p=self.dropout, training=self.training)

        # Residual connection
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        return out, alpha
