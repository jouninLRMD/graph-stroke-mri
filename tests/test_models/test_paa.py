"""Tests for stroke_gat.models.paa -- ProbabilisticAttentionAttribution."""

from __future__ import annotations

import pytest
import torch

from stroke_gat.models.paa import ProbabilisticAttentionAttribution


@pytest.fixture
def paa_module():
    """A PAA module with hidden_dim=32, num_lesion_types=3."""
    return ProbabilisticAttentionAttribution(hidden_dim=32, num_lesion_types=3)


@pytest.fixture
def paa_inputs():
    """Synthetic inputs for PAA forward/attribution tests."""
    num_nodes = 20
    num_edges = 60
    hidden_dim = 32
    num_lesion_types = 3

    x = torch.randn(num_nodes, hidden_dim)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    attention_weights = torch.rand(num_edges)
    connectivity_probs = torch.rand(num_nodes, num_lesion_types)
    connectivity_probs = connectivity_probs / connectivity_probs.sum(dim=1, keepdim=True)

    return x, edge_index, attention_weights, connectivity_probs


class TestProbabilisticAttentionAttribution:
    """Tests for the PAA module."""

    def test_forward_output_shape(self, paa_module, paa_inputs):
        """Output shape should match input x shape (N, hidden_dim)."""
        x, edge_index, attn, conn_probs = paa_inputs
        out = paa_module(x, edge_index, attn, conn_probs)
        assert out.shape == x.shape

    def test_attribution_scores_shape(self, paa_module, paa_inputs):
        """Attribution scores should have shape (E,) matching number of edges."""
        x, edge_index, attn, conn_probs = paa_inputs
        scores = paa_module.compute_attribution_scores(attn, conn_probs, edge_index)
        assert scores.shape == (edge_index.size(1),)

    def test_attribution_scores_nonnegative(self, paa_module, paa_inputs):
        """Attribution scores should be non-negative (product of non-negative terms)."""
        x, edge_index, attn, conn_probs = paa_inputs
        # Ensure inputs are non-negative
        attn_pos = torch.abs(attn)
        conn_pos = torch.abs(conn_probs)
        scores = paa_module.compute_attribution_scores(attn_pos, conn_pos, edge_index)
        assert torch.all(scores >= 0), "Found negative attribution scores"

    def test_without_connectivity_probs_still_works(self, paa_module):
        """When connectivity_probs=None, PAA falls back to using last dims of x."""
        num_nodes = 20
        num_edges = 60
        hidden_dim = 32

        # x needs to have at least num_lesion_types dims at the end
        x = torch.randn(num_nodes, hidden_dim)
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        edge_index = torch.stack([src, dst], dim=0)
        attn = torch.rand(num_edges)

        # Should not raise even without connectivity_probs
        out = paa_module(x, edge_index, attn, connectivity_probs=None)
        assert out.shape == x.shape

    def test_forward_with_multihead_attention(self, paa_module):
        """PAA should handle multi-head attention weights (E, H)."""
        num_nodes = 20
        num_edges = 60
        hidden_dim = 32
        num_heads = 4

        x = torch.randn(num_nodes, hidden_dim)
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        edge_index = torch.stack([src, dst], dim=0)
        attn = torch.rand(num_edges, num_heads)  # multi-head
        conn_probs = torch.rand(num_nodes, 3)
        conn_probs = conn_probs / conn_probs.sum(dim=1, keepdim=True)

        out = paa_module(x, edge_index, attn, conn_probs)
        assert out.shape == x.shape
