"""Tests for stroke_gat.models.gat -- StrokeGAT model."""

from __future__ import annotations

import pytest
import torch

from stroke_gat.config import ModelConfig
from stroke_gat.models.gat import StrokeGAT


@pytest.fixture
def small_model_config() -> ModelConfig:
    """A compact model config for fast testing."""
    return ModelConfig(
        num_layers=3,
        num_heads=4,
        hidden_dim=32,
        dropout=0.0,
        negative_slope=0.2,
        num_classes=3,
        use_paa=True,
        use_edge_weights=True,
    )


@pytest.fixture
def small_graph():
    """A small graph with 20 nodes, 60 edges, 16-dim features."""
    num_nodes = 20
    in_channels = 16
    x = torch.randn(num_nodes, in_channels)
    # Build undirected edges
    src = torch.randint(0, num_nodes, (30,))
    dst = torch.randint(0, num_nodes, (30,))
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    )
    connectivity_probs = torch.rand(num_nodes, 3)
    # Normalize to probabilities
    connectivity_probs = connectivity_probs / connectivity_probs.sum(dim=1, keepdim=True)
    return x, edge_index, connectivity_probs, in_channels


class TestStrokeGAT:
    """Tests for the StrokeGAT forward pass and utilities."""

    def test_forward_output_shape(self, small_model_config, small_graph):
        """Output should be (N, num_classes)."""
        x, edge_index, conn_probs, in_channels = small_graph
        model = StrokeGAT(in_channels=in_channels, config=small_model_config)
        model.eval()

        with torch.no_grad():
            out = model(x, edge_index, connectivity_probs=conn_probs)

        assert out.shape == (x.size(0), small_model_config.num_classes)

    def test_forward_with_edge_weight(self, small_model_config, small_graph):
        """Forward pass should accept edge_weight without error."""
        x, edge_index, conn_probs, in_channels = small_graph
        model = StrokeGAT(in_channels=in_channels, config=small_model_config)
        model.eval()

        edge_weight = torch.rand(edge_index.size(1))
        with torch.no_grad():
            out = model(x, edge_index, edge_weight=edge_weight, connectivity_probs=conn_probs)

        assert out.shape == (x.size(0), small_model_config.num_classes)

    def test_return_attention_flag(self, small_model_config, small_graph):
        """When return_attention=True, forward returns (logits, list_of_alphas)."""
        x, edge_index, conn_probs, in_channels = small_graph
        model = StrokeGAT(in_channels=in_channels, config=small_model_config)
        model.eval()

        with torch.no_grad():
            result = model(x, edge_index, connectivity_probs=conn_probs, return_attention=True)

        assert isinstance(result, tuple), "Expected tuple when return_attention=True"
        logits, attn_list = result
        assert logits.shape == (x.size(0), small_model_config.num_classes)
        assert isinstance(attn_list, list)
        assert len(attn_list) == small_model_config.num_layers

    def test_gradient_flow(self, small_model_config, small_graph):
        """loss.backward() should not raise, and all parameters should have gradients."""
        x, edge_index, conn_probs, in_channels = small_graph
        model = StrokeGAT(in_channels=in_channels, config=small_model_config)
        model.train()

        out = model(x, edge_index, connectivity_probs=conn_probs)
        target = torch.randint(0, small_model_config.num_classes, (x.size(0),))
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_get_attention_maps_returns_three_items(self, small_model_config, small_graph):
        """get_attention_maps should return (logits, attn_weights_list, paa_scores)."""
        x, edge_index, conn_probs, in_channels = small_graph
        model = StrokeGAT(in_channels=in_channels, config=small_model_config)

        result = model.get_attention_maps(x, edge_index, connectivity_probs=conn_probs)

        assert isinstance(result, tuple)
        assert len(result) == 3
        logits, attn_list, paa_scores = result
        assert logits.shape == (x.size(0), small_model_config.num_classes)
        assert isinstance(attn_list, list)
        # PAA scores should exist since use_paa=True and connectivity_probs provided
        if paa_scores is not None:
            assert paa_scores.dim() == 1  # (E,)
