#!/usr/bin/env python
"""Generate visualization figures for a trained model and subject data.

Usage:
    python scripts/visualize.py --checkpoint outputs/checkpoints/best_model.pt \
        --graph data/graphs/sub-0001_supervoxel_graph.pt \
        --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from stroke_gat.config import load_config
from stroke_gat.models.gat import StrokeGAT
from stroke_gat.training.callbacks import ModelCheckpoint
from stroke_gat.utils.logging import setup_logging
from stroke_gat.visualization.attribution import (
    plot_graph_attention_attribution,
    plot_paa_attribution_map,
    plot_paa_weight_distribution,
)
from stroke_gat.visualization.graph_3d import plot_graph_3d_interactive

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures.")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint.")
    parser.add_argument("--graph", type=str, required=True, help="Path to a .pt graph file.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    config = load_config(args.config)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    data = torch.load(args.graph, weights_only=False)
    logger.info("Loaded graph: %d nodes, %d edges", data.num_nodes, data.num_edges)

    # 3D graph visualization
    logger.info("Generating 3D graph visualization...")
    fig_3d = plot_graph_3d_interactive(data, title="Brain Supervoxel Graph")
    fig_3d.write_html(str(output_dir / "graph_3d.html"))
    logger.info("  Saved graph_3d.html")

    # If checkpoint provided, run inference for attention/PAA maps
    if args.checkpoint:
        in_dim = data.x.size(1)
        model = StrokeGAT(in_channels=in_dim, config=config.model)
        ModelCheckpoint.load(args.checkpoint, model)
        model.to(device)
        model.eval()

        data_dev = data.to(device)
        logits, attn_weights, paa_scores = model.get_attention_maps(
            data_dev.x, data_dev.edge_index,
            connectivity_probs=getattr(data_dev, "connectivity_probs", None),
        )

        # PAA attribution graph
        if paa_scores is not None:
            logger.info("Generating PAA attribution graph...")
            fig_paa = plot_graph_attention_attribution(data, paa_scores, title="PAA Attribution")
            fig_paa.write_html(str(output_dir / "paa_attribution_3d.html"))
            logger.info("  Saved paa_attribution_3d.html")

            # PAA weight distribution
            logger.info("Generating PAA weight distribution...")
            fig_dist = plot_paa_weight_distribution(paa_scores, data.edge_index, data.y)
            fig_dist.savefig(str(output_dir / "paa_distribution.png"), dpi=150, bbox_inches="tight")
            plt.close(fig_dist)
            logger.info("  Saved paa_distribution.png")

        # Predictions summary
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        from stroke_gat.training.metrics import MetricsComputer

        mc = MetricsComputer(num_classes=config.model.num_classes)
        mc.update(logits.cpu(), data.y.cpu())
        metrics = mc.compute()

        logger.info("Subject metrics: dice=%.4f, auc=%.4f, balanced_acc=%.4f",
                     metrics.get("dice_mean", 0), metrics.get("auc_roc", 0),
                     metrics.get("balanced_accuracy", 0))

    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
