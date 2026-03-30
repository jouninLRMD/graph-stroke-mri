#!/usr/bin/env python
"""Evaluate a trained StrokeGAT model on the test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from stroke_gat.config import load_config
from stroke_gat.data.loader import StrokeDataModule
from stroke_gat.models.gat import StrokeGAT
from stroke_gat.training.callbacks import ModelCheckpoint
from stroke_gat.training.metrics import MetricsComputer
from stroke_gat.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate StrokeGAT model on test set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML.")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Output directory.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    config = load_config(args.config)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data
    data_module = StrokeDataModule(config.training, config.paths)
    data_module.setup()

    # Build and load model
    in_dim = data_module.get_input_dim()
    model = StrokeGAT(in_channels=in_dim, config=config.model)
    ModelCheckpoint.load(args.checkpoint, model)
    model.to(device)
    model.eval()

    logger.info("Loaded model from %s", args.checkpoint)

    # Evaluate
    test_loader = data_module.test_dataloader()
    metrics = MetricsComputer(num_classes=config.model.num_classes)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index,
                connectivity_probs=getattr(batch, "connectivity_probs", None),
            )
            metrics.update(logits, batch.y)

    results = metrics.compute()
    composite = MetricsComputer.composite_score(results, config.training.composite_weights)
    results["composite_score"] = composite

    # Print results
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info("Composite score: %.4f", composite)
    logger.info("Dice mean:       %.4f", results.get("dice_mean", 0))
    logger.info("AUC-ROC:         %.4f", results.get("auc_roc", 0))
    logger.info("Balanced acc:    %.4f", results.get("balanced_accuracy", 0))
    logger.info("F1 macro:        %.4f", results.get("f1_macro", 0))
    logger.info("Sensitivity:     %.4f", results.get("sensitivity_macro", 0))
    logger.info("Precision:       %.4f", results.get("precision_macro", 0))
    logger.info("-" * 60)

    for name in MetricsComputer.CLASS_NAMES:
        logger.info(
            "%s: dice=%.4f  sens=%.4f  spec=%.4f  prec=%.4f  f1=%.4f",
            name.ljust(12),
            results.get(f"{name}_dice", 0),
            results.get(f"{name}_sensitivity", 0),
            results.get(f"{name}_specificity", 0),
            results.get(f"{name}_precision", 0),
            results.get(f"{name}_f1", 0),
        )

    logger.info("-" * 60)
    logger.info("Confusion Matrix:")
    cm = results.get("confusion_matrix", [])
    for row in cm:
        logger.info("  %s", row)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    logger.info("Results saved to %s", output_dir / "test_results.json")


if __name__ == "__main__":
    main()
