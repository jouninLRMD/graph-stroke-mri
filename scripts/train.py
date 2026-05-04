#!/usr/bin/env python
"""Train a StrokeGAT model on pre-computed supervoxel graphs.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 100 --lr 0.0005
    python scripts/train.py --config configs/default.yaml --resume outputs/checkpoints/latest_checkpoint.pt
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
from stroke_gat.training.trainer import Trainer
from stroke_gat.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train StrokeGAT model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    config = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Set up data
    data_module = StrokeDataModule(config.training, config.paths)
    data_module.setup()

    # Build model
    in_dim = data_module.get_input_dim()
    logger.info("Input feature dimension: %d", in_dim)

    model = StrokeGAT(in_channels=in_dim, config=config.model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", num_params)

    # Train
    trainer = Trainer(
        model=model,
        data_module=data_module,
        training_config=config.training,
        model_config=config.model,
        device=device,
        output_dir=args.output_dir,
    )

    results = trainer.train(resume_from=args.resume)

    # Test
    test_results = trainer.test()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter non-serializable values from results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(make_serializable(test_results), f, indent=2)

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
