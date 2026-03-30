"""Training loop for StrokeGAT with early stopping and LR scheduling.

Implements the training procedure from Section 3.5:
- Adam optimizer (lr=0.001, betas=(0.9, 0.999))
- Exponential LR decay (gamma=0.95)
- Early stopping (patience=20)
- L2 regularization (weight_decay=5e-4)
- Model selection by composite score: 0.5*Dice + 0.3*AUC + 0.2*Sensitivity
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from stroke_gat.config import ModelConfig, TrainingConfig
from stroke_gat.data.loader import StrokeDataModule
from stroke_gat.models.gat import StrokeGAT
from stroke_gat.models.loss import CombinedLoss
from stroke_gat.training.callbacks import EarlyStopping, ModelCheckpoint
from stroke_gat.training.metrics import MetricsComputer

logger = logging.getLogger(__name__)


class Trainer:
    """Trains and evaluates a StrokeGAT model.

    Args:
        model: The GAT model.
        data_module: Data module providing train/val/test loaders.
        training_config: Training hyperparameters.
        model_config: Model architecture config.
        device: torch device.
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: StrokeGAT,
        data_module: StrokeDataModule,
        training_config: TrainingConfig,
        model_config: ModelConfig | None = None,
        device: torch.device | None = None,
        output_dir: str | Path = "outputs",
    ):
        self.model = model
        self.data_module = data_module
        self.config = training_config
        self.model_config = model_config or ModelConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)

        self.model.to(self.device)

        # Loss function with class weights
        class_weights = data_module.get_class_weights().to(self.device)
        self.criterion = CombinedLoss(class_weights=class_weights)

        # Optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=training_config.lr,
            betas=(0.9, 0.999),
            weight_decay=training_config.weight_decay,
        )
        self.scheduler = ExponentialLR(self.optimizer, gamma=training_config.lr_decay_rate)

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=training_config.early_stopping_patience,
            min_delta=training_config.early_stopping_min_delta,
        )
        self.checkpoint = ModelCheckpoint(
            save_dir=self.output_dir / "checkpoints",
            monitor="composite_score",
        )

        # Metrics
        self.metrics = MetricsComputer(num_classes=self.model_config.num_classes)

        # History
        self.history: list[dict] = []

    def train(self, resume_from: str | Path | None = None) -> dict:
        """Run the full training loop.

        Args:
            resume_from: Optional checkpoint path to resume training.

        Returns:
            Dict with best metrics and training history.
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = ModelCheckpoint.load(
                resume_from, self.model, self.optimizer, self.scheduler
            )
            logger.info("Resumed training from epoch %d", start_epoch)

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        logger.info(
            "Starting training: %d epochs, device=%s, lr=%.4f",
            self.config.epochs,
            self.device,
            self.config.lr,
        )

        best_metrics = {}

        for epoch in range(start_epoch, self.config.epochs):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._evaluate(val_loader)
            val_loss = val_metrics.pop("loss")

            # Composite score for model selection
            composite = MetricsComputer.composite_score(
                val_metrics, self.config.composite_weights
            )
            val_metrics["composite_score"] = composite

            # LR scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Log
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "composite_score": composite,
                "dice_mean": val_metrics.get("dice_mean", 0),
                "auc_roc": val_metrics.get("auc_roc", 0),
                "balanced_accuracy": val_metrics.get("balanced_accuracy", 0),
                "lr": current_lr,
                "time": epoch_time,
            }
            self.history.append(record)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "composite=%.4f  dice=%.4f  auc=%.4f  lr=%.6f  (%.1fs)",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss,
                composite,
                val_metrics.get("dice_mean", 0),
                val_metrics.get("auc_roc", 0),
                current_lr,
                epoch_time,
            )

            # Checkpointing
            if self.checkpoint.step(composite, self.model, self.optimizer, epoch, val_metrics):
                best_metrics = val_metrics.copy()

            self.checkpoint.save_latest(
                self.model, self.optimizer, self.scheduler, epoch, val_metrics
            )

            # Early stopping
            if self.early_stopping.step(composite):
                logger.info("Training stopped early at epoch %d", epoch + 1)
                break

        # Load best model for final evaluation
        if self.checkpoint.best_path is not None:
            ModelCheckpoint.load(self.checkpoint.best_path, self.model)
            logger.info("Loaded best model from %s", self.checkpoint.best_path)

        return {"best_metrics": best_metrics, "history": self.history}

    def _train_epoch(self, loader) -> float:
        """Run a single training epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(
                batch.x,
                batch.edge_index,
                connectivity_probs=getattr(batch, "connectivity_probs", None),
            )

            loss = self.criterion(logits, batch.y)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader) -> dict:
        """Evaluate the model on a data loader.

        Returns:
            Dict with loss and all computed metrics.
        """
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            logits = self.model(
                batch.x,
                batch.edge_index,
                connectivity_probs=getattr(batch, "connectivity_probs", None),
            )

            loss = self.criterion(logits, batch.y)
            total_loss += loss.item()
            num_batches += 1

            self.metrics.update(logits, batch.y)

        results = self.metrics.compute()
        results["loss"] = total_loss / max(num_batches, 1)
        return results

    def test(self) -> dict:
        """Evaluate the model on the test set.

        Returns:
            Dict with test metrics.
        """
        test_loader = self.data_module.test_dataloader()
        results = self._evaluate(test_loader)
        loss = results.pop("loss")

        composite = MetricsComputer.composite_score(
            results, self.config.composite_weights
        )
        results["composite_score"] = composite
        results["test_loss"] = loss

        logger.info("Test results:")
        logger.info("  Loss: %.4f", loss)
        logger.info("  Composite: %.4f", composite)
        logger.info("  Dice mean: %.4f", results.get("dice_mean", 0))
        logger.info("  AUC-ROC: %.4f", results.get("auc_roc", 0))
        logger.info("  Balanced accuracy: %.4f", results.get("balanced_accuracy", 0))
        logger.info("  Sensitivity (macro): %.4f", results.get("sensitivity_macro", 0))

        for name in MetricsComputer.CLASS_NAMES:
            logger.info(
                "  %s — dice=%.4f  sens=%.4f  spec=%.4f  prec=%.4f  f1=%.4f",
                name,
                results.get(f"{name}_dice", 0),
                results.get(f"{name}_sensitivity", 0),
                results.get(f"{name}_specificity", 0),
                results.get(f"{name}_precision", 0),
                results.get(f"{name}_f1", 0),
            )

        return results
