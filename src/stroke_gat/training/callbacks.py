"""Training callbacks: early stopping, checkpointing, LR scheduling."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'max' for metrics where higher is better (default),
              'min' for loss-like metrics.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-3, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )
                return True

        return False


class ModelCheckpoint:
    """Save model checkpoints when a monitored metric improves.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric name to monitor (for logging).
        mode: 'max' or 'min'.
    """

    def __init__(self, save_dir: str | Path, monitor: str = "composite_score", mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None
        self.best_path: Path | None = None

    def step(
        self,
        score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict | None = None,
    ) -> bool:
        """Save checkpoint if score improved.

        Returns:
            True if a new best checkpoint was saved.
        """
        is_best = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )

        if is_best:
            self.best_score = score
            self.best_path = self.save_dir / "best_model.pt"
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
                "monitor": self.monitor,
            }
            if metrics is not None:
                checkpoint["metrics"] = metrics
            torch.save(checkpoint, self.best_path)
            logger.info(
                "Saved best model (epoch %d, %s=%.4f) to %s",
                epoch,
                self.monitor,
                score,
                self.best_path,
            )
            return True
        return False

    def save_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
        metrics: dict | None = None,
    ) -> None:
        """Save a periodic checkpoint (for resuming training)."""
        path = self.save_dir / "latest_checkpoint.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if metrics is not None:
            checkpoint["metrics"] = metrics
        torch.save(checkpoint, path)

    @staticmethod
    def load(
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> int:
        """Load a checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", 0)
