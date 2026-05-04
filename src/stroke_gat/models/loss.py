"""Loss functions for stroke lesion node classification.

Implements weighted cross-entropy loss from Equation 13:
  L_w = -(1/N) * sum_i sum_c w_c * y_i,c * log(y_hat_i,c)

Where class weights w_c = N / (N_c * C) to handle class imbalance
between no-lesion, acute, and chronic supervoxels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss for multi-class node classification.

    Handles severe class imbalance typical in medical imaging where
    non-lesion supervoxels vastly outnumber lesion supervoxels.
    """

    def __init__(self, class_weights: torch.Tensor | None = None):
        """Initialize loss function.

        Args:
            class_weights: (C,) tensor of per-class weights.
                          If None, uniform weights are used.
        """
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    @staticmethod
    def compute_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
        """Compute inverse-frequency class weights from training labels.

        w_c = N / (N_c * C) where N = total samples, N_c = class count, C = num classes.

        Args:
            labels: (N,) integer tensor of class labels.
            num_classes: Number of classes.

        Returns:
            (C,) float tensor of class weights.
        """
        counts = torch.bincount(labels.long(), minlength=num_classes).float()
        total = labels.size(0)
        weights = total / (num_classes * counts.clamp(min=1.0))
        return weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: (N, C) raw model outputs (before softmax).
            targets: (N,) integer class labels.

        Returns:
            Scalar loss value.
        """
        return F.cross_entropy(logits, targets.long(), weight=self.class_weights)


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Complementary to cross-entropy, particularly useful for
    imbalanced classes as it directly optimizes the Dice metric.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: (N, C) raw model outputs.
            targets: (N,) integer class labels.

        Returns:
            Scalar Dice loss (1 - mean Dice coefficient).
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.long(), num_classes=num_classes).float()

        dims = (0,)  # Sum over samples
        intersection = (probs * targets_onehot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_onehot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined weighted cross-entropy and Dice loss.

    L_total = alpha * L_CE + beta * L_Dice + lambda_L2 * ||W||^2
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
    ):
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss(class_weights)
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce_loss(logits, targets) + self.dice_weight * self.dice_loss(logits, targets)
