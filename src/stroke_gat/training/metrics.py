"""Evaluation metrics for stroke lesion node classification.

Computes: Dice coefficient, AUC-ROC, sensitivity, specificity,
precision, F1-score, and balanced accuracy with per-class breakdown.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsComputer:
    """Accumulates predictions and computes classification metrics."""

    CLASS_NAMES = ["no_lesion", "acute", "chronic"]

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self._preds: list[np.ndarray] = []
        self._probs: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def reset(self) -> None:
        self._preds.clear()
        self._probs.clear()
        self._targets.clear()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate batch predictions.

        Args:
            logits: (N, C) raw model outputs.
            targets: (N,) integer labels.
        """
        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        preds = logits.detach().argmax(dim=1).cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        self._probs.append(probs)
        self._preds.append(preds)
        self._targets.append(targets_np)

    def compute(self) -> dict:
        """Compute all metrics from accumulated predictions.

        Returns:
            Dict with overall and per-class metrics.
        """
        all_preds = np.concatenate(self._preds)
        all_probs = np.concatenate(self._probs)
        all_targets = np.concatenate(self._targets)

        results = {}

        # Overall metrics
        results["balanced_accuracy"] = balanced_accuracy_score(all_targets, all_preds)
        results["f1_macro"] = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        results["f1_weighted"] = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
        results["precision_macro"] = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        results["sensitivity_macro"] = recall_score(all_targets, all_preds, average="macro", zero_division=0)

        # Per-class metrics
        for c in range(self.num_classes):
            name = self.CLASS_NAMES[c] if c < len(self.CLASS_NAMES) else f"class_{c}"
            binary_targets = (all_targets == c).astype(int)
            binary_preds = (all_preds == c).astype(int)

            results[f"{name}_precision"] = precision_score(binary_targets, binary_preds, zero_division=0)
            results[f"{name}_sensitivity"] = recall_score(binary_targets, binary_preds, zero_division=0)
            results[f"{name}_f1"] = f1_score(binary_targets, binary_preds, zero_division=0)

            # Specificity
            cm = confusion_matrix(binary_targets, binary_preds, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp = cm[0, 0], cm[0, 1]
                results[f"{name}_specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                results[f"{name}_specificity"] = 0.0

        # Dice coefficient (per-class, then average)
        dice_scores = []
        for c in range(self.num_classes):
            binary_targets = (all_targets == c).astype(float)
            binary_preds = (all_preds == c).astype(float)
            intersection = (binary_targets * binary_preds).sum()
            cardinality = binary_targets.sum() + binary_preds.sum()
            dice = (2 * intersection + 1e-6) / (cardinality + 1e-6)
            name = self.CLASS_NAMES[c] if c < len(self.CLASS_NAMES) else f"class_{c}"
            results[f"{name}_dice"] = dice
            dice_scores.append(dice)
        results["dice_mean"] = np.mean(dice_scores)

        # AUC-ROC (one-vs-rest, macro average)
        try:
            if all_probs.shape[1] == self.num_classes:
                results["auc_roc"] = roc_auc_score(
                    all_targets, all_probs, multi_class="ovr", average="macro"
                )
            else:
                results["auc_roc"] = 0.0
        except ValueError:
            results["auc_roc"] = 0.0

        # Confusion matrix
        results["confusion_matrix"] = confusion_matrix(
            all_targets, all_preds, labels=list(range(self.num_classes))
        ).tolist()

        return results

    @staticmethod
    def composite_score(metrics: dict, weights: dict | None = None) -> float:
        """Compute composite validation score for model selection.

        Default: 0.5 * Dice + 0.3 * AUC + 0.2 * Sensitivity

        Args:
            metrics: Dict of computed metrics.
            weights: Optional dict with 'dice', 'auc', 'sensitivity' weights.

        Returns:
            Scalar composite score.
        """
        if weights is None:
            weights = {"dice": 0.5, "auc": 0.3, "sensitivity": 0.2}

        return (
            weights.get("dice", 0) * metrics.get("dice_mean", 0)
            + weights.get("auc", 0) * metrics.get("auc_roc", 0)
            + weights.get("sensitivity", 0) * metrics.get("sensitivity_macro", 0)
        )
