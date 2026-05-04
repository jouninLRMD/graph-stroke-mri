"""StrokeDataModule: manages train/val/test splits with stratification."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stroke_gat.config import PathsConfig, TrainingConfig
from stroke_gat.data.soop_dataset import SOOPGraphDataset

logger = logging.getLogger(__name__)


class StrokeDataModule:
    """Manages train/val/test splits, feature scaling, and DataLoaders.

    Stratifies by lesion_level to ensure balanced representation.
    Feature scaling is fitted only on training data.
    """

    def __init__(self, training_config: TrainingConfig, paths_config: PathsConfig):
        self.config = training_config
        self.paths = paths_config
        self.train_dataset: SOOPGraphDataset | None = None
        self.val_dataset: SOOPGraphDataset | None = None
        self.test_dataset: SOOPGraphDataset | None = None
        self.node_scaler = StandardScaler()
        self._class_weights: torch.Tensor | None = None

    def setup(self, graph_dir: str | Path | None = None) -> None:
        """Discover graph files, split, and prepare datasets.

        Args:
            graph_dir: Directory containing *_supervoxel_graph.pt files.
                       Defaults to paths_config.output_graphs.
        """
        graph_dir = Path(graph_dir or self.paths.output_graphs)
        graph_files = sorted(graph_dir.glob("*_supervoxel_graph.pt"))

        if not graph_files:
            raise FileNotFoundError(f"No graph files found in {graph_dir}")

        logger.info("Found %d graph files", len(graph_files))

        # Extract subject IDs and load metadata for stratification
        subjects = [f.stem.replace("_supervoxel_graph", "") for f in graph_files]
        metadata_path = graph_dir.parent / "subjects_metadata.csv"

        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)
            strat_labels = self._get_stratification_labels(subjects, metadata_df)
        else:
            logger.warning("No metadata file found, using random splits")
            strat_labels = None

        file_strs = [str(f) for f in graph_files]

        # Split: train+val / test
        train_val_files, test_files = train_test_split(
            file_strs,
            test_size=self.config.test_split,
            random_state=self.config.seed,
            stratify=strat_labels,
        )

        # Derive stratification for train_val subset
        if strat_labels is not None:
            train_val_indices = [file_strs.index(f) for f in train_val_files]
            train_val_strat = [strat_labels[i] for i in train_val_indices]
        else:
            train_val_strat = None

        # Split: train / val
        adjusted_val = self.config.val_split / (1 - self.config.test_split)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=adjusted_val,
            random_state=self.config.seed,
            stratify=train_val_strat,
        )

        self.train_dataset = SOOPGraphDataset(train_files)
        self.val_dataset = SOOPGraphDataset(val_files)
        self.test_dataset = SOOPGraphDataset(test_files)

        logger.info(
            "Splits: train=%d, val=%d, test=%d",
            len(train_files),
            len(val_files),
            len(test_files),
        )

        # Fit scaler on training data
        self._fit_scaler()

    def _get_stratification_labels(
        self, subjects: list[str], metadata_df: pd.DataFrame
    ) -> list[str]:
        """Build stratification labels from metadata."""
        meta_map = dict(zip(metadata_df["subject"].astype(str), metadata_df["lesion_level"]))
        return [meta_map.get(s, "unknown") for s in subjects]

    def _fit_scaler(self) -> None:
        """Fit StandardScaler on training node features."""
        logger.info("Fitting feature scaler on training data...")
        features = []
        for data in tqdm(self.train_dataset, desc="Fitting scaler"):
            features.append(data.x.numpy())
        all_features = np.vstack(features)
        self.node_scaler.fit(all_features)
        logger.info("Scaler fitted on %d feature vectors, dim=%d", all_features.shape[0], all_features.shape[1])

    def _scale_transform(self, data):
        """Apply fitted scaler to node features."""
        data.x = torch.tensor(self.node_scaler.transform(data.x.numpy()), dtype=torch.float32)
        return data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from training labels.

        Returns:
            Float tensor of shape (num_classes,).
        """
        if self._class_weights is not None:
            return self._class_weights

        all_labels = []
        for data in self.train_dataset:
            if data.y.dim() == 1:
                all_labels.append(data.y)
            else:
                all_labels.append(data.y.argmax(dim=1))

        labels = torch.cat(all_labels)
        num_classes = int(labels.max().item()) + 1
        counts = torch.bincount(labels, minlength=num_classes).float()
        total = labels.size(0)
        weights = total / (num_classes * counts.clamp(min=1))

        self._class_weights = weights
        logger.info("Class weights: %s", weights.tolist())
        return weights

    def get_input_dim(self) -> int:
        """Return node feature dimension from the first training sample."""
        sample = self.train_dataset.get(0)
        return sample.x.size(1)
