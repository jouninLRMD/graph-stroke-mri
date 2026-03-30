"""Configuration dataclasses and YAML loading for the stroke-GAT pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class PathsConfig:
    """Paths to data directories and atlas files."""

    raw_bids: str = ""
    soop_normalized: str = ""
    output_preprocessed: str = "data/preprocessed"
    output_graphs: str = "data/graphs"
    output_models: str = "data/models"
    output_visualizations: str = "data/visualizations"
    atlas_file: str = ""
    atlas_labels_file: str = ""


@dataclass
class SLICConfig:
    """Anatomically-constrained 3D SLIC supervoxel parameters."""

    target_supervoxel_size: int = 256
    compactness: float = 0.1
    atlas_penalty_lambda: float = 0.5
    max_iterations: int = 10
    convergence_threshold: float = 1e-4
    min_cluster_size: int = 20
    modality_weights: dict[str, float] = field(
        default_factory=lambda: {"T1": 0.25, "FLAIR": 0.25, "ADC": 0.25, "TRACE": 0.25}
    )


@dataclass
class GraphConfig:
    """Graph construction parameters."""

    connectivity: int = 26
    edge_weight_alpha: float = 0.25  # intensity similarity
    edge_weight_beta: float = 0.25  # atlas similarity
    edge_weight_gamma: float = 0.25  # spatial proximity
    edge_weight_delta: float = 0.25  # lesion co-occurrence
    num_lesion_types: int = 3
    spatial_sigma: float = 10.0
    batch_size_nodes: int = 512
    batch_size_edges: int = 4


@dataclass
class ModelConfig:
    """GAT model architecture parameters."""

    num_layers: int = 3
    num_heads: int = 8
    hidden_dim: int = 128
    dropout: float = 0.1
    negative_slope: float = 0.2
    num_classes: int = 3
    use_paa: bool = True
    use_edge_weights: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 200
    lr: float = 0.001
    lr_decay_rate: float = 0.95
    weight_decay: float = 5e-4
    batch_size: int = 8
    num_workers: int = 4
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    composite_weights: dict[str, float] = field(
        default_factory=lambda: {"dice": 0.5, "auc": 0.3, "sensitivity": 0.2}
    )


@dataclass
class PipelineConfig:
    """Top-level configuration composing all sub-configs."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    slic: SLICConfig = field(default_factory=SLICConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: Optional[str | Path] = None) -> PipelineConfig:
    """Load pipeline configuration from YAML file with defaults.

    Args:
        config_path: Path to a YAML config file. If None, uses defaults.

    Returns:
        Merged PipelineConfig with file overrides applied on top of defaults.
    """
    schema = OmegaConf.structured(PipelineConfig)

    if config_path is not None:
        file_conf = OmegaConf.load(config_path)
        merged = OmegaConf.merge(schema, file_conf)
    else:
        merged = schema

    return OmegaConf.to_object(merged)
