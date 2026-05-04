"""Shared fixtures for stroke_gat test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from stroke_gat.config import GraphConfig, ModelConfig, SLICConfig


# ---------------------------------------------------------------------------
# Volume fixtures (32x32x32 -- small enough for fast tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_volume() -> np.ndarray:
    """32x32x32 random float volume in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((32, 32, 32), dtype=np.float32)


@pytest.fixture
def synthetic_atlas() -> np.ndarray:
    """32x32x32 integer atlas with 4 regions (8x32x32 blocks labeled 1-4)."""
    atlas = np.zeros((32, 32, 32), dtype=np.int32)
    atlas[0:8, :, :] = 1
    atlas[8:16, :, :] = 2
    atlas[16:24, :, :] = 3
    atlas[24:32, :, :] = 4
    return atlas


@pytest.fixture
def synthetic_brain_mask() -> np.ndarray:
    """32x32x32 boolean mask -- sphere of radius 14 centred at (16,16,16)."""
    coords = np.indices((32, 32, 32)).astype(np.float32)
    dist = np.sqrt(
        (coords[0] - 16) ** 2 + (coords[1] - 16) ** 2 + (coords[2] - 16) ** 2
    )
    return dist <= 14.0


@pytest.fixture
def synthetic_modalities() -> dict[str, np.ndarray]:
    """Dict with T1, FLAIR, ADC, TRACE as 32x32x32 arrays (different seeds)."""
    modalities: dict[str, np.ndarray] = {}
    for i, name in enumerate(["T1", "FLAIR", "ADC", "TRACE"]):
        rng = np.random.default_rng(seed=42 + i)
        modalities[name] = rng.random((32, 32, 32)).astype(np.float32)
    return modalities


@pytest.fixture
def synthetic_masks() -> dict[str, np.ndarray]:
    """Dict with 'Acute' (sphere r=3 at 20,20,20) and 'Chronic' (sphere r=3 at 10,10,10)."""
    coords = np.indices((32, 32, 32)).astype(np.float32)

    acute_dist = np.sqrt(
        (coords[0] - 20) ** 2 + (coords[1] - 20) ** 2 + (coords[2] - 20) ** 2
    )
    chronic_dist = np.sqrt(
        (coords[0] - 10) ** 2 + (coords[1] - 10) ** 2 + (coords[2] - 10) ** 2
    )

    acute = (acute_dist <= 3.0).astype(np.uint8)
    chronic = (chronic_dist <= 3.0).astype(np.uint8)
    general = np.clip(acute + chronic, 0, 1).astype(np.uint8)

    return {"Acute": acute, "Chronic": chronic, "General": general}


# ---------------------------------------------------------------------------
# Graph / torch fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_graph_data() -> Data:
    """A torch_geometric Data object: x=(100,55), edge_index=(2,500), y=(100,), pos=(100,3)."""
    rng = np.random.default_rng(99)
    x = torch.tensor(rng.standard_normal((100, 55)), dtype=torch.float32)
    # Random edges -- values in [0, 100)
    src = torch.randint(0, 100, (500,))
    dst = torch.randint(0, 100, (500,))
    edge_index = torch.stack([src, dst], dim=0)
    y = torch.randint(0, 3, (100,))
    pos = torch.tensor(rng.standard_normal((100, 3)), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y, pos=pos)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_config() -> ModelConfig:
    """A ModelConfig instance with defaults."""
    return ModelConfig()


@pytest.fixture
def slic_config() -> SLICConfig:
    """A SLICConfig instance with defaults."""
    return SLICConfig()


@pytest.fixture
def graph_config() -> GraphConfig:
    """A GraphConfig instance with defaults."""
    return GraphConfig()
