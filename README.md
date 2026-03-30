# Stroke-GAT: Graph-based Multi-modal MRI Analysis with Probabilistic Attention for Stroke Lesion Detection

Implementation of the Neurocomputing 2025 paper:

> **Graph-based multi-modal MRI analysis with probabilistic attention for stroke lesion detection**
> Mercado-Diaz et al., Neurocomputing (2025)

This project detects stroke lesions in brain MRI using Probabilistic Graph Attention Networks operating on anatomically-constrained supervoxel graphs.

## Overview

The pipeline converts multi-modal brain MRI volumes into supervoxel-based graphs, then classifies each supervoxel as:
- **Class 0**: No lesion
- **Class 1**: Acute stroke
- **Class 2**: Chronic stroke

### Key Components

| Component | Description | Paper Reference |
|---|---|---|
| **Anatomically-Constrained 3D SLIC** | Custom supervoxel segmentation respecting atlas boundaries | Eq. 5-6 |
| **Graph Construction** | 4-component edge weights (intensity, atlas, spatial, co-occurrence) | Eq. 7 |
| **Graph Attention Network** | 3-layer multi-head GAT (8 heads, 128 hidden dim) | Eq. 8-11 |
| **Probabilistic Attention Attribution (PAA)** | Modulates attention with neighborhood lesion-type agreement | Eq. 12 |
| **Weighted Cross-Entropy Loss** | Inverse class frequency weighting | Eq. 13 |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/graph-stroke-mri.git
cd graph-stroke-mri

# Install in development mode
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- CUDA-capable GPU (recommended)

See `pyproject.toml` for the full dependency list.

## Project Structure

```
graph-stroke-mri/
├── configs/                    # YAML configuration files
│   └── default.yaml            # Master config with all parameters
├── notebooks/                  # Demo Jupyter notebooks (01-05)
├── scripts/                    # CLI entry points
│   ├── preprocess_dataset.py   # Register modalities, normalize
│   ├── generate_graphs.py      # SLIC + graph construction
│   ├── train.py                # Train GAT model
│   ├── evaluate.py             # Evaluate on test set
│   └── visualize.py            # Generate visualization figures
├── src/stroke_gat/             # Main package
│   ├── config.py               # Dataclass configs + YAML loading
│   ├── data/                   # Data loading, transforms, DataModule
│   ├── preprocessing/          # Registration, normalization
│   ├── slic/                   # Anatomically-constrained 3D SLIC
│   ├── graph/                  # Graph construction pipeline
│   ├── models/                 # GAT, PAA, loss functions
│   ├── training/               # Trainer, metrics, callbacks
│   ├── visualization/          # Parcellation, supervoxels, attention maps
│   └── utils/                  # Memory, checkpoints, logging
└── tests/                      # Pytest test suite
```

## Quick Start

### 1. Configure data paths

Edit `configs/default.yaml` to point to your data:

```yaml
paths:
  raw_bids: "/path/to/SOOP_BIDS"
  soop_normalized: "/path/to/SOOP_NIfTI_normalized/NIfTI"
  atlas_file: "/path/to/ArterialAtlas136.nii.gz"
  atlas_labels_file: "/path/to/ArterialAtlas136.txt"
  output_graphs: "data/graphs"
  output_models: "data/models"
```

### 2. Preprocess data

```bash
python scripts/preprocess_dataset.py --config configs/default.yaml
```

### 3. Generate supervoxel graphs

```bash
python scripts/generate_graphs.py --config configs/default.yaml --use-preprocessed
```

### 4. Train the model

```bash
python scripts/train.py --config configs/default.yaml --output-dir outputs
```

### 5. Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt --config configs/default.yaml
```

### 6. Visualize

```bash
python scripts/visualize.py --checkpoint outputs/checkpoints/best_model.pt \
    --graph data/graphs/sub-0001_supervoxel_graph.pt
```

## Demo Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Explore multi-modal MRI data, atlas, and lesion masks |
| `02_parcellation_and_slic_demo.ipynb` | Run SLIC segmentation, visualize supervoxels and atlas adherence |
| `03_graph_construction_demo.ipynb` | Build graphs from supervoxels, interactive 3D visualization |
| `04_training_demo.ipynb` | Train a model on a small subset, plot learning curves |
| `05_attention_attribution_gallery.ipynb` | Attention heatmaps, PAA attribution, stroke core vs penumbra |

## Configuration

All parameters are configurable via YAML. Key hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `slic.target_supervoxel_size` | 256 | Voxels per supervoxel (~8000 supervoxels per brain) |
| `slic.compactness` | 0.1 | Spatial vs intensity trade-off |
| `slic.atlas_penalty_lambda` | 0.5 | Atlas boundary enforcement strength |
| `model.num_layers` | 3 | GAT layers |
| `model.num_heads` | 8 | Attention heads per layer |
| `model.hidden_dim` | 128 | Hidden feature dimension |
| `training.lr` | 0.001 | Learning rate (Adam) |
| `training.lr_decay_rate` | 0.95 | Exponential LR decay per epoch |
| `training.early_stopping_patience` | 20 | Epochs without improvement |
| `training.epochs` | 200 | Maximum training epochs |

## Data

This project uses the **SOOP dataset** (Stroke Outcome Optimization Project):
- 1715 subjects (1461 stroke, 254 controls)
- Multi-modal MRI: T1, FLAIR, ADC, TRACE
- Lesion masks: acute, chronic, combined
- Atlas: ArterialAtlas136 (32 arterial territory regions)
- Available on OpenNeuro: ds004889

## Model Architecture

```
Input Features (55 dims)
    ├── Intensity stats: mean/std/min/max x 4 modalities (16)
    ├── Centroid: x, y, z (3)
    ├── Volume: normalized voxel count (1)
    ├── Atlas: one-hot region encoding (32)
    └── Connectivity: lesion-type probabilities (3)
         │
    GATConv Layer 1 (55 -> 128, 8 heads)
    LayerNorm + LeakyReLU + Dropout
         │
    GATConv Layer 2 (128 -> 128, 8 heads)
    LayerNorm + LeakyReLU + Dropout
         │
    PAA Module (modulates attention with connectivity probs)
         │
    GATConv Layer 3 (128 -> 3, 1 head)
         │
    3-class logits (no lesion, acute, chronic)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_models/ -v
pytest tests/test_slic/ -v
```

## Citation

If you use this code, please cite:

```bibtex
@article{mercadodiaz2025graph,
  title={Graph-based multi-modal MRI analysis with probabilistic attention for stroke lesion detection},
  author={Mercado-Diaz, Luis R. and others},
  journal={Neurocomputing},
  year={2025},
  publisher={Elsevier}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
