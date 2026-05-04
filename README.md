# graph-stroke-mri

Reference implementation of:

> Mercado-Diaz, L. R., Vyas, R., Vu, P., Saver, J. L., Liebeskind, D. S., Posner, J., Hsu, J., Carmichael, S. T., and Scalzo, F.
> *Graph-based multi-modal MRI analysis with probabilistic attention for stroke lesion detection.*
> Neurocomputing, 2025. https://doi.org/10.1016/j.neucom.2025.131xxx

The pipeline turns a subject's multi-modal MRI (T1, FLAIR, ADC, TRACE) into an
anatomically-constrained supervoxel graph and classifies every node as
*no lesion* (0), *acute* (1), or *chronic* (2) with a Graph Attention Network
augmented by a Probabilistic Attention Attribution (PAA) module.

If you use this code, please cite the paper (BibTeX at the bottom of this file
or use GitHub's "Cite this repository" button, which reads `CITATION.cff`).

---

## Repository layout

```
configs/                     YAML configs (default.yaml is the master)
docs/DATA_LAYOUT.md          Expected on-disk layout + voxel labeling rule
notebooks/                   Worked examples 01..05
scripts/                     CLI entry points (preprocess, generate_graphs, train, ...)
src/stroke_gat/              Package source
  config.py                  Dataclass configs + YAML loader
  data/                      DataService (BIDS + SOOP), DataModule, NIfTI I/O
  preprocessing/             Registration, intensity normalization
  slic/                      Anatomically-constrained 3D SLIC (Eq. 5-6)
  graph/                     Node features, 26-connectivity edges, builder (Eq. 7)
  models/                    GAT (Eq. 8-11), PAA (Eq. 12), losses (Eq. 13)
  training/                  Trainer, metrics, callbacks
  visualization/             Atlas overlays, supervoxel boundaries, PAA maps
  utils/                     Memory tracking, checkpointing, logging
tests/                       Pytest suite (synthetic-data fixtures, no real MRI needed)
```

## Installation

```bash
git clone https://github.com/jouninLRMD/graph-stroke-mri.git
cd graph-stroke-mri
pip install -e ".[dev]"
```

Tested on Python 3.10 / 3.11 with PyTorch 2.x and a CUDA GPU. The full dependency
list lives in `pyproject.toml`; `torch-scatter` may need a wheel that matches
your CUDA version.

## Reproducing the paper from a clean machine

### 1. Get the data

The dataset (Stroke Outcome Optimization Project, ~1715 subjects) is on
OpenNeuro as `ds004889`:

> https://openneuro.org/datasets/ds004889

DataLad is the most reliable way to fetch it:

```bash
datalad install https://github.com/OpenNeuroDatasets/ds004889.git
cd ds004889
datalad get .
```

Or, if you prefer S3:

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004889 ./ds004889
```

The expected on-disk layout (and per-voxel labeling rule) is documented in
[`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md). The `validate_dataset.py` script
in step 3 will check it for you.

The atlas (`ArterialAtlas136.nii.gz` and its `.txt` label file) ships with the
SOOP normalized release. Notebook 01 can also use that release as a FLAIR-only
quickstart if you want to poke around before downloading the full BIDS data.

### 2. Configure paths

```bash
cp configs/default.yaml configs/local.yaml
```

Edit `configs/local.yaml`:

```yaml
paths:
  raw_bids:           "/abs/path/to/ds004889"
  soop_normalized:    "/abs/path/to/SOOP_NIfTI_normalized/NIfTI"
  atlas_file:         "/abs/path/to/ArterialAtlas136.nii.gz"
  atlas_labels_file:  "/abs/path/to/ArterialAtlas136.txt"
```

The values committed in `configs/default.yaml` are my own local paths — copy
to `configs/local.yaml` before editing so you don't accidentally commit yours.

### 3. Validate the setup

```bash
python scripts/validate_dataset.py --config configs/local.yaml
```

The validator walks every assumption (raw BIDS layout, atlas readability,
acute/chronic mask presence, SOOP files) and prints `[OK]` / `[WARN]` /
`[ERR]` per check. Fix the `[ERR]` items before continuing.

### 4. Run the pipeline

```bash
python scripts/preprocess_dataset.py --config configs/local.yaml
python scripts/generate_graphs.py    --config configs/local.yaml --use-preprocessed
python scripts/train.py              --config configs/local.yaml --output-dir outputs
python scripts/evaluate.py           --config configs/local.yaml \
    --checkpoint outputs/checkpoints/best_model.pt
```

Each script accepts `--help`. `train.py` writes the best model to
`outputs/checkpoints/best_model.pt` (selected by composite score
`0.5·Dice + 0.3·AUC + 0.2·Sensitivity`) and dumps a JSON history alongside.

To produce the figures from the paper for a single subject:

```bash
python scripts/visualize.py --config configs/local.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --graph data/graphs/sub-1000_supervoxel_graph.pt
```

## Notebooks

Five worked examples in `notebooks/`, intended to be run in order:

1. `01_data_exploration.ipynb` — load a subject, view modalities and lesion masks.
2. `02_parcellation_and_slic_demo.ipynb` — atlas parcellation and supervoxel segmentation.
3. `03_graph_construction_demo.ipynb` — build the graph for one subject and inspect it.
4. `04_training_demo.ipynb` — short training run with reduced epochs.
5. `05_attention_attribution_gallery.ipynb` — attention heatmaps and PAA attribution maps.

Notebook 01 falls back to the SOOP-normalized FLAIR-only data if the raw BIDS
hasn't been downloaded yet, so you can browse the atlas and a sample subject
before committing to the full download. Notebooks 02-05 require the full BIDS
data (multi-modal + acute/chronic masks).

## Hyperparameters

Defaults match what was used for the paper. Key knobs (full list in
`configs/default.yaml`):

| Group       | Parameter                       | Default | Where it matters |
|-------------|---------------------------------|---------|------------------|
| `slic`      | `target_supervoxel_size`        | 256     | yields ~8000 supervoxels per brain |
| `slic`      | `compactness`                   | 0.1     | spatial vs intensity weight in Eq. 5 |
| `slic`      | `atlas_penalty_lambda`          | 0.5     | `λ` in Eq. 5; pushes supervoxels to respect atlas borders |
| `model`     | `num_layers`, `num_heads`, `hidden_dim` | 3, 8, 128 | GAT capacity |
| `training`  | `lr`, `lr_decay_rate`           | 1e-3, 0.95 | Adam + ExponentialLR |
| `training`  | `early_stopping_patience`       | 20      | epochs without composite-score improvement |
| `training`  | `epochs`                        | 200     | hard cap |

## Tests

```bash
pytest tests/ -v
```

The test suite uses synthetic 32³ volumes and a small mock atlas, so the
fixtures fit in memory and tests don't need real MRI data.

## Citation

```bibtex
@article{mercadodiaz2025graph,
  title   = {Graph-based multi-modal {MRI} analysis with probabilistic attention for stroke lesion detection},
  author  = {Mercado-Diaz, Luis R. and Vyas, Reshma and Vu, Phuc and
             Saver, Jeffrey L. and Liebeskind, David S. and Posner, Jonathan and
             Hsu, Joseph and Carmichael, S. Thomas and Scalzo, Fabien},
  journal = {Neurocomputing},
  year    = {2025},
  publisher = {Elsevier}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

Luis R. Mercado-Diaz — `luis.mercado_diaz@uconn.edu`
Department of Biomedical Engineering, University of Connecticut.
