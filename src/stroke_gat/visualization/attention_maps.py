"""Attention heatmap visualizations projected onto brain slices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize


def project_attention_to_volume(
    attention_weights: torch.Tensor | np.ndarray,
    edge_index: torch.Tensor | np.ndarray,
    supervoxel_labels: np.ndarray,
    aggregation: str = "mean",
) -> np.ndarray:
    """Project per-edge attention weights back to voxel space.

    For each node, aggregate incoming attention weights, then map
    the scalar value to all voxels belonging to that supervoxel.

    Args:
        attention_weights: (E,) or (E, H) attention coefficients.
        edge_index: (2, E) edge connectivity.
        supervoxel_labels: 3D integer supervoxel label volume.
        aggregation: 'mean' or 'max' aggregation of incoming attention.

    Returns:
        3D float array (same shape as supervoxel_labels) with per-voxel
        attention values.
    """
    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.detach().cpu().numpy()
    else:
        attn = np.asarray(attention_weights)

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # Average multi-head attention
    if attn.ndim == 2:
        attn = attn.mean(axis=-1)

    # Aggregate per-node incoming attention
    unique_labels = np.unique(supervoxel_labels[supervoxel_labels >= 0])
    num_nodes = len(unique_labels)
    dst = edge_index[1]

    node_attn = np.zeros(num_nodes, dtype=np.float64)
    node_count = np.zeros(num_nodes, dtype=np.float64)

    for e in range(len(attn)):
        d = dst[e]
        if d < num_nodes:
            if aggregation == "max":
                node_attn[d] = max(node_attn[d], attn[e])
            else:
                node_attn[d] += attn[e]
            node_count[d] += 1

    if aggregation == "mean":
        node_attn = np.divide(node_attn, node_count, where=node_count > 0, out=node_attn)

    # Map to voxel space
    label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}
    volume = np.zeros(supervoxel_labels.shape, dtype=np.float32)
    for label, idx in label_to_idx.items():
        volume[supervoxel_labels == label] = node_attn[idx]

    return volume


def plot_attention_heatmap(
    attention_volume: np.ndarray,
    t1_data: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
    cmap: str = "hot",
    alpha: float = 0.5,
    title: str = "Attention Heatmap",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Overlay attention heatmap on a brain slice.

    Args:
        attention_volume: 3D float attention values (from project_attention_to_volume).
        t1_data: 3D brain volume for background.
        slice_idx: Slice index.
        axis: Slicing axis.
        cmap: Colormap for attention.
        alpha: Overlay transparency.
        title: Plot title.
        ax: Optional Axes.

    Returns:
        Figure if ax was not provided.
    """
    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2

    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx
    t1_slice = t1_data[tuple(slicing)]
    attn_slice = attention_volume[tuple(slicing)]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    ax.imshow(t1_slice.T, cmap="gray", origin="lower")

    masked_attn = np.ma.masked_where(attn_slice <= 0, attn_slice)
    im = ax.imshow(masked_attn.T, cmap=cmap, alpha=alpha, origin="lower")

    if created_fig:
        plt.colorbar(im, ax=ax, label="Attention", shrink=0.8)

    ax.set_title(title)
    ax.axis("off")

    if created_fig:
        fig.tight_layout()
    return fig


def plot_attention_by_layer(
    attention_weights_list: list[torch.Tensor | np.ndarray],
    edge_index: torch.Tensor | np.ndarray,
    supervoxel_labels: np.ndarray,
    t1_data: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
) -> plt.Figure:
    """Show attention heatmaps from each GAT layer side by side.

    Args:
        attention_weights_list: List of attention tensors, one per layer.
        edge_index: (2, E) edge connectivity.
        supervoxel_labels: 3D supervoxel labels.
        t1_data: 3D brain volume.
        slice_idx: Slice index.
        axis: Slicing axis.

    Returns:
        Matplotlib Figure.
    """
    n_layers = len(attention_weights_list)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for i, attn in enumerate(attention_weights_list):
        vol = project_attention_to_volume(attn, edge_index, supervoxel_labels)
        plot_attention_heatmap(vol, t1_data, slice_idx=slice_idx, axis=axis,
                              title=f"Layer {i + 1}", ax=axes[i])

    fig.suptitle("Attention Weights by GAT Layer", fontsize=14)
    fig.tight_layout()
    return fig


def plot_attention_by_region(
    attention_volume: np.ndarray,
    atlas_data: np.ndarray,
    labels_dict: dict[int, str],
    top_n: int = 15,
) -> plt.Figure:
    """Bar chart of mean attention per atlas region.

    Args:
        attention_volume: 3D voxel-level attention.
        atlas_data: 3D atlas labels.
        labels_dict: Atlas ID to name mapping.
        top_n: Show top-N regions.

    Returns:
        Matplotlib Figure.
    """
    unique_regions = np.unique(atlas_data[atlas_data > 0])
    region_attn = {}

    for rid in unique_regions:
        mask = atlas_data == rid
        values = attention_volume[mask]
        if len(values) > 0:
            name = labels_dict.get(int(rid), f"Region {rid}")
            region_attn[name] = float(np.mean(values))

    # Sort by attention
    sorted_regions = sorted(region_attn.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [r[0] for r in sorted_regions]
    values = [r[1] for r in sorted_regions]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.barh(range(len(names)), values, color="coral", edgecolor="black")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Attention")
    ax.set_title(f"Top {top_n} Regions by Attention")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig
