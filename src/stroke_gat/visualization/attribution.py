"""PAA attribution visualization: stroke core, penumbra, and interpretable maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from stroke_gat.visualization.attention_maps import project_attention_to_volume


def plot_paa_attribution_map(
    paa_scores: torch.Tensor | np.ndarray,
    edge_index: torch.Tensor | np.ndarray,
    supervoxel_labels: np.ndarray,
    t1_data: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
    title: str = "PAA Attribution Map",
) -> plt.Figure:
    """Overlay PAA attribution scores on a brain slice.

    High attribution indicates edges where both the model attention
    and neighborhood lesion-type agreement are strong.

    Args:
        paa_scores: (E,) per-edge attribution scores from PAA.
        edge_index: (2, E) edge connectivity.
        supervoxel_labels: 3D supervoxel labels.
        t1_data: 3D brain volume.
        slice_idx: Slice index.
        axis: Slicing axis.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    paa_volume = project_attention_to_volume(
        paa_scores, edge_index, supervoxel_labels, aggregation="mean"
    )

    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2

    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx
    t1_slice = t1_data[tuple(slicing)]
    paa_slice = paa_volume[tuple(slicing)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(t1_slice.T, cmap="gray", origin="lower")

    # Custom colormap: transparent -> yellow -> red
    colors = [(0, 0, 0, 0), (1, 1, 0, 0.5), (1, 0, 0, 0.9)]
    paa_cmap = LinearSegmentedColormap.from_list("paa", colors, N=256)

    masked = np.ma.masked_where(paa_slice <= 0, paa_slice)
    im = ax.imshow(masked.T, cmap=paa_cmap, origin="lower")
    plt.colorbar(im, ax=ax, label="Attribution Score", shrink=0.8)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_stroke_penumbra_detection(
    paa_scores: torch.Tensor | np.ndarray,
    edge_index: torch.Tensor | np.ndarray,
    supervoxel_labels: np.ndarray,
    lesion_mask: np.ndarray,
    t1_data: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
    core_threshold: float = 0.7,
    penumbra_threshold: float = 0.3,
) -> plt.Figure:
    """Visualize stroke core vs penumbra using PAA attribution.

    High PAA scores within the lesion mask indicate the stroke core.
    Moderate PAA scores in surrounding tissue suggest penumbra.

    Args:
        paa_scores: (E,) per-edge attribution scores.
        edge_index: (2, E) edge connectivity.
        supervoxel_labels: 3D supervoxel labels.
        lesion_mask: 3D binary lesion mask.
        t1_data: 3D brain volume.
        slice_idx: Slice index.
        axis: Slicing axis.
        core_threshold: Attribution threshold for stroke core.
        penumbra_threshold: Attribution threshold for penumbra.

    Returns:
        Matplotlib Figure with 3 panels.
    """
    paa_volume = project_attention_to_volume(
        paa_scores, edge_index, supervoxel_labels, aggregation="mean"
    )

    # Normalize to [0, 1]
    paa_max = paa_volume.max()
    if paa_max > 0:
        paa_norm = paa_volume / paa_max
    else:
        paa_norm = paa_volume

    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2

    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx

    t1_slice = t1_data[tuple(slicing)]
    paa_slice = paa_norm[tuple(slicing)]
    lesion_slice = lesion_mask[tuple(slicing)].astype(bool)

    # Classify regions
    core_mask = (paa_slice >= core_threshold) & lesion_slice
    penumbra_mask = (paa_slice >= penumbra_threshold) & (paa_slice < core_threshold)
    # Include both lesion boundary and high-attribution non-lesion areas
    penumbra_mask = penumbra_mask & ~core_mask

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: T1 + lesion mask
    axes[0].imshow(t1_slice.T, cmap="gray", origin="lower")
    lesion_overlay = np.zeros((*lesion_slice.T.shape, 4))
    lesion_overlay[lesion_slice.T, :] = [1, 0, 0, 0.3]
    axes[0].imshow(lesion_overlay, origin="lower")
    axes[0].set_title("Ground Truth Lesion")
    axes[0].axis("off")

    # Panel 2: PAA attribution
    axes[1].imshow(t1_slice.T, cmap="gray", origin="lower")
    masked_paa = np.ma.masked_where(paa_slice <= 0, paa_slice)
    im = axes[1].imshow(masked_paa.T, cmap="hot", alpha=0.6, origin="lower")
    plt.colorbar(im, ax=axes[1], label="PAA Score", shrink=0.8)
    axes[1].set_title("PAA Attribution")
    axes[1].axis("off")

    # Panel 3: Core + penumbra
    axes[2].imshow(t1_slice.T, cmap="gray", origin="lower")
    detection = np.zeros((*t1_slice.T.shape, 4))
    detection[core_mask.T, :] = [1, 0, 0, 0.7]       # Core = red
    detection[penumbra_mask.T, :] = [1, 1, 0, 0.5]    # Penumbra = yellow
    axes[2].imshow(detection, origin="lower")
    axes[2].set_title("Core (red) + Penumbra (yellow)")
    axes[2].axis("off")

    # Add legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=(1, 0, 0, 0.7), label="Stroke Core"),
        mpatches.Patch(color=(1, 1, 0, 0.5), label="Penumbra"),
    ]
    axes[2].legend(handles=legend_patches, loc="lower right")

    fig.suptitle("Stroke Core vs Penumbra Detection via PAA", fontsize=14)
    fig.tight_layout()
    return fig


def plot_graph_attention_attribution(
    data,
    paa_scores: torch.Tensor | np.ndarray,
    title: str = "Graph with PAA Attribution",
    node_size: float = 4.0,
):
    """3D graph with nodes colored by aggregated PAA attribution.

    Args:
        data: PyG Data object.
        paa_scores: (E,) per-edge attribution scores.
        title: Plot title.
        node_size: Point size.

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    if isinstance(paa_scores, torch.Tensor):
        paa_np = paa_scores.detach().cpu().numpy()
    else:
        paa_np = np.asarray(paa_scores)

    if paa_np.ndim == 2:
        paa_np = paa_np.mean(axis=-1)

    edge_index = data.edge_index.cpu().numpy()
    dst = edge_index[1]
    num_nodes = data.x.size(0)

    # Aggregate per-node
    node_scores = np.zeros(num_nodes)
    node_count = np.zeros(num_nodes)
    for e in range(len(paa_np)):
        d = dst[e]
        if d < num_nodes:
            node_scores[d] += paa_np[e]
            node_count[d] += 1
    node_scores = np.divide(node_scores, node_count, where=node_count > 0, out=node_scores)

    # Positions
    if hasattr(data, "pos") and data.pos is not None:
        pos = data.pos.cpu().numpy()
    else:
        x_feat = data.x.cpu().numpy()
        pos = x_feat[:, 16:19] if x_feat.shape[1] >= 19 else x_feat[:, :3]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_scores,
                colorscale="YlOrRd",
                colorbar=dict(title="PAA Score"),
                opacity=0.8,
            ),
            text=[f"Node {i}<br>PAA: {node_scores[i]:.4f}" for i in range(num_nodes)],
            hoverinfo="text",
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=900,
        height=700,
    )
    return fig


def plot_paa_weight_distribution(
    paa_scores: torch.Tensor | np.ndarray,
    edge_index: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> plt.Figure:
    """Distribution of PAA scores for different edge types (lesion-lesion,
    lesion-normal, normal-normal).

    Args:
        paa_scores: (E,) attribution scores.
        edge_index: (2, E) edge connectivity.
        labels: (N,) node labels.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(paa_scores, torch.Tensor):
        scores = paa_scores.detach().cpu().numpy()
    else:
        scores = np.asarray(paa_scores)
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if scores.ndim == 2:
        scores = scores.mean(axis=-1)

    src_labels = labels[edge_index[0]]
    dst_labels = labels[edge_index[1]]

    # Categorize edges
    both_lesion = (src_labels > 0) & (dst_labels > 0)
    mixed = (src_labels > 0) ^ (dst_labels > 0)
    both_normal = (src_labels == 0) & (dst_labels == 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for mask, name, color in [
        (both_lesion, "Lesion-Lesion", "#F44336"),
        (mixed, "Lesion-Normal", "#FF9800"),
        (both_normal, "Normal-Normal", "#2196F3"),
    ]:
        if mask.any():
            ax.hist(scores[mask], bins=50, alpha=0.6, color=color, label=name, density=True)

    ax.set_xlabel("PAA Attribution Score")
    ax.set_ylabel("Density")
    ax.set_title("PAA Score Distribution by Edge Type")
    ax.legend()

    fig.tight_layout()
    return fig
