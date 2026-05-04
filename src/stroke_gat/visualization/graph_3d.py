"""Interactive 3D graph visualization using Plotly."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.data import Data


# Class label display names and colors
CLASS_NAMES = ["No Lesion", "Acute", "Chronic"]
CLASS_COLORS = ["#2196F3", "#F44336", "#FF9800"]  # Blue, Red, Orange


def plot_graph_3d_interactive(
    data: Data,
    node_colors: np.ndarray | None = None,
    title: str = "Brain Graph (3D)",
    node_size: float = 3.0,
    edge_opacity: float = 0.15,
    show_edges: bool = True,
    max_edges: int = 50000,
) -> go.Figure:
    """Interactive 3D scatter plot of graph nodes with edges.

    Args:
        data: PyG Data object with x (node features), edge_index,
              y (labels), and node positions in x[:, -6:-3] or
              stored in data.pos.
        node_colors: Optional (N,) array of values for node coloring.
                     If None, uses data.y labels.
        title: Plot title.
        node_size: Scatter point size.
        edge_opacity: Edge line opacity.
        show_edges: Whether to draw edges.
        max_edges: Max edges to render (subsample for performance).

    Returns:
        Plotly Figure.
    """
    # Extract node positions
    if hasattr(data, "pos") and data.pos is not None:
        pos = data.pos.cpu().numpy() if isinstance(data.pos, torch.Tensor) else np.asarray(data.pos)
    else:
        # Assume centroids are in features at columns for x, y, z
        # Typically after intensity stats (16 dims): indices 16, 17, 18
        x_feat = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else np.asarray(data.x)
        if x_feat.shape[1] >= 19:
            pos = x_feat[:, 16:19]
        else:
            pos = x_feat[:, :3]

    labels = data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else np.asarray(data.y)

    # Node colors
    if node_colors is None:
        colors = [CLASS_COLORS[int(l) % len(CLASS_COLORS)] for l in labels]
    else:
        colors = node_colors

    # Node trace
    node_trace = go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode="markers",
        marker=dict(size=node_size, color=colors, opacity=0.8),
        text=[f"Node {i}<br>Label: {CLASS_NAMES[int(l) % len(CLASS_NAMES)]}" for i, l in enumerate(labels)],
        hoverinfo="text",
        name="Nodes",
    )

    traces = [node_trace]

    # Edge traces (subsample if too many)
    if show_edges and data.edge_index is not None:
        edge_index = data.edge_index.cpu().numpy()
        n_edges = edge_index.shape[1]

        if n_edges > max_edges:
            idx = np.random.choice(n_edges, max_edges, replace=False)
            edge_index = edge_index[:, idx]

        # Build edge lines with None separators
        xe, ye, ze = [], [], []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            xe.extend([pos[src, 0], pos[dst, 0], None])
            ye.extend([pos[src, 1], pos[dst, 1], None])
            ze.extend([pos[src, 2], pos[dst, 2], None])

        edge_trace = go.Scatter3d(
            x=xe, y=ye, z=ze,
            mode="lines",
            line=dict(color="gray", width=0.5),
            opacity=edge_opacity,
            hoverinfo="none",
            name="Edges",
        )
        traces.insert(0, edge_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        showlegend=True,
        width=900,
        height=700,
    )

    return fig


def plot_graph_2d_projection(
    data: Data,
    projection: str = "axial",
    title: str = "Brain Graph (2D projection)",
    node_size: float = 5.0,
    show_edges: bool = True,
    max_edges: int = 30000,
    ax=None,
) -> "plt.Figure | None":
    """2D scatter projection of the graph.

    Args:
        data: PyG Data object.
        projection: 'axial' (x,y), 'coronal' (x,z), or 'sagittal' (y,z).
        title: Plot title.
        node_size: Marker size.
        show_edges: Whether to draw edges.
        max_edges: Max edges to draw.
        ax: Optional matplotlib Axes.

    Returns:
        Figure if ax was not provided.
    """
    import matplotlib.pyplot as plt

    if hasattr(data, "pos") and data.pos is not None:
        pos = data.pos.cpu().numpy() if isinstance(data.pos, torch.Tensor) else np.asarray(data.pos)
    else:
        x_feat = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else np.asarray(data.x)
        pos = x_feat[:, 16:19] if x_feat.shape[1] >= 19 else x_feat[:, :3]

    labels = data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else np.asarray(data.y)

    proj_map = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}
    dim1, dim2 = proj_map.get(projection, (0, 1))

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = None

    # Draw edges
    if show_edges and data.edge_index is not None:
        edge_index = data.edge_index.cpu().numpy()
        n_edges = edge_index.shape[1]
        if n_edges > max_edges:
            idx = np.random.choice(n_edges, max_edges, replace=False)
            edge_index = edge_index[:, idx]

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            ax.plot(
                [pos[src, dim1], pos[dst, dim1]],
                [pos[src, dim2], pos[dst, dim2]],
                "gray", alpha=0.05, linewidth=0.3,
            )

    # Draw nodes by class
    for c in range(3):
        mask = labels == c
        ax.scatter(
            pos[mask, dim1], pos[mask, dim2],
            s=node_size, c=CLASS_COLORS[c], label=CLASS_NAMES[c], alpha=0.7, edgecolors="none",
        )

    axis_names = {0: "X", 1: "Y", 2: "Z"}
    ax.set_xlabel(axis_names[dim1])
    ax.set_ylabel(axis_names[dim2])
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.set_aspect("equal")

    if created_fig:
        fig.tight_layout()
    return fig
