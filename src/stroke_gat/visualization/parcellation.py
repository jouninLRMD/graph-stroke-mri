"""Visualization of atlas parcellation overlays on brain images."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_atlas_overlay(
    t1_data: np.ndarray,
    atlas_data: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
    alpha: float = 0.4,
    title: str = "Atlas Parcellation Overlay",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Overlay atlas regions on a T1 brain slice.

    Args:
        t1_data: 3D T1-weighted volume.
        atlas_data: 3D integer atlas labels (same shape as t1_data).
        slice_idx: Slice index. Defaults to the midpoint along axis.
        axis: Slicing axis (0=sagittal, 1=coronal, 2=axial).
        alpha: Overlay transparency.
        title: Plot title.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        Figure if ax was not provided, else None.
    """
    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2

    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx
    t1_slice = t1_data[tuple(slicing)]
    atlas_slice = atlas_data[tuple(slicing)]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    ax.imshow(t1_slice.T, cmap="gray", origin="lower")

    # Mask background (label 0)
    masked_atlas = np.ma.masked_where(atlas_slice == 0, atlas_slice)
    unique_labels = np.unique(atlas_slice[atlas_slice > 0])
    n_regions = max(len(unique_labels), 1)

    cmap = plt.cm.get_cmap("tab20", n_regions)
    ax.imshow(masked_atlas.T, cmap=cmap, alpha=alpha, origin="lower")
    ax.set_title(title)
    ax.axis("off")

    if created_fig:
        fig.tight_layout()
    return fig


def plot_atlas_slices(
    t1_data: np.ndarray,
    atlas_data: np.ndarray,
    n_slices: int = 6,
    axis: int = 2,
    alpha: float = 0.35,
) -> plt.Figure:
    """Show multiple axial slices with atlas overlay.

    Args:
        t1_data: 3D T1 volume.
        atlas_data: 3D atlas volume.
        n_slices: Number of evenly spaced slices to display.
        axis: Slicing axis.
        alpha: Overlay transparency.

    Returns:
        Matplotlib Figure.
    """
    total = t1_data.shape[axis]
    indices = np.linspace(total * 0.2, total * 0.8, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        plot_atlas_overlay(t1_data, atlas_data, slice_idx=idx, axis=axis, alpha=alpha, ax=ax)
        ax.set_title(f"Slice {idx}")

    fig.suptitle("ArterialAtlas136 Parcellation", fontsize=14)
    fig.tight_layout()
    return fig


def plot_arterial_territories(
    atlas_data: np.ndarray,
    labels_dict: dict[int, str],
    axis: int = 2,
    slice_idx: int | None = None,
) -> plt.Figure:
    """Show arterial territory regions with labeled colorbar.

    Args:
        atlas_data: 3D atlas label volume.
        labels_dict: Mapping of atlas integer ID to region name.
        axis: Slicing axis.
        slice_idx: Slice index. Defaults to midpoint.

    Returns:
        Matplotlib Figure.
    """
    if slice_idx is None:
        slice_idx = atlas_data.shape[axis] // 2

    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx
    atlas_slice = atlas_data[tuple(slicing)]

    unique_ids = sorted(np.unique(atlas_slice[atlas_slice > 0]))
    n_regions = len(unique_ids)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", max(n_regions, 1))

    masked = np.ma.masked_where(atlas_slice == 0, atlas_slice)
    im = ax.imshow(masked.T, cmap=cmap, origin="lower")
    ax.set_title(f"Arterial Territories (slice {slice_idx})")
    ax.axis("off")

    # Build legend for visible regions
    present_labels = {uid: labels_dict.get(uid, f"Region {uid}") for uid in unique_ids[:20]}
    patches = []
    import matplotlib.patches as mpatches
    for i, (uid, name) in enumerate(present_labels.items()):
        color = cmap(i / max(n_regions - 1, 1))
        patches.append(mpatches.Patch(color=color, label=f"{uid}: {name}"))

    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    return fig
