"""Visualization of SLIC supervoxel segmentation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries


def plot_supervoxel_boundaries(
    t1_data: np.ndarray,
    supervoxel_labels: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
    boundary_color: tuple = (1.0, 0.0, 0.0),
    title: str = "Supervoxel Boundaries",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Show supervoxel boundaries overlaid on a brain slice.

    Args:
        t1_data: 3D brain volume (for background).
        supervoxel_labels: 3D integer supervoxel labels.
        slice_idx: Slice index. Defaults to midpoint.
        axis: Slicing axis.
        boundary_color: RGB tuple for boundary color.
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
    sv_slice = supervoxel_labels[tuple(slicing)]

    boundaries = find_boundaries(sv_slice, mode="thick")

    # Build RGB overlay
    t1_norm = t1_slice.astype(float)
    if t1_norm.max() > 0:
        t1_norm /= t1_norm.max()
    rgb = np.stack([t1_norm] * 3, axis=-1)
    for c in range(3):
        rgb[:, :, c][boundaries] = boundary_color[c]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    ax.imshow(rgb.transpose(1, 0, 2), origin="lower")
    ax.set_title(title)
    ax.axis("off")

    if created_fig:
        fig.tight_layout()
    return fig


def plot_supervoxel_slices(
    t1_data: np.ndarray,
    supervoxel_labels: np.ndarray,
    n_slices: int = 6,
    axis: int = 2,
) -> plt.Figure:
    """Show multiple slices with supervoxel boundaries.

    Args:
        t1_data: 3D brain volume.
        supervoxel_labels: 3D supervoxel labels.
        n_slices: Number of slices.
        axis: Slicing axis.

    Returns:
        Matplotlib Figure.
    """
    total = t1_data.shape[axis]
    indices = np.linspace(total * 0.2, total * 0.8, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        plot_supervoxel_boundaries(t1_data, supervoxel_labels, slice_idx=idx, axis=axis, ax=ax)
        ax.set_title(f"Slice {idx}")

    fig.suptitle("SLIC Supervoxel Segmentation", fontsize=14)
    fig.tight_layout()
    return fig


def plot_supervoxel_parcellated_brain(
    t1_data: np.ndarray,
    supervoxel_labels: np.ndarray,
    atlas_data: np.ndarray,
    slice_idx: int | None = None,
    axis: int = 2,
) -> plt.Figure:
    """Combined view: brain + supervoxels + atlas regions.

    Args:
        t1_data: 3D brain volume.
        supervoxel_labels: 3D supervoxel labels.
        atlas_data: 3D atlas labels.
        slice_idx: Slice index. Defaults to midpoint.
        axis: Slicing axis.

    Returns:
        Matplotlib Figure with three panels.
    """
    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: T1 only
    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx
    axes[0].imshow(t1_data[tuple(slicing)].T, cmap="gray", origin="lower")
    axes[0].set_title("T1-weighted")
    axes[0].axis("off")

    # Panel 2: Supervoxel boundaries
    plot_supervoxel_boundaries(t1_data, supervoxel_labels, slice_idx=slice_idx, axis=axis, ax=axes[1])
    axes[1].set_title("Supervoxel Boundaries")

    # Panel 3: Atlas + supervoxels
    from stroke_gat.visualization.parcellation import plot_atlas_overlay

    plot_atlas_overlay(t1_data, atlas_data, slice_idx=slice_idx, axis=axis, alpha=0.3, ax=axes[2])
    sv_slice = supervoxel_labels[tuple(slicing)]
    boundaries = find_boundaries(sv_slice, mode="thick")
    # Overlay boundaries in white
    extent = axes[2].images[0].get_extent() if axes[2].images else None
    boundary_img = np.zeros((*boundaries.T.shape, 4))
    boundary_img[boundaries.T, :] = [1, 1, 1, 0.6]
    axes[2].imshow(boundary_img, origin="lower")
    axes[2].set_title("Atlas + Supervoxels")

    fig.suptitle(f"Parcellated Brain (slice {slice_idx})", fontsize=14)
    fig.tight_layout()
    return fig


def plot_supervoxel_size_histogram(
    supervoxel_labels: np.ndarray,
    target_size: int = 256,
    title: str = "Supervoxel Size Distribution",
) -> plt.Figure:
    """Plot histogram of supervoxel sizes.

    Args:
        supervoxel_labels: 3D integer supervoxel labels.
        target_size: Target supervoxel size (shown as vertical line).
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    unique, counts = np.unique(supervoxel_labels[supervoxel_labels >= 0], return_counts=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(counts, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(target_size, color="red", linestyle="--", linewidth=2, label=f"Target: {target_size}")
    ax.axvline(np.median(counts), color="orange", linestyle="-", linewidth=2, label=f"Median: {np.median(counts):.0f}")
    ax.set_xlabel("Supervoxel Size (voxels)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    stats_text = (
        f"N={len(counts)}\n"
        f"Mean={np.mean(counts):.0f}\n"
        f"Std={np.std(counts):.0f}\n"
        f"Min={np.min(counts)}, Max={np.max(counts)}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, verticalalignment="top",
            horizontalalignment="right", fontsize=9, bbox=dict(boxstyle="round", alpha=0.8, facecolor="wheat"))

    fig.tight_layout()
    return fig


def plot_atlas_adherence(
    supervoxel_labels: np.ndarray,
    atlas_data: np.ndarray,
) -> plt.Figure:
    """Measure and plot how well supervoxels respect atlas boundaries.

    For each supervoxel, compute the fraction of its voxels that belong
    to the dominant atlas region. Higher values indicate better adherence.

    Args:
        supervoxel_labels: 3D supervoxel labels.
        atlas_data: 3D atlas labels.

    Returns:
        Matplotlib Figure.
    """
    unique_svs = np.unique(supervoxel_labels[supervoxel_labels >= 0])
    adherence_scores = []

    for sv_id in unique_svs:
        mask = supervoxel_labels == sv_id
        atlas_vals = atlas_data[mask]
        if len(atlas_vals) == 0:
            continue
        unique_atlas, atlas_counts = np.unique(atlas_vals, return_counts=True)
        dominant_fraction = atlas_counts.max() / atlas_counts.sum()
        adherence_scores.append(dominant_fraction)

    adherence_scores = np.array(adherence_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(adherence_scores, bins=50, edgecolor="black", alpha=0.7, color="mediumseagreen")
    axes[0].axvline(np.mean(adherence_scores), color="red", linestyle="--", label=f"Mean: {np.mean(adherence_scores):.3f}")
    axes[0].set_xlabel("Atlas Adherence (dominant region fraction)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Supervoxel Atlas Adherence")
    axes[0].legend()

    # Cumulative
    sorted_scores = np.sort(adherence_scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cdf, color="mediumseagreen", linewidth=2)
    axes[1].axhline(0.9, color="gray", linestyle=":", alpha=0.5)
    axes[1].axvline(0.9, color="red", linestyle="--", alpha=0.7, label=f">90%: {(adherence_scores >= 0.9).mean():.1%}")
    axes[1].set_xlabel("Atlas Adherence Threshold")
    axes[1].set_ylabel("Cumulative Fraction")
    axes[1].set_title("Cumulative Adherence Distribution")
    axes[1].legend()

    fig.suptitle("Atlas Boundary Adherence Analysis", fontsize=14)
    fig.tight_layout()
    return fig
