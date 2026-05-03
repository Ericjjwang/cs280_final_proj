"""
Visualization utilities. All functions return matplotlib Figure objects.
Use save_figure() to persist and close.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
from PIL import Image


def plot_sampson_heatmap(
    image: Image.Image,
    kpts: np.ndarray,
    sampson_dist: np.ndarray,
    mask: Optional[np.ndarray] = None,
    vmax: float = 20.0,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> matplotlib.figure.Figure:
    """
    Scatter keypoints on image, coloured green→red by Sampson distance.

    Args:
        image:       PIL Image background.
        kpts:        (N, 2) float pixel coords (x, y).
        sampson_dist:(N,)  float Sampson distances in pixels.
        mask:        (H, W) uint8 optional; object boundary drawn as white contour.
        vmax:        colour scale upper bound (px).
        title:       axes title.
        ax:          existing Axes; new Figure created if None.

    Returns:
        matplotlib Figure.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    ax.imshow(image)

    if len(kpts) > 0:
        sc = ax.scatter(
            kpts[:, 0], kpts[:, 1],
            c=np.clip(sampson_dist, 0, vmax),
            cmap="RdYlGn_r", s=12, alpha=0.85,
            vmin=0, vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, label="Sampson dist (px)", fraction=0.046)

    if mask is not None:
        ax.contour(mask, levels=[0.5], colors="white", linewidths=1.0)

    median_s = float(np.median(sampson_dist)) if len(sampson_dist) > 0 else float("nan")
    ax.set_title(f"{title}\nmedian={median_s:.1f} px  N={len(kpts)}", fontsize=10)
    ax.axis("off")
    return fig


def plot_region_bar(
    pair_stats: list[dict],
    ax: Optional[plt.Axes] = None,
) -> matplotlib.figure.Figure:
    """
    Grouped bar chart: median Sampson distance per view, split by region.

    Args:
        pair_stats: list of dicts, each with keys:
            "view_idx", "azimuth",
            "median_sampson_object", "median_sampson_background",
            "n_object_matches", "n_background_matches".
            Views with n_matches == 0 are skipped.

    Returns:
        matplotlib Figure.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    valid = [
        p for p in pair_stats
        if (p.get("n_object_matches", 0) or 0) + (p.get("n_background_matches", 0) or 0) > 0
    ]
    if not valid:
        ax.text(0.5, 0.5, "No valid pairs", ha="center", transform=ax.transAxes)
        return fig

    xs     = np.arange(len(valid))
    labels = [f"V{p['view_idx']}\naz={p['azimuth']}°" for p in valid]
    obj_sd = [p.get("median_sampson_object")     or 0 for p in valid]
    bg_sd  = [p.get("median_sampson_background") or 0 for p in valid]

    w = 0.35
    ax.bar(xs - w/2, obj_sd, w, label="Object",     color="#e07b54")
    ax.bar(xs + w/2, bg_sd,  w, label="Background", color="#5ba4cf")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median Sampson dist (px)")
    ax.set_title("Sampson distance: object vs background regions")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    return fig


def plot_4panel(
    cond_img: Image.Image,
    view_img: Image.Image,
    kpts_b: np.ndarray,
    sampson_dist: np.ndarray,
    pair_stats_list: Optional[list[dict]] = None,
    mask_cond: Optional[np.ndarray] = None,
    mask_view: Optional[np.ndarray] = None,
    view_label: str = "",
    vmax: float = 20.0,
) -> matplotlib.figure.Figure:
    """
    Four-panel comparison figure.

    Layout:
        [ cond (+ mask boundary) | view image           ]
        [ Sampson heatmap        | region bar chart     ]
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Top-left: condition image
    axes[0, 0].imshow(cond_img)
    if mask_cond is not None:
        axes[0, 0].contour(mask_cond, levels=[0.5], colors="lime", linewidths=1.5)
    axes[0, 0].set_title("Condition image", fontsize=11)
    axes[0, 0].axis("off")

    # Top-right: novel view
    axes[0, 1].imshow(view_img)
    if mask_view is not None:
        axes[0, 1].contour(mask_view, levels=[0.5], colors="lime", linewidths=1.5)
    axes[0, 1].set_title(f"Novel view  {view_label}", fontsize=11)
    axes[0, 1].axis("off")

    # Bottom-left: Sampson heatmap on view
    plot_sampson_heatmap(
        view_img, kpts_b, sampson_dist,
        mask=mask_view, vmax=vmax,
        title=f"Sampson heatmap  {view_label}",
        ax=axes[1, 0],
    )

    # Bottom-right: region bar chart
    if pair_stats_list:
        plot_region_bar(pair_stats_list, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "No stats", ha="center", transform=axes[1, 1].transAxes)
        axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def save_figure(fig: matplotlib.figure.Figure, path: str, dpi: int = 150) -> None:
    """Save figure to `path` and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
