"""
Shared Plot Utilities
=====================

Common helpers used by all specialized plot modules:
  - resolve matplotlib backend (Agg for headless environments)
  - figure factory respecting config
  - save-figure helper (PNG + optional PDF, configurable DPI)
  - colour resolution for candidate / selected models
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# Use non-interactive Agg backend so figures can be generated on headless
# servers (Hadoop clusters) without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------

def make_figure(
    config: Dict[str, Any],
    figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a new (fig, ax) pair using configured defaults."""
    fig_cfg = config.get("visualization", {}).get("figures", {})
    default_size = fig_cfg.get("figure_size", [12, 6])
    size = figsize or tuple(default_size)
    font_cfg = fig_cfg.get("font", {})

    plt.rcParams.update({
        "font.size": font_cfg.get("tick", 10),
        "axes.titlesize": font_cfg.get("title", 14),
        "axes.labelsize": font_cfg.get("axis_label", 12),
        "legend.fontsize": font_cfg.get("legend", 10),
        "xtick.labelsize": font_cfg.get("tick", 10),
        "ytick.labelsize": font_cfg.get("tick", 10),
    })

    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def make_figure_multi(
    config: Dict[str, Any],
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, Any]:
    """Create a multi-panel figure."""
    fig_cfg = config.get("visualization", {}).get("figures", {})
    default_size = fig_cfg.get("figure_size", [12, 6])
    size = figsize or (default_size[0] * ncols, default_size[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
    return fig, axes


# ---------------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    output_dir: str,
    filename_stem: str,
    config: Dict[str, Any],
) -> List[str]:
    """
    Save *fig* to all configured formats and return the list of saved paths.

    Args:
        fig: Matplotlib figure object.
        output_dir: Directory to write into.
        filename_stem: Filename without extension (e.g. ``"rmse_bar_task1"``).
        config: Merged application config.

    Returns:
        List of absolute file paths that were written.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig_cfg = config.get("visualization", {}).get("figures", {})
    dpi = int(fig_cfg.get("dpi", 150))
    formats = fig_cfg.get("formats", ["png"])

    saved = []
    for fmt in formats:
        fmt = fmt.lower().strip()
        path = os.path.join(output_dir, f"{filename_stem}.{fmt}")
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved.append(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Failed to save figure %s: %s", path, exc)
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def get_bar_colours(
    model_names: List[str],
    selected_model: str,
    config: Dict[str, Any],
) -> List[str]:
    """
    Return a colour per model, highlighting the selected model distinctly.
    """
    fig_cfg = config.get("visualization", {}).get("figures", {})
    selected_color = fig_cfg.get("selected_model_color", "#e67e22")
    palette = fig_cfg.get("candidate_palette", [
        "#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12"
    ])
    colours = []
    pal_idx = 0
    for name in model_names:
        if name == selected_model:
            colours.append(selected_color)
        else:
            colours.append(palette[pal_idx % len(palette)])
            pal_idx += 1
    return colours


def selected_legend_patch(selected_model: str, config: Dict[str, Any]) -> mpatches.Patch:
    """Return a legend patch marking the selected model colour."""
    fig_cfg = config.get("visualization", {}).get("figures", {})
    color = fig_cfg.get("selected_model_color", "#e67e22")
    return mpatches.Patch(color=color, label=f"Selected: {selected_model}")


# ---------------------------------------------------------------------------
# Sampling helper (for time-series display only, never for metric computation)
# ---------------------------------------------------------------------------

def display_sample(
    values: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Uniformly subsample *values* so the array length ≤ max_plot_points.
    This is ONLY for display — evaluation metrics use the full data.
    """
    max_pts = int(
        config.get("visualization", {})
              .get("figures", {})
              .get("max_plot_points", 5000)
    )
    n = len(values)
    if n <= max_pts:
        return values
    idx = np.round(np.linspace(0, n - 1, max_pts)).astype(int)
    return values[idx]
