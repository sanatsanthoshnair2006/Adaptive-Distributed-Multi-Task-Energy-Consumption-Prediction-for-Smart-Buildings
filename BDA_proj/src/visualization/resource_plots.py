"""
Resource Performance Plots
===========================

Generates accuracy-vs-efficiency scatter plots from the
``acc_eff_profiles_*.csv`` artefacts produced by the evaluation framework.

Supported plot pairs:
    • RMSE vs Training Time
    • RMSE vs Prediction Time
    • RMSE vs Memory Usage       (skipped if data unavailable)
    • MAE vs Resource Usage      (training time)
    • Composite Score vs Composite metrics

Each plot annotates each point with the model name and visually
distinguishes the selected model.  If a resource metric column is
entirely null/NaN, that plot is skipped and logged.

Pareto-style overlay is added where it makes analytical sense
(RMSE vs training time) to help readers identify dominant models.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.logging.logger_factory import LoggerFactory
from src.visualization._plot_utils import (
    make_figure,
    save_figure,
    get_bar_colours,
)


class ResourcePlots:
    """
    Generates accuracy-efficiency scatter plots from flat profile CSVs.

    Args:
        config: Merged application configuration.
        figures_dir: Root output directory for figures.
    """

    def __init__(self, config: Dict[str, Any], figures_dir: str) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._figures_dir = figures_dir
        self._enabled = config.get("visualization", {}).get("enabled_plots", {})

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        task_name: str,
        acc_eff_df: Optional[pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """Generate resource-performance scatter plots for *task_name*."""
        if acc_eff_df is None or acc_eff_df.empty:
            self._logger.warning("[%s] No acc-eff profile data — skipping resource plots.", task_name)
            return []

        self._logger.info("[%s] Generating resource performance plots...", task_name)
        task_dir = os.path.join(self._figures_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        selected_model = self._detect_selected(acc_eff_df)
        generated = []

        # RMSE vs training time (with Pareto overlay)
        generated += self._scatter(
            task_name, task_dir, acc_eff_df, selected_model,
            x_col="training_time", y_col="rmse",
            x_label="Training Time (s)", y_label="RMSE",
            title=f"RMSE vs Training Time — {task_name}",
            enabled_key="rmse_vs_training_time",
            pareto=True,
        )
        # RMSE vs prediction time
        generated += self._scatter(
            task_name, task_dir, acc_eff_df, selected_model,
            x_col="prediction_time", y_col="rmse",
            x_label="Prediction Time (s)", y_label="RMSE",
            title=f"RMSE vs Prediction Time — {task_name}",
            enabled_key="rmse_vs_prediction_time",
        )
        # RMSE vs memory
        generated += self._scatter(
            task_name, task_dir, acc_eff_df, selected_model,
            x_col="memory_usage_mb", y_col="rmse",
            x_label="Memory Usage (MB)", y_label="RMSE",
            title=f"RMSE vs Memory Usage — {task_name}",
            enabled_key="rmse_vs_memory",
        )
        # MAE vs training time
        generated += self._scatter(
            task_name, task_dir, acc_eff_df, selected_model,
            x_col="training_time", y_col="mae",
            x_label="Training Time (s)", y_label="MAE",
            title=f"MAE vs Training Time — {task_name}",
            enabled_key="mae_vs_resources",
        )
        # Composite score vs RMSE
        generated += self._scatter(
            task_name, task_dir, acc_eff_df, selected_model,
            x_col="composite_score", y_col="rmse",
            x_label="Composite Score", y_label="RMSE",
            title=f"Composite Score vs RMSE — {task_name}",
            enabled_key="rmse_vs_training_time",  # reuse flag; always useful
        )

        return generated

    # ------------------------------------------------------------------
    # Core scatter plot
    # ------------------------------------------------------------------

    def _scatter(
        self,
        task_name: str,
        out_dir: str,
        df: pd.DataFrame,
        selected_model: str,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        title: str,
        enabled_key: str,
        pareto: bool = False,
    ) -> List[Dict[str, Any]]:
        if not self._enabled.get(enabled_key, True):
            return []

        # Validate required columns
        for col in [x_col, y_col, "model_name"]:
            if col not in df.columns:
                self._logger.debug("[%s] Column '%s' missing — skipping '%s' plot.", task_name, col, title)
                return []

        sub = df[[x_col, y_col, "model_name"]].dropna(subset=[x_col, y_col])
        if sub.empty:
            self._logger.warning("[%s] All values null for '%s' — skipping.", task_name, x_col)
            return []

        models = sub["model_name"].tolist()
        xs = sub[x_col].to_numpy(dtype=float)
        ys = sub[y_col].to_numpy(dtype=float)
        colours = get_bar_colours(models, selected_model, self._config)

        fig, ax = make_figure(self._config)

        for i, (m, x, y, c) in enumerate(zip(models, xs, ys, colours)):
            marker = "*" if m == selected_model else "o"
            size = 200 if m == selected_model else 80
            ax.scatter(x, y, color=c, s=size, marker=marker, zorder=3)
            ax.annotate(m, (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=c)

        # Pareto frontier overlay (lower RMSE AND lower time = Pareto-dominant)
        if pareto and len(xs) > 1:
            self._draw_pareto_frontier(ax, xs, ys)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Legend for selected model
        if selected_model:
            from src.visualization._plot_utils import selected_legend_patch
            ax.legend(handles=[selected_legend_patch(selected_model, self._config)], fontsize=9)

        fig.tight_layout()
        stem_x = x_col.replace("_", "")
        stem_y = y_col.replace("_", "")
        stem = f"resource_{stem_y}_vs_{stem_x}_{task_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": f"resource_{stem_y}_vs_{stem_x}", "task": task_name} for p in paths]

    # ------------------------------------------------------------------
    # Pareto frontier helper
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_pareto_frontier(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
        """
        Draw a step-line along the Pareto-dominant points (lower x and lower y).
        This is for visual guidance only — it does NOT rank the models.
        """
        points = sorted(zip(xs, ys), key=lambda p: p[0])
        pareto = []
        min_y = float("inf")
        for x, y in points:
            if y < min_y:
                pareto.append((x, y))
                min_y = y
        if len(pareto) >= 2:
            px, py = zip(*pareto)
            ax.step(px, py, where="post", color="gray", linestyle=":",
                    linewidth=1.2, label="Pareto frontier", zorder=2)
            ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_selected(df: pd.DataFrame) -> str:
        """Infer selected model from DataFrame if a 'rank' column exists."""
        if "rank" in df.columns:
            sub = df[df["rank"] == 1]
            if not sub.empty and "model_name" in sub.columns:
                return sub["model_name"].iloc[0]
        return ""
