"""
Model Comparison Plots
======================

Generates per-task bar charts comparing all candidate models on:

    • RMSE
    • MAE
    • R²
    • MAPE  (when marked as reliable by the evaluator)

The selected model is visually distinguished using the configured
``selected_model_color`` from ``visualization_config.yaml``.

Data source: ``eval_report_*.json``  →  ``evaluation_metrics`` section.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from src.logging.logger_factory import LoggerFactory
from src.visualization._plot_utils import (
    make_figure,
    save_figure,
    get_bar_colours,
    selected_legend_patch,
)


class ModelComparisonPlots:
    """
    Generates candidate-model bar-chart comparisons per prediction task.

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
        eval_report: Dict[str, Any],
        exp_record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate model comparison bar charts for *task_name*."""
        if not self._enabled.get("model_comparison_bar", True):
            return []

        self._logger.info("[%s] Generating model comparison plots...", task_name)

        metrics_map: Dict[str, Dict] = eval_report.get("evaluation_metrics", {})
        selected_model: str = eval_report.get("selected_model", "")

        if not metrics_map:
            self._logger.warning("[%s] No evaluation metrics found. Skipping comparison plots.", task_name)
            return []

        task_dir = os.path.join(self._figures_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        generated = []
        generated += self._bar_chart(task_name, task_dir, metrics_map, selected_model,
                                     "rmse", "RMSE", lower_is_better=True)
        generated += self._bar_chart(task_name, task_dir, metrics_map, selected_model,
                                     "mae", "MAE", lower_is_better=True)
        generated += self._bar_chart(task_name, task_dir, metrics_map, selected_model,
                                     "r2", "R²", lower_is_better=False)

        # MAPE only when reliable for at least one model
        mape_reliable = any(
            v.get("mape_is_reliable", False) for v in metrics_map.values()
        )
        if mape_reliable:
            generated += self._bar_chart(task_name, task_dir, metrics_map, selected_model,
                                         "mape", "MAPE", lower_is_better=True)
        else:
            self._logger.info("[%s] Skipping MAPE bar chart — not reliable for this task.", task_name)

        # Combined 2×2 overview
        generated += self._combined_bar_chart(task_name, task_dir, metrics_map, selected_model)

        return generated

    # ------------------------------------------------------------------
    # Single metric bar chart
    # ------------------------------------------------------------------

    def _bar_chart(
        self,
        task_name: str,
        out_dir: str,
        metrics_map: Dict[str, Dict],
        selected_model: str,
        metric_key: str,
        metric_label: str,
        lower_is_better: bool,
    ) -> List[Dict[str, Any]]:
        """Generate a single horizontal bar chart for *metric_key*."""
        models = list(metrics_map.keys())
        values = [metrics_map[m].get(metric_key) for m in models]

        # Drop models with missing values
        pairs = [(m, v) for m, v in zip(models, values) if v is not None]
        if not pairs:
            self._logger.debug("[%s] No valid values for metric '%s'.", task_name, metric_key)
            return []

        models, values = zip(*pairs)
        models, values = list(models), list(values)
        colours = get_bar_colours(models, selected_model, self._config)

        # Sort by value
        order = np.argsort(values)
        if not lower_is_better:
            order = order[::-1]
        models = [models[i] for i in order]
        values = [values[i] for i in order]
        colours = [colours[i] for i in order]

        fig, ax = make_figure(self._config, figsize=(10, max(4, len(models) * 0.7)))
        bars = ax.barh(models, values, color=colours, edgecolor="white", height=0.6)

        # Value annotations
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=self._config.get("visualization", {}).get("figures", {}).get("font", {}).get("tick", 10)
            )

        ax.set_title(f"{metric_label} Comparison — {task_name}")
        ax.set_xlabel(metric_label)
        if selected_model in models:
            ax.legend(handles=[selected_legend_patch(selected_model, self._config)], loc="lower right")
        fig.tight_layout()

        stem = f"bar_{metric_key}_{task_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": f"bar_{metric_key}", "task": task_name, "model": selected_model} for p in paths]

    # ------------------------------------------------------------------
    # Combined 2×2 overview
    # ------------------------------------------------------------------

    def _combined_bar_chart(
        self,
        task_name: str,
        out_dir: str,
        metrics_map: Dict[str, Dict],
        selected_model: str,
    ) -> List[Dict[str, Any]]:
        """2×2 grid showing RMSE / MAE / R² / MAPE side by side."""
        models = list(metrics_map.keys())
        colours = get_bar_colours(models, selected_model, self._config)

        metric_defs = [
            ("rmse", "RMSE", True),
            ("mae", "MAE", True),
            ("r2", "R²", False),
            ("mape", "MAPE", True),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes_flat = axes.flatten()

        for ax, (key, label, lib) in zip(axes_flat, metric_defs):
            values = [metrics_map[m].get(key) for m in models]
            pairs = [(m, v, c) for m, v, c in zip(models, values, colours) if v is not None]
            if not pairs:
                ax.set_title(f"{label} — no data")
                ax.axis("off")
                continue

            m_plot, v_plot, c_plot = zip(*pairs)
            order = np.argsort(v_plot)
            if not lib:
                order = order[::-1]
            m_plot = [m_plot[i] for i in order]
            v_plot = [v_plot[i] for i in order]
            c_plot = [c_plot[i] for i in order]

            ax.barh(m_plot, v_plot, color=c_plot, edgecolor="white", height=0.6)
            ax.set_title(label)
            ax.set_xlabel(label)

        fig.suptitle(f"Candidate Model Comparison — {task_name}", fontsize=14, fontweight="bold")
        if selected_model:
            fig.legend(
                handles=[selected_legend_patch(selected_model, self._config)],
                loc="lower center", ncol=1
            )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        stem = f"bar_combined_{task_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "bar_combined", "task": task_name, "model": selected_model} for p in paths]
