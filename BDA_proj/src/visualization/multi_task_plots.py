"""
Multi-Task Comparison Plots
============================

Generates cross-task visualizations from the
``results/evaluation/multi_task/cross_task_comparison_*.csv`` artefact
and the per-task experiment records.

Plots generated:
    1. RMSE per task (grouped bar — all candidate models side by side)
    2. MAE per task
    3. R² per task
    4. Composite score per task (selected model only)
    5. Training time per task (selected model only)
    6. Selected model identity per task (text summary figure)

The goal is to demonstrate whether task-specific model selection is
beneficial and whether the same model dominates across all tasks.
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
)


class MultiTaskPlots:
    """
    Generates cross-task comparison figures.

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
        all_exp: Dict[str, Dict[str, Any]],
        multi_task_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Generate multi-task comparison plots.

        Args:
            all_exp: Per-task experiment records (keyed by task name).
            multi_task_df: Cross-task summary DataFrame from evaluator.
        """
        self._logger.info("Generating multi-task comparison plots...")

        mt_dir = os.path.join(self._figures_dir, "multi_task")
        os.makedirs(mt_dir, exist_ok=True)

        generated = []
        generated += self._plot_metric_per_task(mt_dir, multi_task_df, "rmse", "RMSE",
                                                 lower_is_better=True,
                                                 enabled_key="multi_task_metrics_bar")
        generated += self._plot_metric_per_task(mt_dir, multi_task_df, "mae", "MAE",
                                                 lower_is_better=True,
                                                 enabled_key="multi_task_metrics_bar")
        generated += self._plot_metric_per_task(mt_dir, multi_task_df, "r2", "R²",
                                                 lower_is_better=False,
                                                 enabled_key="multi_task_metrics_bar")
        generated += self._plot_selected_model_table(mt_dir, multi_task_df)
        generated += self._plot_grouped_metrics(mt_dir, all_exp)
        return generated

    # ------------------------------------------------------------------
    # Single-metric bar per task
    # ------------------------------------------------------------------

    def _plot_metric_per_task(
        self,
        out_dir: str,
        df: pd.DataFrame,
        metric_col: str,
        metric_label: str,
        lower_is_better: bool,
        enabled_key: str,
    ) -> List[Dict[str, Any]]:
        if not self._enabled.get(enabled_key, True):
            return []
        if metric_col not in df.columns or "task_name" not in df.columns:
            return []

        sub = df[["task_name", metric_col]].dropna(subset=[metric_col])
        if sub.empty:
            return []

        tasks = sub["task_name"].tolist()
        values = sub[metric_col].to_numpy(dtype=float)

        order = np.argsort(values)
        if not lower_is_better:
            order = order[::-1]
        tasks = [tasks[i] for i in order]
        values = values[order]

        fig_cfg = self._config.get("visualization", {}).get("figures", {})
        palette = fig_cfg.get("candidate_palette", ["#3498db"] * 10)
        colours = [palette[i % len(palette)] for i in range(len(tasks))]

        fig, ax = make_figure(self._config, figsize=(10, max(4, len(tasks) * 0.8)))
        bars = ax.barh(tasks, values, color=colours, edgecolor="white", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=9)

        ax.set_title(f"{metric_label} per Task (selected model)")
        ax.set_xlabel(metric_label)
        fig.tight_layout()

        stem = f"multi_task_{metric_col}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": f"multi_task_{metric_col}", "task": "multi_task"} for p in paths]

    # ------------------------------------------------------------------
    # Selected model text-table figure
    # ------------------------------------------------------------------

    def _plot_selected_model_table(
        self,
        out_dir: str,
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        if not self._enabled.get("multi_task_selected_model", True):
            return []

        cols_wanted = ["task_name", "selected_model", "rmse", "mae", "r2",
                       "composite_vs_rmse_same"]
        present = [c for c in cols_wanted if c in df.columns]
        if "task_name" not in present or "selected_model" not in present:
            return []

        sub = df[present].copy()
        col_labels = {
            "task_name": "Task",
            "selected_model": "Selected Model",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "R²",
            "composite_vs_rmse_same": "Same as RMSE Baseline?",
        }

        cell_data = []
        for _, row in sub.iterrows():
            cell_row = []
            for c in present:
                val = row[c]
                if isinstance(val, float):
                    cell_row.append(f"{val:.4f}")
                elif isinstance(val, bool):
                    cell_row.append("Yes" if val else "No")
                else:
                    cell_row.append(str(val) if val is not None else "N/A")
            cell_data.append(cell_row)

        header = [col_labels.get(c, c) for c in present]

        fig, ax = plt.subplots(figsize=(14, max(3, len(sub) * 0.7 + 1.5)))
        ax.axis("off")
        tbl = ax.table(
            cellText=cell_data,
            colLabels=header,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.6)
        ax.set_title("Multi-Task Summary: Selected Model & Performance",
                     fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()

        stem = "multi_task_selected_model_table"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "multi_task_table", "task": "multi_task"} for p in paths]

    # ------------------------------------------------------------------
    # Grouped bar chart: RMSE for all candidates across tasks
    # ------------------------------------------------------------------

    def _plot_grouped_metrics(
        self,
        out_dir: str,
        all_exp: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each task, show per-candidate RMSE side by side in a grouped bar chart.
        This highlights whether the same model is consistently best.
        """
        if not self._enabled.get("multi_task_metrics_bar", True):
            return []
        if not all_exp:
            return []

        # Collect all unique models across tasks
        all_models: set = set()
        task_model_rmse: Dict[str, Dict[str, float]] = {}
        for task, exp in all_exp.items():
            raw = exp.get("raw_metrics", [])
            task_model_rmse[task] = {m["model_name"]: m.get("rmse", np.nan) for m in raw}
            all_models.update(task_model_rmse[task].keys())

        all_models_list = sorted(all_models)
        tasks = sorted(task_model_rmse.keys())

        x = np.arange(len(tasks))
        n_models = len(all_models_list)
        width = 0.8 / max(n_models, 1)

        fig_cfg = self._config.get("visualization", {}).get("figures", {})
        palette = fig_cfg.get("candidate_palette", ["#3498db"] * 10)

        fig, ax = plt.subplots(figsize=(max(12, len(tasks) * 2.5), 6))
        for i, model in enumerate(all_models_list):
            rmse_vals = [task_model_rmse[t].get(model, np.nan) for t in tasks]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, rmse_vals, width=width * 0.9,
                   label=model, color=palette[i % len(palette)], edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=20, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title("Candidate RMSE Across All Tasks\n"
                     "(demonstrates whether task-specific model selection is beneficial)")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

        stem = "multi_task_grouped_rmse"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "multi_task_grouped_rmse", "task": "multi_task"} for p in paths]
