"""
Composite Adaptive Selection Plots
====================================

This module is central to the research paper contribution. It generates:

    1. Composite Score bar chart          — final ranking per task
    2. Normalised-metrics stacked bar     — contribution of each dimension
    3. RMSE-only vs Composite comparison  — objective side-by-side
    4. Raw metric comparison table plot   — all candidates × all raw metrics

All data is read from the experiment JSON produced by
``CompositeAdaptiveSelector`` / ``ExperimentManager``.

Design:
    - Never claims composite is "better" — presents actual numbers.
    - Weights are read from the experiment JSON snapshot, never hardcoded.
    - Selected model is visually highlighted consistently.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.logging.logger_factory import LoggerFactory
from src.visualization._plot_utils import (
    make_figure,
    make_figure_multi,
    save_figure,
    get_bar_colours,
    selected_legend_patch,
)


class SelectionPlots:
    """
    Visualizes the Composite Adaptive Model Selection process and results.

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
        exp_record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate all composite selection plots for *task_name*."""
        self._logger.info("[%s] Generating composite selection plots...", task_name)

        if not exp_record:
            self._logger.warning("[%s] Empty experiment record — skipping selection plots.", task_name)
            return []

        composite_scores = exp_record.get("composite_scores", [])
        raw_metrics = exp_record.get("raw_metrics", [])
        norm_metrics = exp_record.get("normalised_metrics", [])
        selected_model = exp_record.get("selected_model", "")
        profile_weights = exp_record.get("profile_weights", {})

        if not composite_scores:
            self._logger.warning("[%s] No composite scores in experiment record.", task_name)
            return []

        task_dir = os.path.join(self._figures_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        generated = []
        generated += self._plot_composite_ranking(task_name, task_dir, composite_scores, selected_model)
        generated += self._plot_score_breakdown(task_name, task_dir, norm_metrics, composite_scores,
                                                 profile_weights, selected_model)
        generated += self._plot_rmse_vs_composite(task_name, task_dir, raw_metrics,
                                                   composite_scores, selected_model)
        generated += self._plot_raw_metrics_heatmap(task_name, task_dir, raw_metrics, selected_model)
        return generated

    # ------------------------------------------------------------------
    # Composite ranking bar
    # ------------------------------------------------------------------

    def _plot_composite_ranking(
        self,
        task_name: str,
        out_dir: str,
        composite_scores: List[Dict],
        selected_model: str,
    ) -> List[Dict]:
        if not self._enabled.get("composite_ranking", True):
            return []

        models = [c["model_name"] for c in composite_scores]
        scores = [c["composite_score"] for c in composite_scores]
        ranks = [c["rank"] for c in composite_scores]

        # Sort by score (lower is better)
        order = np.argsort(scores)
        models = [models[i] for i in order]
        scores = [scores[i] for i in order]
        colours = get_bar_colours(models, selected_model, self._config)

        fig, ax = make_figure(self._config, figsize=(10, max(4, len(models) * 0.8)))
        bars = ax.barh(models, scores, color=colours, edgecolor="white", height=0.6)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                    f"{s:.4f}", va="center", ha="left", fontsize=9)

        ax.set_title(f"Composite Performance Score Ranking — {task_name}\n(lower = better)")
        ax.set_xlabel("Composite Score")
        ax.legend(handles=[selected_legend_patch(selected_model, self._config)])
        fig.tight_layout()

        stem = f"composite_ranking_{task_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "composite_ranking", "task": task_name} for p in paths]

    # ------------------------------------------------------------------
    # Normalised score breakdown (stacked bar — contribution per dimension)
    # ------------------------------------------------------------------

    def _plot_score_breakdown(
        self,
        task_name: str,
        out_dir: str,
        norm_metrics: List[Dict],
        composite_scores: List[Dict],
        profile_weights: Dict[str, float],
        selected_model: str,
    ) -> List[Dict]:
        if not self._enabled.get("composite_score_breakdown", True):
            return []
        if not norm_metrics or not profile_weights:
            return []

        # Dimension key → display label
        dims = {
            "normalised_rmse": "RMSE",
            "normalised_mae": "MAE",
            "normalised_r2_penalty": "R² Penalty",
            "normalised_training_time": "Train Time",
            "normalised_prediction_time": "Pred Time",
            "normalised_memory_mb": "Memory",
            "normalised_cpu_pct": "CPU",
        }
        # Only keep dimensions with non-zero weight
        active_dims = {k: v for k, v in dims.items()
                       if profile_weights.get(k.replace("normalised_", ""), 0) > 0}
        if not active_dims:
            active_dims = dims

        models = [m["model_name"] for m in norm_metrics]
        # For each model, compute the weighted contribution per dimension
        contributions = []
        weight_sum_map = {k: profile_weights.get(k.replace("normalised_", ""), 0)
                         for k in active_dims}

        for nm in norm_metrics:
            row = {}
            for k, lbl in active_dims.items():
                val = nm.get(k, 0.0) or 0.0
                w = weight_sum_map[k]
                row[lbl] = val * w
            contributions.append(row)

        # Sort by composite score
        score_order = {c["model_name"]: c["composite_score"] for c in composite_scores}
        order = np.argsort([score_order.get(m, 0) for m in models])
        models = [models[i] for i in order]
        contributions = [contributions[i] for i in order]

        labels = list(active_dims.values())
        fig_cfg = self._config.get("visualization", {}).get("figures", {})
        palette = fig_cfg.get("candidate_palette", ["#3498db", "#2ecc71", "#9b59b6",
                                                      "#e74c3c", "#1abc9c", "#f39c12"])

        fig, ax = plt.subplots(figsize=(12, max(5, len(models) * 0.9)))
        left = np.zeros(len(models))
        for j, lbl in enumerate(labels):
            vals = np.array([c.get(lbl, 0.0) for c in contributions])
            ax.barh(models, vals, left=left,
                    color=palette[j % len(palette)], label=lbl, edgecolor="white", height=0.6)
            left += vals

        ax.set_title(
            f"Composite Score Breakdown by Dimension — {task_name}\n"
            "Each bar segment = normalised metric × configured weight"
        )
        ax.set_xlabel("Weighted Contribution to Composite Score")
        ax.legend(loc="lower right", fontsize=9)
        # Mark selected model
        if selected_model in models:
            idx = models.index(selected_model)
            ax.get_yticklabels()[idx].set_fontweight("bold")
            ax.get_yticklabels()[idx].set_color(
                fig_cfg.get("selected_model_color", "#e67e22")
            )
        fig.tight_layout()

        stem = f"score_breakdown_{task_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "score_breakdown", "task": task_name} for p in paths]

    # ------------------------------------------------------------------
    # RMSE-only vs Composite comparison
    # ------------------------------------------------------------------

    def _plot_rmse_vs_composite(
        self,
        task_name: str,
        out_dir: str,
        raw_metrics: List[Dict],
        composite_scores: List[Dict],
        selected_model: str,
    ) -> List[Dict]:
        if not self._enabled.get("composite_vs_rmse_comparison", True):
            return []
        if not raw_metrics:
            return []

        # RMSE winner = model with lowest raw RMSE
        rmse_winner = min(raw_metrics, key=lambda x: x.get("rmse", float("inf")))["model_name"]
        # Composite winner = selected_model

        models = [m["model_name"] for m in raw_metrics]
        rmse_vals = [m.get("rmse", 0) for m in raw_metrics]
        score_map = {c["model_name"]: c["composite_score"] for c in composite_scores}
        comp_vals = [score_map.get(m, 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(models) * 0.8)))

        colours_rmse = get_bar_colours(models, rmse_winner, self._config)
        colours_comp = get_bar_colours(models, selected_model, self._config)

        # Left: RMSE ranking
        order_rmse = np.argsort(rmse_vals)
        ax1.barh([models[i] for i in order_rmse], [rmse_vals[i] for i in order_rmse],
                 color=[colours_rmse[i] for i in order_rmse], edgecolor="white", height=0.6)
        ax1.set_title(f"RMSE-Only Ranking\n(Winner: {rmse_winner})")
        ax1.set_xlabel("RMSE (lower = better)")
        ax1.legend(handles=[selected_legend_patch(rmse_winner, self._config)], fontsize=9)

        # Right: Composite ranking
        order_comp = np.argsort(comp_vals)
        ax2.barh([models[i] for i in order_comp], [comp_vals[i] for i in order_comp],
                 color=[colours_comp[i] for i in order_comp], edgecolor="white", height=0.6)
        ax2.set_title(f"Composite Score Ranking\n(Winner: {selected_model})")
        ax2.set_xlabel("Composite Score (lower = better)")
        ax2.legend(handles=[selected_legend_patch(selected_model, self._config)], fontsize=9)

        same = rmse_winner == selected_model
        fig.suptitle(
            f"RMSE-Only vs Composite Adaptive Selection — {task_name}\n"
            f"Same model selected: {'YES' if same else 'NO  ← Different models'}",
            fontsize=13, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        stem = f"rmse_vs_composite_{task_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "rmse_vs_composite", "task": task_name} for p in paths]

    # ------------------------------------------------------------------
    # Raw metrics overview heatmap-style table
    # ------------------------------------------------------------------

    def _plot_raw_metrics_heatmap(
        self,
        task_name: str,
        out_dir: str,
        raw_metrics: List[Dict],
        selected_model: str,
    ) -> List[Dict]:
        """
        Renders a colour-mapped table of raw metrics across candidates,
        suitable for direct inclusion in a research paper as a figure.
        """
        if not raw_metrics:
            return []

        metric_cols = ["rmse", "mae", "r2", "training_time", "prediction_time"]
        col_labels = ["RMSE", "MAE", "R²", "Train Time (s)", "Pred Time (s)"]

        models = [m["model_name"] for m in raw_metrics]
        data = np.array([
            [m.get(c, np.nan) for c in metric_cols]
            for m in raw_metrics
        ], dtype=float)

        fig, ax = plt.subplots(figsize=(12, max(3, len(models) * 0.6 + 1)))
        ax.axis("off")

        table = ax.table(
            cellText=[[f"{v:.4f}" if not np.isnan(v) else "N/A" for v in row] for row in data],
            rowLabels=models,
            colLabels=col_labels,
            cellLoc="center",
            rowLoc="right",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Highlight selected model row
        if selected_model in models:
            sel_idx = models.index(selected_model)
            fig_cfg = self._config.get("visualization", {}).get("figures", {})
            sel_col = fig_cfg.get("selected_model_color", "#e67e22")
            for col_idx in range(len(col_labels)):
                table[sel_idx + 1, col_idx].set_facecolor(sel_col)
                table[sel_idx + 1, col_idx].set_text_props(color="white", fontweight="bold")

        ax.set_title(f"Raw Metric Summary — {task_name}", fontsize=13, fontweight="bold", pad=10)
        fig.tight_layout()

        stem = f"raw_metrics_table_{task_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "raw_metrics_table", "task": task_name} for p in paths]
