"""
Prediction Plots
================

Generates per-task, per-model prediction quality visualizations:

    1. Actual vs Predicted time-series  (chronological, display-sampled)
    2. Actual vs Predicted scatter       (with 45° reference line)
    3. Residual vs Timestamp             (or vs Predicted value)
    4. Residual distribution             (histogram + KDE)

All data is loaded from the evaluation report JSON produced by
``src.evaluation.evaluator.Evaluator``.  Prediction DataFrames are NOT
re-loaded here; the evaluator already persisted the metrics.

For the time-series and scatter plots we need the actual prediction arrays,
which are read from the flat ``acc_eff_profiles`` artefact when available.
If the raw predictions CSV is present in the evaluation output, it is used;
otherwise the plots are skipped and a warning is logged.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.logging.logger_factory import LoggerFactory
from src.visualization._plot_utils import (
    make_figure,
    save_figure,
    display_sample,
)


class PredictionPlots:
    """
    Generates prediction-quality visualizations from saved evaluation artefacts.

    Args:
        config: Merged application configuration.
        figures_dir: Root output directory for figures.
    """

    def __init__(self, config: Dict[str, Any], figures_dir: str) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._figures_dir = figures_dir
        self._enabled = (
            config.get("visualization", {})
                  .get("enabled_plots", {})
        )

    # ------------------------------------------------------------------
    # Public entry point (called by VisualizationManager)
    # ------------------------------------------------------------------

    def generate(
        self,
        task_name: str,
        eval_report: Dict[str, Any],
        exp_record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate all prediction plots for *task_name*.

        Args:
            task_name: Prediction task identifier.
            eval_report: Dict loaded from ``eval_report_*.json``.
            exp_record: Dict loaded from the experiment JSON.

        Returns:
            List of artefact metadata dicts.
        """
        self._logger.info("[%s] Generating prediction plots...", task_name)
        generated = []

        selected_model = eval_report.get("selected_model", "")
        metrics_per_model = eval_report.get("evaluation_metrics", {})
        acc_eff_profiles = eval_report.get("accuracy_efficiency_profiles", [])

        # We need prediction arrays — try to find them via the project's
        # saved prediction CSVs (task stores these via save_predictions).
        # If unavailable we produce only the metric-summary plots.
        pred_df = self._try_load_predictions_csv(task_name, selected_model, exp_record)

        task_dir = os.path.join(self._figures_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        if pred_df is not None and not pred_df.empty:
            generated += self._plot_timeseries(task_name, task_dir, pred_df, selected_model)
            generated += self._plot_scatter(task_name, task_dir, pred_df, selected_model)
            generated += self._plot_residual_vs_time(task_name, task_dir, pred_df, selected_model)
            generated += self._plot_residual_distribution(task_name, task_dir, pred_df, selected_model)
        else:
            self._logger.warning(
                "[%s] Prediction arrays unavailable — skipping time-series/scatter/residual plots.", task_name
            )

        return generated

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def _plot_timeseries(
        self, task_name: str, out_dir: str, df: pd.DataFrame, model: str
    ) -> List[Dict[str, Any]]:
        """Actual vs Predicted time-series (chronologically ordered, display-sampled)."""
        if not self._enabled.get("actual_vs_predicted_timeseries", True):
            return []

        label_col = "label"
        pred_col = "prediction"
        if label_col not in df.columns or pred_col not in df.columns:
            self._logger.warning("[%s] Missing label/prediction columns for time-series plot.", task_name)
            return []

        actual = df[label_col].to_numpy()
        predicted = df[pred_col].to_numpy()
        n = len(actual)
        xs = np.arange(n)

        # Display-only sampling
        xs_s = display_sample(xs, self._config)
        actual_s = display_sample(actual, self._config)
        predicted_s = display_sample(predicted, self._config)

        fig, ax = make_figure(self._config, figsize=(14, 5))
        ax.plot(xs_s, actual_s, label="Actual", color="#2c3e50", linewidth=1.0, alpha=0.85)
        ax.plot(xs_s, predicted_s, label=f"Predicted ({model})", color="#e74c3c",
                linewidth=1.0, linestyle="--", alpha=0.85)
        ax.set_title(f"Actual vs Predicted — {task_name}")
        ax.set_xlabel("Time Step (chronological)")
        ax.set_ylabel(task_name)
        ax.legend()
        if n > len(xs_s):
            ax.set_xlabel(
                f"Time Step (chronological) — display-sampled to {len(xs_s)}/{n} points"
            )
        fig.tight_layout()

        stem = f"timeseries_{task_name}_{model}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "timeseries", "task": task_name, "model": model} for p in paths]

    def _plot_scatter(
        self, task_name: str, out_dir: str, df: pd.DataFrame, model: str
    ) -> List[Dict[str, Any]]:
        """Actual vs Predicted scatter with 45° reference line."""
        if not self._enabled.get("actual_vs_predicted_scatter", True):
            return []

        label_col, pred_col = "label", "prediction"
        if label_col not in df.columns or pred_col not in df.columns:
            return []

        actual = df[label_col].to_numpy()
        predicted = df[pred_col].to_numpy()
        actual_s = display_sample(actual, self._config)
        predicted_s = display_sample(predicted, self._config)

        fig, ax = make_figure(self._config)
        ax.scatter(actual_s, predicted_s, alpha=0.3, s=8, color="#3498db", label="Predictions")
        lims = [
            min(actual_s.min(), predicted_s.min()),
            max(actual_s.max(), predicted_s.max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1.0, label="Perfect prediction")
        ax.set_title(f"Actual vs Predicted Scatter — {task_name}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        fig.tight_layout()

        stem = f"scatter_{task_name}_{model}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "scatter", "task": task_name, "model": model} for p in paths]

    def _plot_residual_vs_time(
        self, task_name: str, out_dir: str, df: pd.DataFrame, model: str
    ) -> List[Dict[str, Any]]:
        """Residual (actual − predicted) vs chronological time step."""
        if not self._enabled.get("residual_vs_time", True):
            return []

        label_col, pred_col = "label", "prediction"
        if label_col not in df.columns or pred_col not in df.columns:
            return []

        residuals = df[label_col].to_numpy() - df[pred_col].to_numpy()
        n = len(residuals)
        xs = np.arange(n)
        xs_s = display_sample(xs, self._config)
        res_s = display_sample(residuals, self._config)

        fig, ax = make_figure(self._config, figsize=(14, 4))
        ax.plot(xs_s, res_s, color="#9b59b6", linewidth=0.8, alpha=0.8, label="Residual")
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"Residuals vs Time — {task_name} / {model}")
        ax.set_xlabel("Time Step (chronological)")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.legend()
        fig.tight_layout()

        stem = f"residual_vs_time_{task_name}_{model}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "residual_vs_time", "task": task_name, "model": model} for p in paths]

    def _plot_residual_distribution(
        self, task_name: str, out_dir: str, df: pd.DataFrame, model: str
    ) -> List[Dict[str, Any]]:
        """Residual distribution histogram with KDE overlay."""
        if not self._enabled.get("residual_distribution", True):
            return []

        label_col, pred_col = "label", "prediction"
        if label_col not in df.columns or pred_col not in df.columns:
            return []

        residuals = df[label_col].to_numpy() - df[pred_col].to_numpy()

        fig, ax = make_figure(self._config)
        ax.hist(residuals, bins=60, density=True, alpha=0.6, color="#1abc9c", label="Histogram")

        # KDE
        try:
            kde = gaussian_kde(residuals)
            xs = np.linspace(residuals.min(), residuals.max(), 300)
            ax.plot(xs, kde(xs), color="#e74c3c", linewidth=1.5, label="KDE")
        except Exception:
            pass

        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"Residual Distribution — {task_name} / {model}")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()

        stem = f"residual_dist_{task_name}_{model}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "residual_distribution", "task": task_name, "model": model} for p in paths]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_load_predictions_csv(
        self,
        task_name: str,
        model_name: str,
        exp_record: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Try to locate and load a saved predictions CSV for the selected model.

        The BaseTask.save_predictions typically writes to:
            results/predictions/{task_name}/{model_name}/{version}/predictions.csv
        or similar.  We probe several candidate paths.
        """
        import glob as _glob

        project_root = self._config.get("project_root", ".")
        experiment_id = exp_record.get("experiment_id", "")
        version = f"exp_{experiment_id[:8]}" if experiment_id else "*"

        candidate_patterns = [
            os.path.join(project_root, "results", "predictions", task_name, model_name, version, "*.csv"),
            os.path.join(project_root, "results", "predictions", task_name, model_name, "*.csv"),
            os.path.join(project_root, "results", "predictions", task_name, "*.csv"),
        ]

        for pattern in candidate_patterns:
            files = _glob.glob(pattern)
            if files:
                try:
                    df = pd.read_csv(files[0])
                    self._logger.info(
                        "[%s/%s] Loaded prediction CSV: %s", task_name, model_name, files[0]
                    )
                    return df
                except Exception as exc:
                    self._logger.warning(
                        "[%s/%s] Failed to read prediction CSV %s: %s", task_name, model_name, files[0], exc
                    )

        self._logger.debug(
            "[%s/%s] No prediction CSV found; prediction plots will be skipped.", task_name, model_name
        )
        return pd.DataFrame()
