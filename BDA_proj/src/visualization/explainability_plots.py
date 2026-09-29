"""
Explainability Plots
====================

Generates feature importance and SHAP-based visualizations using outputs
produced by the project's ExplanationManager (when present).

Graceful degradation:
    - If the explanation output directory does not exist or contains no
      artefacts, all methods log a warning and return empty lists.
    - No exception is propagated — the VisualizationManager continues.

Expected artefact layout (from ExplanationManager):
    results/explanations/{task_name}/{model_name}/
        feature_importance.json   or   feature_importance.csv
        shap_values.csv           (optional)
        local_explanations.csv    (optional)

If the layout differs in the actual implementation, the file-search
fallback (`glob`) will attempt to locate matching files automatically.

Configurable:
    visualization.explainability.top_n_features   (default: 15)
"""

from __future__ import annotations

import glob
import json
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


class ExplainabilityPlots:
    """
    Generates feature importance and optional SHAP visualizations.

    Args:
        config: Merged application configuration.
        figures_dir: Root output directory for figures.
    """

    def __init__(self, config: Dict[str, Any], figures_dir: str) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._figures_dir = figures_dir
        self._enabled = config.get("visualization", {}).get("enabled_plots", {})

        xai_cfg = config.get("visualization", {}).get("explainability", {})
        self._top_n = int(xai_cfg.get("top_n_features", 15))
        project_root = config.get("project_root", ".")
        self._xai_dir = os.path.join(
            project_root, xai_cfg.get("output_dir", "results/explanations")
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        task_name: str,
        eval_report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate explainability figures for *task_name*.

        Args:
            task_name: Prediction task identifier.
            eval_report: Evaluation report dict (for selected_model).
        """
        self._logger.info("[%s] Generating explainability plots...", task_name)

        selected_model = eval_report.get("selected_model", "")
        task_dir = os.path.join(self._figures_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        generated = []

        # Global feature importance
        fi_data = self._load_feature_importance(task_name, selected_model)
        if fi_data is not None and not fi_data.empty:
            generated += self._plot_feature_importance(task_name, task_dir, fi_data, selected_model)
        else:
            self._logger.warning(
                "[%s] Feature importance data unavailable — skipping importance plots.", task_name
            )

        # SHAP summary (optional)
        shap_data = self._load_shap_values(task_name, selected_model)
        if shap_data is not None and not shap_data.empty:
            generated += self._plot_shap_summary(task_name, task_dir, shap_data, selected_model)
        else:
            self._logger.debug("[%s] SHAP values not found — skipping SHAP plot.", task_name)

        return generated

    # ------------------------------------------------------------------
    # Feature importance bar chart
    # ------------------------------------------------------------------

    def _plot_feature_importance(
        self,
        task_name: str,
        out_dir: str,
        fi_df: pd.DataFrame,
        model_name: str,
    ) -> List[Dict[str, Any]]:
        if not self._enabled.get("feature_importance_bar", True):
            return []

        # Expect columns: feature, importance (or weight / coefficient)
        feat_col = self._detect_col(fi_df, ["feature", "feature_name", "name"])
        imp_col = self._detect_col(fi_df, ["importance", "weight", "coefficient", "value"])

        if feat_col is None or imp_col is None:
            self._logger.warning("[%s] Cannot identify feature/importance columns in %s.", task_name, list(fi_df.columns))
            return []

        fi_df = fi_df[[feat_col, imp_col]].copy()
        fi_df.columns = ["feature", "importance"]
        fi_df["importance"] = pd.to_numeric(fi_df["importance"], errors="coerce")
        fi_df = fi_df.dropna(subset=["importance"])
        fi_df = fi_df.sort_values("importance", ascending=False).head(self._top_n)

        fig, ax = make_figure(self._config, figsize=(10, max(5, len(fi_df) * 0.5)))

        fig_cfg = self._config.get("visualization", {}).get("figures", {})
        palette = fig_cfg.get("candidate_palette", ["#3498db"] * self._top_n)

        colours = [palette[i % len(palette)] for i in range(len(fi_df))]
        ax.barh(fi_df["feature"], fi_df["importance"], color=colours, edgecolor="white", height=0.6)
        ax.invert_yaxis()
        ax.set_title(
            f"Top {self._top_n} Feature Importances — {task_name}\nModel: {model_name}"
        )
        ax.set_xlabel("Importance")
        fig.tight_layout()

        stem = f"feature_importance_{task_name}_{model_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "feature_importance", "task": task_name, "model": model_name} for p in paths]

    # ------------------------------------------------------------------
    # SHAP summary plot
    # ------------------------------------------------------------------

    def _plot_shap_summary(
        self,
        task_name: str,
        out_dir: str,
        shap_df: pd.DataFrame,
        model_name: str,
    ) -> List[Dict[str, Any]]:
        if not self._enabled.get("shap_summary", True):
            return []

        # SHAP df: rows = observations, columns = features (mean abs shap per feature)
        # If the df has a 'feature' column, treat it as aggregated importance already.
        if "feature" in shap_df.columns:
            feat_col = "feature"
            imp_col = self._detect_col(shap_df, ["mean_abs_shap", "shap", "importance", "value"])
            if imp_col is None:
                return []
            top_feats = (
                shap_df[[feat_col, imp_col]]
                .rename(columns={feat_col: "feature", imp_col: "shap"})
                .sort_values("shap", ascending=False)
                .head(self._top_n)
            )
        else:
            # Compute mean |SHAP| per column
            numeric_cols = shap_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return []
            mean_abs = shap_df[numeric_cols].abs().mean()
            top_feats = (
                mean_abs.sort_values(ascending=False)
                .head(self._top_n)
                .reset_index()
            )
            top_feats.columns = ["feature", "shap"]

        fig, ax = make_figure(self._config, figsize=(10, max(5, len(top_feats) * 0.5)))
        fig_cfg = self._config.get("visualization", {}).get("figures", {})
        palette = fig_cfg.get("candidate_palette", ["#e74c3c"] * self._top_n)
        colours = [palette[i % len(palette)] for i in range(len(top_feats))]

        ax.barh(top_feats["feature"], top_feats["shap"], color=colours, edgecolor="white", height=0.6)
        ax.invert_yaxis()
        ax.set_title(
            f"SHAP Feature Importance (Mean |SHAP|) — {task_name}\nModel: {model_name}"
        )
        ax.set_xlabel("Mean |SHAP Value|")
        fig.tight_layout()

        stem = f"shap_summary_{task_name}_{model_name}"
        paths = save_figure(fig, out_dir, stem, self._config)
        return [{"path": p, "kind": "shap_summary", "task": task_name, "model": model_name} for p in paths]

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_feature_importance(self, task_name: str, model_name: str) -> Optional[pd.DataFrame]:
        """Try to load a feature importance file from the explanations directory."""
        patterns = [
            os.path.join(self._xai_dir, task_name, model_name, "feature_importance.csv"),
            os.path.join(self._xai_dir, task_name, model_name, "feature_importance.json"),
            os.path.join(self._xai_dir, task_name, "feature_importance.csv"),
            os.path.join(self._xai_dir, task_name, "*importance*.csv"),
        ]
        return self._probe_files(patterns)

    def _load_shap_values(self, task_name: str, model_name: str) -> Optional[pd.DataFrame]:
        """Try to load SHAP value file from the explanations directory."""
        patterns = [
            os.path.join(self._xai_dir, task_name, model_name, "shap_values.csv"),
            os.path.join(self._xai_dir, task_name, "shap_values.csv"),
            os.path.join(self._xai_dir, task_name, "*shap*.csv"),
        ]
        return self._probe_files(patterns)

    def _probe_files(self, patterns: List[str]) -> Optional[pd.DataFrame]:
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                f = files[0]
                try:
                    if f.endswith(".json"):
                        with open(f, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        return pd.DataFrame(data if isinstance(data, list) else [data])
                    else:
                        return pd.read_csv(f)
                except Exception as exc:
                    self._logger.warning("Failed to load explainability file %s: %s", f, exc)
        return None

    # ------------------------------------------------------------------
    # Column detection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        # Case-insensitive fallback
        lower_map = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None
