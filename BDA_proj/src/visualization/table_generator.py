"""
Table Generator
===============

Generates the seven publication-ready structured tables required for the
research paper.  All data is loaded dynamically from experiment and
evaluation artefacts — nothing is manually populated.

Tables generated:
    1. Dataset Summary
    2. Candidate Model Configuration
    3. Prediction Performance (per task × per model)
    4. Resource Performance (per task × per model)
    5. Composite Adaptive Model Selection
    6. Statistical Validation
    7. Explainability Summary

Output formats: CSV and Markdown (configurable).  LaTeX is produced as
an optional Markdown code-fence that can be copied into a paper.

All output paths are resolved from configuration.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.logging.logger_factory import LoggerFactory


class TableGenerator:
    """
    Generates publication-ready tables from experiment and evaluation artefacts.

    Args:
        config: Merged application configuration.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config

        tbl_cfg = config.get("visualization", {}).get("tables", {})
        project_root = config.get("project_root", ".")
        self._out_dir = os.path.join(project_root, tbl_cfg.get("output_dir", "reports/tables"))
        os.makedirs(self._out_dir, exist_ok=True)

        self._formats: List[str] = [f.lower() for f in tbl_cfg.get("formats", ["csv", "markdown"])]
        self._enabled: Dict[str, bool] = tbl_cfg.get("enabled", {})

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_all(
        self,
        all_exp: Dict[str, Dict],
        all_eval: Dict[str, Dict],
        all_acc_eff: Dict[str, pd.DataFrame],
        all_stat: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """Generate all enabled tables."""
        self._logger.info("Generating publication tables...")
        generated = []

        if self._enabled.get("dataset_summary", True):
            generated += self._table_dataset_summary(all_exp)

        if self._enabled.get("candidate_model_config", True):
            generated += self._table_candidate_config(all_exp)

        if self._enabled.get("prediction_performance", True):
            generated += self._table_prediction_performance(all_eval)

        if self._enabled.get("resource_performance", True):
            generated += self._table_resource_performance(all_acc_eff)

        if self._enabled.get("composite_selection", True):
            generated += self._table_composite_selection(all_exp)

        if self._enabled.get("statistical_validation", True):
            generated += self._table_statistical_validation(all_stat)

        if self._enabled.get("explainability_summary", True):
            generated += self._table_explainability_summary(all_eval, all_exp)

        self._logger.info("Tables generated: %d artefacts.", len(generated))
        return generated

    # ------------------------------------------------------------------
    # TABLE 1 — Dataset Summary
    # ------------------------------------------------------------------

    def _table_dataset_summary(self, all_exp: Dict[str, Dict]) -> List[Dict]:
        self._logger.info("Generating TABLE 1: Dataset Summary")

        rows = []
        for task_name, exp in all_exp.items():
            env = exp.get("environment", {})
            rows.append({
                "Task": task_name,
                "Records (train+test)": env.get("num_records", "N/A"),
                "Time Range Start": env.get("time_range_start", "N/A"),
                "Time Range End": env.get("time_range_end", "N/A"),
                "Input Features": env.get("num_features", "N/A"),
                "Engineered Features": env.get("num_engineered_features", "N/A"),
                "Missing Values (%)": env.get("missing_pct", "N/A"),
            })

        if not rows:
            # Provide structure even if env info is absent
            rows = [{"Note": "No environment metadata recorded in experiment files. "
                              "Run the ingestion and feature engineering pipeline first."}]

        df = pd.DataFrame(rows)
        return self._save_table(df, "table1_dataset_summary", "TABLE 1: Dataset Summary")

    # ------------------------------------------------------------------
    # TABLE 2 — Candidate Model Configuration
    # ------------------------------------------------------------------

    def _table_candidate_config(self, all_exp: Dict[str, Dict]) -> List[Dict]:
        self._logger.info("Generating TABLE 2: Candidate Model Configuration")

        rows = []
        seen = set()
        for task_name, exp in all_exp.items():
            model_cfg = exp.get("model_config_snapshot", {})
            candidates_cfg = model_cfg.get("models", {}).get("candidates", {})
            for model_name, params in candidates_cfg.items():
                key = model_name
                if key in seen:
                    continue
                seen.add(key)
                hparams = params if isinstance(params, dict) else {}
                rows.append({
                    "Model": model_name,
                    "Hyperparameters (snapshot)": str(hparams),
                    "Validation Strategy": model_cfg.get("training", {}).get("validation_strategy", "time_series_cv"),
                })

        if not rows:
            rows = [{"Note": "No model configuration snapshot found in experiment files."}]

        df = pd.DataFrame(rows)
        return self._save_table(df, "table2_candidate_config", "TABLE 2: Candidate Model Configuration")

    # ------------------------------------------------------------------
    # TABLE 3 — Prediction Performance
    # ------------------------------------------------------------------

    def _table_prediction_performance(self, all_eval: Dict[str, Dict]) -> List[Dict]:
        self._logger.info("Generating TABLE 3: Prediction Performance")

        rows = []
        for task_name, eva in all_eval.items():
            metrics_map = eva.get("evaluation_metrics", {})
            selected = eva.get("selected_model", "")
            for model_name, m in metrics_map.items():
                rows.append({
                    "Task": task_name,
                    "Model": model_name,
                    "Selected": "✓" if model_name == selected else "",
                    "RMSE": f"{m.get('rmse', float('nan')):.6f}",
                    "MAE": f"{m.get('mae', float('nan')):.6f}",
                    "R²": f"{m.get('r2', float('nan')):.6f}",
                    "MAPE": f"{m.get('mape', float('nan')):.6f}" if m.get('mape_is_reliable') else "N/A (unreliable)",
                    "Num Test Rows": m.get("num_rows", "N/A"),
                })

        if not rows:
            rows = [{"Note": "No evaluation metrics found. Run the evaluation pipeline first."}]

        df = pd.DataFrame(rows)
        return self._save_table(df, "table3_prediction_performance", "TABLE 3: Prediction Performance")

    # ------------------------------------------------------------------
    # TABLE 4 — Resource Performance
    # ------------------------------------------------------------------

    def _table_resource_performance(self, all_acc_eff: Dict[str, pd.DataFrame]) -> List[Dict]:
        self._logger.info("Generating TABLE 4: Resource Performance")

        rows = []
        for task_name, df in all_acc_eff.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                rows.append({
                    "Task": row.get("task_name", task_name),
                    "Model": row.get("model_name", "N/A"),
                    "Rank": row.get("rank", "N/A"),
                    "Training Time (s)": self._fmt(row.get("training_time")),
                    "Prediction Time (s)": self._fmt(row.get("prediction_time")),
                    "Memory Usage (MB)": self._fmt(row.get("memory_usage_mb")),
                    "CPU Usage (%)": self._fmt(row.get("cpu_usage_pct")),
                    "Composite Score": self._fmt(row.get("composite_score")),
                })

        if not rows:
            rows = [{"Note": "No resource metrics available."}]

        df_out = pd.DataFrame(rows)
        return self._save_table(df_out, "table4_resource_performance", "TABLE 4: Resource Performance")

    # ------------------------------------------------------------------
    # TABLE 5 — Composite Adaptive Model Selection
    # ------------------------------------------------------------------

    def _table_composite_selection(self, all_exp: Dict[str, Dict]) -> List[Dict]:
        self._logger.info("Generating TABLE 5: Composite Adaptive Model Selection")

        rows = []
        for task_name, exp in all_exp.items():
            composite_scores = exp.get("composite_scores", [])
            selected_model = exp.get("selected_model", "")
            for c in composite_scores:
                rows.append({
                    "Task": task_name,
                    "Model": c.get("model_name", "N/A"),
                    "Composite Score": self._fmt(c.get("composite_score")),
                    "Rank": c.get("rank", "N/A"),
                    "Selected": "✓" if c.get("is_selected") else "",
                    "Selection Profile": exp.get("selection_profile", "N/A"),
                })

        if not rows:
            rows = [{"Note": "No composite selection data found."}]

        df = pd.DataFrame(rows)
        return self._save_table(df, "table5_composite_selection", "TABLE 5: Composite Adaptive Model Selection")

    # ------------------------------------------------------------------
    # TABLE 6 — Statistical Validation
    # ------------------------------------------------------------------

    def _table_statistical_validation(self, all_stat: Dict[str, pd.DataFrame]) -> List[Dict]:
        self._logger.info("Generating TABLE 6: Statistical Validation")

        rows = []
        for task_name, df in all_stat.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                p_val = row.get("p_value")
                rows.append({
                    "Task": task_name,
                    "Baseline Model": row.get("baseline_model", "N/A"),
                    "Candidate Model": row.get("candidate_model", "N/A"),
                    "Test": row.get("test_name", "N/A"),
                    "Test Statistic": self._fmt(row.get("statistic")),
                    "p-value": f"{float(p_val):.4e}" if p_val is not None else "N/A",
                    "α (adjusted)": self._fmt(row.get("alpha")),
                    "Significant?": "Yes" if row.get("is_significant") else "No",
                    "Correction": row.get("correction_method", "none"),
                    "Sample Size": row.get("sample_size", "N/A"),
                    "Notes": row.get("notes", ""),
                })

        if not rows:
            rows = [{"Note": "No statistical test results found. Run evaluation pipeline first."}]

        df_out = pd.DataFrame(rows)
        return self._save_table(df_out, "table6_statistical_validation", "TABLE 6: Statistical Validation")

    # ------------------------------------------------------------------
    # TABLE 7 — Explainability Summary
    # ------------------------------------------------------------------

    def _table_explainability_summary(
        self,
        all_eval: Dict[str, Dict],
        all_exp: Dict[str, Dict],
    ) -> List[Dict]:
        self._logger.info("Generating TABLE 7: Explainability Summary")

        xai_cfg = self._config.get("visualization", {}).get("explainability", {})
        project_root = self._config.get("project_root", ".")
        xai_dir = os.path.join(project_root, xai_cfg.get("output_dir", "results/explanations"))
        top_n = int(xai_cfg.get("top_n_features", 15))

        rows = []
        for task_name, eva in all_eval.items():
            selected_model = eva.get("selected_model", "N/A")
            # Try to load feature importance to find top features
            fi_path = os.path.join(xai_dir, task_name, selected_model, "feature_importance.csv")
            top_features = "N/A"
            method = "N/A"
            if os.path.isfile(fi_path):
                try:
                    fi_df = pd.read_csv(fi_path)
                    feat_col = next((c for c in ["feature", "feature_name", "name"] if c in fi_df.columns), None)
                    imp_col = next((c for c in ["importance", "weight", "value"] if c in fi_df.columns), None)
                    if feat_col and imp_col:
                        fi_df = fi_df.sort_values(imp_col, ascending=False).head(top_n)
                        top_features = ", ".join(fi_df[feat_col].astype(str).tolist())
                        method = "Tree Feature Importance"
                except Exception:
                    pass

            rows.append({
                "Task": task_name,
                "Selected Model": selected_model,
                "Explanation Method": method,
                f"Top {top_n} Features": top_features,
            })

        if not rows:
            rows = [{"Note": "No explainability data found."}]

        df = pd.DataFrame(rows)
        return self._save_table(df, "table7_explainability_summary", "TABLE 7: Explainability Summary")

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _save_table(self, df: pd.DataFrame, stem: str, title: str) -> List[Dict[str, Any]]:
        """Save df in all configured formats. Returns artefact dicts."""
        saved = []
        for fmt in self._formats:
            if fmt == "csv":
                path = os.path.join(self._out_dir, f"{stem}.csv")
                try:
                    df.to_csv(path, index=False)
                    self._logger.info("Table saved: %s", path)
                    saved.append({"path": path, "kind": "table_csv", "title": title})
                except Exception as exc:
                    self._logger.error("Failed to save CSV table %s: %s", path, exc)

            elif fmt == "markdown":
                path = os.path.join(self._out_dir, f"{stem}.md")
                try:
                    md = f"## {title}\n\n"
                    md += df.to_markdown(index=False)
                    md += "\n"
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(md)
                    self._logger.info("Table saved: %s", path)
                    saved.append({"path": path, "kind": "table_markdown", "title": title})
                except Exception as exc:
                    self._logger.error("Failed to save Markdown table %s: %s", path, exc)

        return saved

    @staticmethod
    def _fmt(val) -> str:
        """Format a numeric value to 4 decimal places, or 'N/A' if None/NaN."""
        if val is None:
            return "N/A"
        try:
            f = float(val)
            import math
            if math.isnan(f):
                return "N/A"
            return f"{f:.4f}"
        except (TypeError, ValueError):
            return str(val)
