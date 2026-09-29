"""
Report Generator
================

Generates structured experiment reports in Markdown and HTML formats.

The report assembles information from:
    - All per-task experiment JSON records
    - All per-task evaluation report JSON records
    - Flat CSV artefacts (acc-eff profiles, statistical tests)
    - The cross-task summary CSV
    - The manifest of generated visualizations

Sections (all configurable):
    1.  Project Information
    2.  Dataset Summary
    3.  Experiment Configuration
    4.  Multi-Task Framework Summary
    5.  Candidate Models
    6.  Hyperparameter Optimization Summary
    7.  Model Performance Results
    8.  Composite Adaptive Selection Results
    9.  RMSE-Only vs Composite Selection Comparison
    10. Resource Performance Summary
    11. Statistical Validation Results
    12. Explainability Summary
    13. Energy Consumption Summary
    14. Generated Figure References
    15. Experiment Limitations

Design:
    - Builds a Markdown string, then optionally converts to HTML via the
      stdlib ``html`` module (no external dependency required for Markdown→HTML;
      a simple pandoc-free converter is used).
    - Each section is guarded by the ``report.sections`` config flag.
    - Does NOT overwrite an existing report file unless configured to do so.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from src.logging.logger_factory import LoggerFactory


class ReportGenerator:
    """
    Generates structured experiment reports in Markdown and HTML.

    Args:
        config: Merged application configuration.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config

        rep_cfg = config.get("visualization", {}).get("report", {})
        project_root = config.get("project_root", ".")
        self._out_dir = os.path.join(project_root, rep_cfg.get("output_dir", "reports"))
        os.makedirs(self._out_dir, exist_ok=True)

        self._formats: List[str] = [f.lower() for f in rep_cfg.get("formats", ["markdown", "html"])]
        self._sections: Dict[str, bool] = rep_cfg.get("sections", {})
        self._overwrite: bool = bool(rep_cfg.get("overwrite_existing", False))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        all_exp: Dict[str, Dict],
        all_eval: Dict[str, Dict],
        all_acc_eff: Dict[str, pd.DataFrame],
        all_stat: Dict[str, pd.DataFrame],
        multi_task_df: Optional[pd.DataFrame],
        figure_manifest: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate the full experiment report.

        Returns:
            List of artefact metadata dicts for the generated report files.
        """
        self._logger.info("Generating experiment report...")

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        md_lines: List[str] = []

        self._add_section(md_lines, 1, "project_info",
                          self._section_project_info())
        self._add_section(md_lines, 2, "dataset_summary",
                          self._section_dataset_summary(all_exp))
        self._add_section(md_lines, 3, "experiment_config",
                          self._section_experiment_config(all_exp))
        self._add_section(md_lines, 4, "multi_task_framework",
                          self._section_multi_task_framework(all_eval, multi_task_df))
        self._add_section(md_lines, 5, "candidate_models",
                          self._section_candidate_models(all_exp))
        self._add_section(md_lines, 6, "hyperparameter_summary",
                          self._section_hyperparameter_summary(all_exp))
        self._add_section(md_lines, 7, "model_performance",
                          self._section_model_performance(all_eval))
        self._add_section(md_lines, 8, "composite_selection",
                          self._section_composite_selection(all_exp))
        self._add_section(md_lines, 9, "rmse_vs_composite_comparison",
                          self._section_rmse_vs_composite(all_eval))
        self._add_section(md_lines, 10, "resource_performance",
                          self._section_resource_performance(all_acc_eff))
        self._add_section(md_lines, 11, "statistical_validation",
                          self._section_statistical_validation(all_stat))
        self._add_section(md_lines, 12, "explainability_summary",
                          self._section_explainability_summary(all_eval))
        self._add_section(md_lines, 13, "energy_consumption",
                          self._section_energy_consumption())
        self._add_section(md_lines, 14, "figure_references",
                          self._section_figure_references(figure_manifest))
        self._add_section(md_lines, 15, "limitations",
                          self._section_limitations())

        md_content = "\n".join(md_lines)

        saved = []
        if "markdown" in self._formats:
            saved += self._write("markdown", md_content, timestamp)
        if "html" in self._formats:
            html_content = self._md_to_html(md_content)
            saved += self._write("html", html_content, timestamp)

        self._logger.info("Report generation complete. %d files written.", len(saved))
        return saved

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _section_project_info(self) -> str:
        app_cfg = self._config.get("app", {})
        project = app_cfg.get("project", {})
        return (
            f"**Project:** {project.get('name', 'Adaptive Energy Prediction')}  \n"
            f"**Version:** {project.get('version', '1.0.0')}  \n"
            f"**Execution Mode:** {project.get('execution_mode', 'local')}  \n"
            f"**Report Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  \n"
        )

    def _section_dataset_summary(self, all_exp: Dict[str, Dict]) -> str:
        lines = ["Dataset used: UCI Household Electric Power Consumption\n"]
        for task, exp in all_exp.items():
            env = exp.get("environment", {})
            lines.append(f"- **{task}**: {env.get('num_records', 'N/A')} records, "
                         f"{env.get('num_features', 'N/A')} features")
        return "\n".join(lines)

    def _section_experiment_config(self, all_exp: Dict[str, Dict]) -> str:
        if not all_exp:
            return "_No experiment records found._"
        # Use the first available experiment for global config
        exp = next(iter(all_exp.values()))
        cfg = exp.get("model_config_snapshot", {})
        train_cfg = cfg.get("training", {})
        lines = [
            f"- **Validation Strategy:** {train_cfg.get('validation_strategy', 'N/A')}",
            f"- **CV Folds:** {train_cfg.get('n_splits', 'N/A')}",
            f"- **Selection Profile:** {exp.get('selection_profile', 'N/A')}",
            f"- **Experiment Timestamp:** {exp.get('timestamp', 'N/A')}",
        ]
        return "\n".join(lines)

    def _section_multi_task_framework(
        self, all_eval: Dict[str, Dict], multi_task_df: Optional[pd.DataFrame]
    ) -> str:
        tasks = list(all_eval.keys())
        lines = [
            f"The framework evaluates {len(tasks)} independent prediction task(s):\n",
        ]
        for t in tasks:
            sel = all_eval[t].get("selected_model", "N/A")
            lines.append(f"- **{t}** → Selected: `{sel}`")

        if multi_task_df is not None and not multi_task_df.empty:
            lines.append("\n### Cross-Task Performance Summary\n")
            lines.append(multi_task_df.to_markdown(index=False))

        return "\n".join(lines)

    def _section_candidate_models(self, all_exp: Dict[str, Dict]) -> str:
        seen = set()
        lines = []
        for exp in all_exp.values():
            for m in exp.get("candidates", []):
                if m not in seen:
                    seen.add(m)
                    lines.append(f"- `{m}`")
        return "Candidate models evaluated:\n\n" + "\n".join(lines) if lines else "_No candidate data._"

    def _section_hyperparameter_summary(self, all_exp: Dict[str, Dict]) -> str:
        if not all_exp:
            return "_No experiment data._"
        exp = next(iter(all_exp.values()))
        cfg = exp.get("model_config_snapshot", {})
        tuning = cfg.get("training", {}).get("hyperparameter_tuning", {})
        if not tuning:
            return "_Hyperparameter configuration not recorded in this experiment._"
        lines = [f"- **{k}:** {v}" for k, v in tuning.items()]
        return "\n".join(lines)

    def _section_model_performance(self, all_eval: Dict[str, Dict]) -> str:
        rows = []
        for task, eva in all_eval.items():
            selected = eva.get("selected_model", "")
            for model, m in eva.get("evaluation_metrics", {}).items():
                rows.append({
                    "Task": task, "Model": model,
                    "Sel": "✓" if model == selected else "",
                    "RMSE": f"{m.get('rmse', float('nan')):.5f}",
                    "MAE": f"{m.get('mae', float('nan')):.5f}",
                    "R²": f"{m.get('r2', float('nan')):.5f}",
                })
        if not rows:
            return "_No evaluation metrics found._"
        return pd.DataFrame(rows).to_markdown(index=False)

    def _section_composite_selection(self, all_exp: Dict[str, Dict]) -> str:
        rows = []
        for task, exp in all_exp.items():
            for c in exp.get("composite_scores", []):
                rows.append({
                    "Task": task,
                    "Model": c.get("model_name"),
                    "Composite Score": f"{c.get('composite_score', float('nan')):.5f}",
                    "Rank": c.get("rank"),
                    "Selected": "✓" if c.get("is_selected") else "",
                })
        if not rows:
            return "_No composite selection data._"
        return pd.DataFrame(rows).to_markdown(index=False)

    def _section_rmse_vs_composite(self, all_eval: Dict[str, Dict]) -> str:
        rows = []
        for task, eva in all_eval.items():
            analysis = eva.get("composite_analysis", {})
            comp_winner = analysis.get("composite_winner", "N/A")
            rmse_winner = analysis.get("rmse_winner", "N/A")
            same = analysis.get("same_model_selected", None)
            diffs = analysis.get("differences", {})
            rows.append({
                "Task": task,
                "RMSE Winner": rmse_winner,
                "Composite Winner": comp_winner,
                "Same Model?": "Yes" if same else "No" if same is not None else "N/A",
                "RMSE Δ": f"{diffs.get('rmse_diff', 0):.5f}" if diffs else "—",
                "Train Time Δ (s)": f"{diffs.get('training_time_diff', 0):.3f}" if diffs else "—",
            })
        if not rows:
            return "_No composite vs RMSE comparison data._"

        note = (
            "\n\n> **Note:** A positive RMSE Δ means the composite winner is slightly less accurate "
            "than the pure RMSE winner. The composite framework trades off accuracy for other criteria "
            "(efficiency, memory). Neither method is assumed to be superior — results are reported as-is."
        )
        return pd.DataFrame(rows).to_markdown(index=False) + note

    def _section_resource_performance(self, all_acc_eff: Dict[str, pd.DataFrame]) -> str:
        rows = []
        for task, df in all_acc_eff.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                rows.append({
                    "Task": task,
                    "Model": row.get("model_name", "N/A"),
                    "Train Time (s)": f"{row.get('training_time', float('nan')):.3f}",
                    "Pred Time (s)": f"{row.get('prediction_time', float('nan')):.4f}",
                    "Memory (MB)": row.get("memory_usage_mb", "N/A"),
                    "CPU (%)": row.get("cpu_usage_pct", "N/A"),
                })
        if not rows:
            return "_No resource metrics available._"
        return pd.DataFrame(rows).to_markdown(index=False)

    def _section_statistical_validation(self, all_stat: Dict[str, pd.DataFrame]) -> str:
        rows = []
        for task, df in all_stat.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                p_val = row.get("p_value")
                rows.append({
                    "Task": task,
                    "Baseline": row.get("baseline_model", "N/A"),
                    "Candidate": row.get("candidate_model", "N/A"),
                    "Test": row.get("test_name", "N/A"),
                    "p-value": f"{float(p_val):.4e}" if p_val is not None else "N/A",
                    "α": row.get("alpha", "N/A"),
                    "Significant?": "Yes" if row.get("is_significant") else "No",
                    "Correction": row.get("correction_method", "none"),
                })
        if not rows:
            return "_No statistical test results found._"

        note = (
            "\n\n> **Caution:** Statistical significance does not imply practical significance. "
            "A statistically significant p-value only indicates the observed difference is unlikely "
            "to be due to chance given the sample size. It does not imply one model is categorically superior."
        )
        return pd.DataFrame(rows).to_markdown(index=False) + note

    def _section_explainability_summary(self, all_eval: Dict[str, Dict]) -> str:
        lines = []
        for task, eva in all_eval.items():
            lines.append(f"- **{task}**: Selected model `{eva.get('selected_model', 'N/A')}` — "
                         "see explainability figures for feature importance details.")
        if not lines:
            return "_No evaluation data for explainability section._"
        return "\n".join(lines)

    def _section_energy_consumption(self) -> str:
        return (
            "Energy consumption estimates are derived from the `EnergyEstimator` module. "
            "These values represent aggregated predictions (hourly / daily / monthly) of the "
            "four sub-metering and global power channels. They are **not** directly measured "
            "raw sensor values — they are model-derived estimates. Refer to the visualizations "
            "directory for time-series plots of these aggregates.\n"
        )

    def _section_figure_references(self, manifest: List[Dict]) -> str:
        if not manifest:
            return "_No figures generated._"
        lines = []
        for item in manifest:
            kind = item.get("kind", "figure")
            task = item.get("task", "")
            path = item.get("path", "")
            lines.append(f"- **[{kind}]** Task: `{task or 'multi_task'}` → `{os.path.basename(path)}`")
        return "\n".join(lines)

    def _section_limitations(self) -> str:
        return (
            "1. **MAPE Reliability:** For sub-metering tasks with zero-heavy distributions, "
            "MAPE is flagged as mathematically unreliable. MAE / RMSE remain valid.\n"
            "2. **Statistical Tests:** Tests are performed on paired errors from a single "
            "held-out test set. Results may differ if the test window changes.\n"
            "3. **Resource Metrics:** CPU and memory measurements depend on psutil availability. "
            "On cluster deployments these may be recorded as N/A.\n"
            "4. **SHAP Values:** SHAP explainability is not available for all model types "
            "supported by this framework. Feature importance falls back to tree-native methods.\n"
            "5. **Generalization:** Results apply to the UCI Household Electric Power Consumption "
            "dataset. Generalization to other energy datasets should be verified independently.\n"
        )

    # ------------------------------------------------------------------
    # Markdown to HTML converter (no external dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _md_to_html(md: str) -> str:
        """
        Minimal Markdown → HTML converter sufficient for research reports.
        Converts headings, bold, code, tables, and bullet lists.
        For a richer conversion, pandoc or mistletoe can be used externally.
        """
        import re
        html_lines = [
            "<!DOCTYPE html>",
            "<html lang='en'><head><meta charset='utf-8'>",
            "<title>Experiment Report</title>",
            "<style>",
            "  body { font-family: Arial, sans-serif; max-width: 1100px; margin: auto; padding: 2em; }",
            "  h1,h2,h3 { color: #2c3e50; }",
            "  table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
            "  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }",
            "  th { background: #2c3e50; color: white; }",
            "  tr:nth-child(even) { background: #f2f2f2; }",
            "  code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }",
            "  blockquote { border-left: 4px solid #e67e22; padding-left: 1em; color: #555; }",
            "  pre { background: #f4f4f4; padding: 1em; border-radius: 4px; overflow-x: auto; }",
            "</style></head><body>",
        ]

        for line in md.split("\n"):
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)

            if line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("> "):
                html_lines.append(f"<blockquote>{line[2:]}</blockquote>")
            elif line.startswith("- "):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif re.match(r"^\|.+\|$", line):
                # Markdown table row
                if re.match(r"^\|[-:| ]+\|$", line):
                    pass  # separator row
                else:
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    tag = "th" if not html_lines[-1].startswith("<tr><th") else "td"
                    html_lines.append(
                        "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
                    )
            elif line.strip() == "":
                html_lines.append("<br>")
            else:
                html_lines.append(f"<p>{line}</p>")

        html_lines.append("</body></html>")
        return "\n".join(html_lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_section(self, lines: List[str], num: int, key: str, content: str) -> None:
        if not self._sections.get(key, True):
            return
        title = key.replace("_", " ").title()
        lines.append(f"\n## {num}. {title}\n")
        lines.append(content)

    def _write(self, fmt: str, content: str, timestamp: str) -> List[Dict[str, Any]]:
        ext = "md" if fmt == "markdown" else "html"
        filename = f"experiment_report_{timestamp}.{ext}"
        path = os.path.join(self._out_dir, filename)

        if os.path.exists(path) and not self._overwrite:
            self._logger.warning("Report already exists (overwrite=False): %s", path)
            return []

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            self._logger.info("Report written: %s", path)
            return [{"path": path, "kind": f"report_{fmt}"}]
        except Exception as exc:
            self._logger.error("Failed to write report %s: %s", path, exc)
            return []
