"""
Visualization Manager
=====================

Central orchestrator for the Visualization and Research Reporting module.

Responsibilities:
    - Resolve all result paths from configuration.
    - Load experiment JSON, evaluation JSON, and flat CSV artifacts.
    - Delegate figure generation to specialized plot modules.
    - Delegate table generation to TableGenerator.
    - Delegate report generation to ReportGenerator.
    - Collect and return a metadata manifest of every generated artefact.
    - Fail gracefully: a single missing file or failed plot never aborts the run.

Design Decisions:
    - This class owns NO matplotlib logic; all plt.* calls live in the
      specialized plotters.
    - result_store: a plain dict passed by reference to plotters so they can
      access loaded data without repeated I/O.
    - All output paths are resolved from config; nothing is hardcoded.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.logging.logger_factory import LoggerFactory
from src.visualization.prediction_plots import PredictionPlots
from src.visualization.model_comparison_plots import ModelComparisonPlots
from src.visualization.selection_plots import SelectionPlots
from src.visualization.resource_plots import ResourcePlots
from src.visualization.multi_task_plots import MultiTaskPlots
from src.visualization.explainability_plots import ExplainabilityPlots
from src.visualization.table_generator import TableGenerator
from src.visualization.report_generator import ReportGenerator


class VisualizationManager:
    """
    Coordinates all visualization and report generation.

    Args:
        config (Dict[str, Any]): Merged application configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._project_root = config.get("project_root", ".")

        viz_cfg = config.get("visualization", {})
        fig_cfg = viz_cfg.get("figures", {})
        self._figures_dir = os.path.join(
            self._project_root, fig_cfg.get("output_dir", "visualizations")
        )
        os.makedirs(self._figures_dir, exist_ok=True)

        exp_cfg = config.get("model", {}).get("experiment", {})
        self._experiments_dir = os.path.join(
            self._project_root,
            exp_cfg.get("results_dir", "results/experiments")
        )

        eval_cfg = config.get("evaluation", {}).get("reporting", {})
        self._evaluation_dir = os.path.join(
            self._project_root,
            eval_cfg.get("output_dir", "results/evaluation")
        )

        # Specialized plotters
        self._pred_plots = PredictionPlots(config, self._figures_dir)
        self._comp_plots = ModelComparisonPlots(config, self._figures_dir)
        self._sel_plots = SelectionPlots(config, self._figures_dir)
        self._res_plots = ResourcePlots(config, self._figures_dir)
        self._mt_plots = MultiTaskPlots(config, self._figures_dir)
        self._xai_plots = ExplainabilityPlots(config, self._figures_dir)
        self._tables = TableGenerator(config)
        self._reporter = ReportGenerator(config)

        # Manifest: tracks every generated artefact
        self._manifest: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_all(self, task_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run the complete visualization and reporting pipeline.

        Args:
            task_names: Prediction task identifiers to process. When ``None``,
                        auto-discovers tasks from the experiments directory.

        Returns:
            List of manifest dicts, one per generated artefact.
        """
        self._logger.info("=" * 70)
        self._logger.info("Visualization and Research Reporting pipeline started.")
        self._logger.info("=" * 70)

        if task_names is None:
            task_names = self._discover_tasks()

        if not task_names:
            self._logger.warning("No task results found. Nothing to visualize.")
            return self._manifest

        # ---- Load results -------------------------------------------------
        all_exp: Dict[str, Dict] = {}
        all_eval: Dict[str, Dict] = {}
        all_acc_eff: Dict[str, pd.DataFrame] = {}
        all_stat: Dict[str, pd.DataFrame] = {}

        for task in task_names:
            exp = self._load_latest_experiment(task)
            eva = self._load_latest_eval_report(task)
            acc = self._load_latest_csv(task, "acc_eff_profiles")
            sta = self._load_latest_csv(task, "statistical_tests")
            if exp:
                all_exp[task] = exp
            if eva:
                all_eval[task] = eva
            if acc is not None:
                all_acc_eff[task] = acc
            if sta is not None:
                all_stat[task] = sta

        multi_task_csv = self._load_multi_task_csv()

        # ---- Per-task figures & tables ------------------------------------
        for task in task_names:
            self._logger.info(">>> Generating visualizations for task: %s", task)
            exp = all_exp.get(task, {})
            eva = all_eval.get(task, {})
            acc = all_acc_eff.get(task)
            sta = all_stat.get(task)

            self._run_safely(self._pred_plots.generate, task, eva, exp)
            self._run_safely(self._comp_plots.generate, task, eva, exp)
            self._run_safely(self._sel_plots.generate, task, exp)
            self._run_safely(self._res_plots.generate, task, acc)
            if sta is not None:
                self._run_safely(self._xai_plots.generate, task, eva)

        # ---- Multi-task figures -------------------------------------------
        if multi_task_csv is not None:
            self._run_safely(self._mt_plots.generate, all_exp, multi_task_csv)

        # ---- Tables ----------------------------------------------------------
        self._run_safely(
            self._tables.generate_all, all_exp, all_eval, all_acc_eff, all_stat
        )

        # ---- Report ----------------------------------------------------------
        self._run_safely(
            self._reporter.generate,
            all_exp, all_eval, all_acc_eff, all_stat, multi_task_csv, self._manifest
        )

        self._logger.info(
            "Visualization pipeline complete. %d artefacts generated.", len(self._manifest)
        )
        return self._manifest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_tasks(self) -> List[str]:
        """Auto-detect task names from experiment result filenames."""
        pattern = os.path.join(self._experiments_dir, "*.json")
        files = glob.glob(pattern)
        tasks = set()
        for f in files:
            basename = os.path.basename(f)
            parts = basename.split("_")
            if len(parts) >= 2:
                # First part before the experiment UUID fragment is the task name.
                # Task names themselves can contain underscores (e.g. Global_active_power)
                # so we try to match known task prefixes instead.
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    task = data.get("task_name")
                    if task:
                        tasks.add(task)
                except Exception:
                    pass
        return sorted(tasks)

    def _load_latest_experiment(self, task_name: str) -> Dict[str, Any]:
        """Load the most recently modified experiment JSON for the given task."""
        pattern = os.path.join(self._experiments_dir, f"{task_name}_*.json")
        files = glob.glob(pattern)
        if not files:
            self._logger.warning("[%s] No experiment records found.", task_name)
            return {}
        latest = max(files, key=os.path.getmtime)
        try:
            with open(latest, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._logger.info("[%s] Loaded experiment: %s", task_name, os.path.basename(latest))
            return data
        except Exception as exc:
            self._logger.error("[%s] Failed to load experiment: %s", task_name, exc)
            return {}

    def _load_latest_eval_report(self, task_name: str) -> Dict[str, Any]:
        """Load the most recent evaluation JSON report for the given task."""
        task_eval_dir = os.path.join(self._evaluation_dir, task_name)
        pattern = os.path.join(task_eval_dir, "eval_report_*.json")
        files = glob.glob(pattern)
        if not files:
            self._logger.warning("[%s] No evaluation report found.", task_name)
            return {}
        latest = max(files, key=os.path.getmtime)
        try:
            with open(latest, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._logger.info("[%s] Loaded evaluation report: %s", task_name, os.path.basename(latest))
            return data
        except Exception as exc:
            self._logger.error("[%s] Failed to load eval report: %s", task_name, exc)
            return {}

    def _load_latest_csv(self, task_name: str, prefix: str) -> Optional[pd.DataFrame]:
        """Load the most recent flat CSV artefact matching `prefix` for the given task."""
        task_eval_dir = os.path.join(self._evaluation_dir, task_name)
        pattern = os.path.join(task_eval_dir, f"{prefix}_*.csv")
        files = glob.glob(pattern)
        if not files:
            self._logger.debug("[%s] No CSV found for prefix '%s'.", task_name, prefix)
            return None
        latest = max(files, key=os.path.getmtime)
        try:
            df = pd.read_csv(latest)
            self._logger.info("[%s] Loaded CSV '%s': %d rows.", task_name, os.path.basename(latest), len(df))
            return df
        except Exception as exc:
            self._logger.error("[%s] Failed to load CSV '%s': %s", task_name, prefix, exc)
            return None

    def _load_multi_task_csv(self) -> Optional[pd.DataFrame]:
        """Load the cross-task comparison CSV."""
        pattern = os.path.join(
            self._evaluation_dir, "multi_task", "cross_task_comparison_*.csv"
        )
        files = glob.glob(pattern)
        if not files:
            self._logger.debug("No multi-task CSV found.")
            return None
        latest = max(files, key=os.path.getmtime)
        try:
            return pd.read_csv(latest)
        except Exception as exc:
            self._logger.error("Failed to load multi-task CSV: %s", exc)
            return None

    def _run_safely(self, fn, *args, **kwargs) -> None:
        """Execute a visualization function; log and continue on any exception."""
        try:
            generated = fn(*args, **kwargs)
            if isinstance(generated, list):
                self._manifest.extend(generated)
        except Exception as exc:
            self._logger.error(
                "Visualization step '%s' failed: %s", fn.__qualname__, exc, exc_info=True
            )

    def record_artefact(self, path: str, kind: str, task: str = "", model: str = "") -> None:
        """Public helper called by plotters to register generated artefacts."""
        self._manifest.append({
            "path": path,
            "kind": kind,
            "task": task,
            "model": model,
        })
