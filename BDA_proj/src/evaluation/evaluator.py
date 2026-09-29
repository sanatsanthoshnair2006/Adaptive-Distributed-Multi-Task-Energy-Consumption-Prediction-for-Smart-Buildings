"""
Evaluator Module
================

Master orchestrator for the evaluation and statistical validation framework.

Design Decisions:
    - Reads experiment metadata from the local results directory to discover
      the latest runs and candidate models without needing an external database.
    - Loads prediction artifacts from HDFS using the Task interface.
    - Orchestrates RegressionMetrics, StatisticalTests, and PerformanceAnalyzer.
    - Fails gracefully if one model's predictions are missing, allowing others
      to complete.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Any, List, Optional

from pyspark.sql import SparkSession

from src.logging.logger_factory import LoggerFactory
from src.ingestion.hdfs_client import HDFSClient
from src.tasks.base_task import BaseTask
from src.models.model_metrics import CandidateResult, ModelMetrics
from src.evaluation.regression_metrics import RegressionMetrics
from src.evaluation.statistical_tests import StatisticalTests
from src.evaluation.performance_analyzer import PerformanceAnalyzer
from src.evaluation.evaluation_report import EvaluationReport


class EvaluatorError(Exception):
    """Raised when evaluation fails unrecoverably."""


class Evaluator:
    """
    Executes the complete evaluation pipeline across prediction tasks.

    Args:
        config (Dict[str, Any]): Merged application configuration.
        spark (SparkSession): Active Spark session.
        hdfs_client (HDFSClient): Configured HDFS client.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        spark: SparkSession,
        hdfs_client: HDFSClient,
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._spark = spark
        
        self._metrics_calc = RegressionMetrics(config)
        self._stat_tests = StatisticalTests(config)
        self._analyzer = PerformanceAnalyzer()
        self._report = EvaluationReport(config, hdfs_client)
        
        exp_cfg = config.get("model", {}).get("experiment", {})
        project_root = config.get("project_root", ".")
        results_rel = exp_cfg.get("results_dir", "results/experiments")
        self._experiments_dir = os.path.join(project_root, results_rel)

    def evaluate_tasks(self, tasks: List[BaseTask]) -> None:
        """
        Run the evaluation pipeline for all provided tasks.
        """
        self._logger.info("=" * 70)
        self._logger.info("Evaluation and Validation Framework started.")
        self._logger.info("=" * 70)

        all_task_results = {}

        for task in tasks:
            task_name = task.get_task_name()
            self._logger.info(">>> Evaluating Task: %s", task_name)
            try:
                task_report = self._evaluate_single_task(task)
                if task_report:
                    all_task_results[task_name] = task_report
            except Exception as exc:
                self._logger.error(">>> Evaluation failed for Task %s: %s", task_name, exc, exc_info=True)

        if all_task_results:
            self._generate_multi_task_summary(all_task_results)

    def _evaluate_single_task(self, task: BaseTask) -> Optional[Dict[str, Any]]:
        task_name = task.get_task_name()
        
        # 1. Discover latest experiment
        exp_record = self._get_latest_experiment(task_name)
        if not exp_record:
            self._logger.warning("[%s] No experiment records found. Skipping evaluation.", task_name)
            return None

        experiment_id = exp_record.get("experiment_id")
        selected_model = exp_record.get("selected_model")
        candidates = exp_record.get("candidates", [])
        version = f"exp_{experiment_id[:8]}"
        
        if not selected_model:
            raise EvaluatorError(f"[{task_name}] Experiment {experiment_id} has no selected_model.")

        self._logger.info("[%s] Evaluating experiment %s. Selected: %s", task_name, version, selected_model)

        # 2. Reconstruct CandidateResults from JSON so PerformanceAnalyzer can use them
        ranked_candidates = self._reconstruct_candidates(exp_record)
        
        # 3. Load Predictions & Compute Regression Metrics
        model_predictions = {}
        extended_metrics = {}
        
        for candidate in candidates:
            try:
                preds_df = task.load_predictions(self._spark, candidate, version)
                preds_df.cache()
                model_predictions[candidate] = preds_df
                
                # Compute extended metrics (includes MAPE)
                metrics_res = self._metrics_calc.evaluate(preds_df)
                extended_metrics[candidate] = {
                    "rmse": metrics_res.rmse,
                    "mae": metrics_res.mae,
                    "r2": metrics_res.r2,
                    "mape": metrics_res.mape,
                    "mape_is_reliable": metrics_res.mape_is_reliable,
                    "num_rows": metrics_res.num_rows
                }
            except Exception as exc:
                self._logger.error("[%s/%s] Failed to load or evaluate predictions: %s", task_name, candidate, exc)

        if selected_model not in model_predictions:
            raise EvaluatorError(f"[{task_name}] Predictions for selected model '{selected_model}' missing.")

        # 4. Statistical Testing (Selected vs Candidates)
        stat_results = []
        selected_preds = model_predictions[selected_model]
        
        comparisons = [c for c in candidates if c != selected_model and c in model_predictions]
        num_comparisons = len(comparisons)
        
        for candidate in comparisons:
            cand_preds = model_predictions[candidate]
            stat_res = self._stat_tests.compare_models(
                baseline_name=selected_model,
                baseline_df=selected_preds,
                candidate_name=candidate,
                candidate_df=cand_preds,
                num_comparisons=num_comparisons
            )
            # Convert to dict for JSON
            stat_results.append({
                "baseline_model": stat_res.baseline_model,
                "candidate_model": stat_res.candidate_model,
                "test_name": stat_res.test_name,
                "statistic": stat_res.statistic,
                "p_value": stat_res.p_value,
                "alpha": stat_res.alpha,
                "is_significant": stat_res.is_significant,
                "correction_method": stat_res.correction_method,
                "sample_size": stat_res.sample_size,
                "notes": stat_res.notes
            })

        # Uncache
        for df in model_predictions.values():
            df.unpersist()

        # 5. Accuracy-Efficiency Analysis
        composite_vs_rmse = self._analyzer.analyze_composite_vs_baseline(task_name, ranked_candidates)
        acc_eff_profiles = self._analyzer.generate_accuracy_efficiency_profile(task_name, ranked_candidates)

        # 6. Generate and Save Reports
        full_report = {
            "task_name": task_name,
            "experiment_id": experiment_id,
            "selected_model": selected_model,
            "evaluation_metrics": extended_metrics,
            "statistical_tests": stat_results,
            "composite_analysis": composite_vs_rmse,
            "accuracy_efficiency_profiles": acc_eff_profiles,
            "original_experiment_metadata": {
                "timestamp": exp_record.get("timestamp"),
                "selection_profile": exp_record.get("selection_profile")
            }
        }
        
        self._report.save_task_report(task_name, experiment_id, full_report)
        self._report.save_flat_comparisons(task_name, experiment_id, acc_eff_profiles, "acc_eff_profiles")
        self._report.save_flat_comparisons(task_name, experiment_id, stat_results, "statistical_tests")
        
        self._logger.info(">>> Task %s evaluation complete.", task_name)
        
        return full_report

    def _get_latest_experiment(self, task_name: str) -> Dict[str, Any]:
        """Finds and loads the most recent JSON experiment record for the task."""
        pattern = os.path.join(self._experiments_dir, f"{task_name}_*.json")
        files = glob.glob(pattern)
        if not files:
            return {}
            
        latest_file = max(files, key=os.path.getmtime)
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self._logger.error("Failed to read experiment file %s: %s", latest_file, exc)
            return {}

    def _reconstruct_candidates(self, exp_record: Dict[str, Any]) -> List[CandidateResult]:
        """Reconstructs CandidateResult objects from the JSON dictionary."""
        ranked_list = []
        
        # In ExperimentManager, "composite_scores" and "raw_metrics" and "normalised_metrics" 
        # are stored in sorted order matching ranked_results.
        composite_scores = exp_record.get("composite_scores", [])
        raw_metrics = exp_record.get("raw_metrics", [])
        norm_metrics = exp_record.get("normalised_metrics", [])
        
        for i, comp in enumerate(composite_scores):
            raw = raw_metrics[i]
            norm = norm_metrics[i]
            
            m = ModelMetrics(
                task_name=exp_record.get("task_name", ""),
                model_name=raw["model_name"],
                rmse=raw["rmse"],
                mae=raw["mae"],
                r2=raw["r2"],
                training_time=raw["training_time"],
                prediction_time=raw["prediction_time"],
                memory_usage_mb=raw.get("memory_usage_mb"),
                cpu_usage_pct=raw.get("cpu_usage_pct"),
                experiment_id=raw.get("experiment_id", ""),
                selection_profile=raw.get("selection_profile", "")
            )
            
            cr = CandidateResult(
                metrics=m,
                normalised_rmse=norm["normalised_rmse"],
                normalised_mae=norm["normalised_mae"],
                normalised_r2_penalty=norm["normalised_r2_penalty"],
                normalised_training_time=norm["normalised_training_time"],
                normalised_prediction_time=norm["normalised_prediction_time"],
                normalised_memory_mb=norm["normalised_memory_mb"],
                normalised_cpu_pct=norm["normalised_cpu_pct"],
                composite_score=comp["composite_score"],
                rank=comp["rank"],
                is_selected=comp["is_selected"]
            )
            ranked_list.append(cr)
            
        return ranked_list

    def _generate_multi_task_summary(self, all_task_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Generates and saves a cross-task summary report to objectively compare
        difficulty and best models across all prediction tasks.
        """
        self._logger.info("Generating Multi-Task Analysis Summary...")
        
        summary_rows = []
        for task_name, report in all_task_results.items():
            selected_model = report["selected_model"]
            metrics = report["evaluation_metrics"].get(selected_model, {})
            
            comp_analysis = report.get("composite_analysis", {})
            diffs = comp_analysis.get("differences", {})
            
            summary_rows.append({
                "task_name": task_name,
                "selected_model": selected_model,
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "r2": metrics.get("r2"),
                "mape": metrics.get("mape"),
                "composite_vs_rmse_same": comp_analysis.get("same_model_selected"),
                "composite_score_diff": diffs.get("composite_score_diff", 0.0),
                "rmse_penalty_vs_baseline": diffs.get("rmse_diff", 0.0),
            })
            
        # Sort by RMSE (or task difficulty proxy)
        summary_rows.sort(key=lambda x: x["rmse"] if x["rmse"] is not None else float('inf'))
        
        # Save as a special cross-task report
        self._report.save_flat_comparisons(
            task_name="multi_task",
            experiment_id="summary",
            flat_data=summary_rows,
            prefix="cross_task_comparison"
        )
        self._logger.info("Multi-Task Analysis Summary saved.")
