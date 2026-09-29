"""
Performance Analyzer Module
===========================

Calculates accuracy-efficiency relationships and evaluates the practical impact
of the Composite Adaptive Selection framework vs traditional minimum-RMSE selection.

Design Decisions:
    - Independent analysis layer that does not modify the raw metrics.
    - Generates flat, serializable dictionaries to facilitate downstream reporting.
    - Safely handles missing resource metrics (e.g., if psutil was unavailable).
"""

from __future__ import annotations

from typing import Dict, Any, List

from src.logging.logger_factory import LoggerFactory
from src.models.model_metrics import CandidateResult


class PerformanceAnalyzer:
    """Analyzes trade-offs between prediction accuracy and resource consumption."""

    def __init__(self) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

    def analyze_composite_vs_baseline(
        self,
        task_name: str,
        ranked_candidates: List[CandidateResult]
    ) -> Dict[str, Any]:
        """
        Compare the model selected by the composite framework against the model
        that would have been selected using raw RMSE alone.

        Args:
            task_name (str): Identifier for the task.
            ranked_candidates: List of CandidateResult objects, already sorted by composite score.

        Returns:
            Dict: Dictionary containing the comparison results.
        """
        if not ranked_candidates:
            return {"task_name": task_name, "status": "no_candidates"}

        # The composite winner is simply the first item, as it's sorted by composite score
        composite_winner = ranked_candidates[0]

        # The RMSE winner is the one with the absolute lowest raw RMSE
        rmse_winner = min(ranked_candidates, key=lambda c: c.metrics.rmse)

        same_model = (composite_winner.metrics.model_name == rmse_winner.metrics.model_name)

        diff = {}
        if not same_model:
            # Calculate differences (Composite Winner - RMSE Winner)
            # Positive diff in RMSE means composite winner is worse at accuracy.
            # Negative diff in time means composite winner is faster.
            diff = {
                "rmse_diff": composite_winner.metrics.rmse - rmse_winner.metrics.rmse,
                "mae_diff": composite_winner.metrics.mae - rmse_winner.metrics.mae,
                "training_time_diff": composite_winner.metrics.training_time - rmse_winner.metrics.training_time,
                "prediction_time_diff": composite_winner.metrics.prediction_time - rmse_winner.metrics.prediction_time,
                "composite_score_diff": composite_winner.composite_score - rmse_winner.composite_score,
            }

            mem_c = composite_winner.metrics.memory_usage_mb
            mem_r = rmse_winner.metrics.memory_usage_mb
            if mem_c is not None and mem_r is not None:
                diff["memory_diff_mb"] = mem_c - mem_r
            else:
                diff["memory_diff_mb"] = None

            cpu_c = composite_winner.metrics.cpu_usage_pct
            cpu_r = rmse_winner.metrics.cpu_usage_pct
            if cpu_c is not None and cpu_r is not None:
                diff["cpu_diff_pct"] = cpu_c - cpu_r
            else:
                diff["cpu_diff_pct"] = None

        self._logger.info(
            "[%s] Composite Selection vs RMSE Baseline: Same Model = %s",
            task_name, same_model
        )

        return {
            "task_name": task_name,
            "composite_winner": composite_winner.metrics.model_name,
            "rmse_winner": rmse_winner.metrics.model_name,
            "same_model_selected": same_model,
            "differences": diff
        }

    def generate_accuracy_efficiency_profile(
        self,
        task_name: str,
        ranked_candidates: List[CandidateResult]
    ) -> List[Dict[str, Any]]:
        """
        Extract structured accuracy vs. efficiency data for visualization or reporting.

        Args:
            task_name (str): Identifier for the task.
            ranked_candidates: List of CandidateResult objects.

        Returns:
            List[Dict]: Flat list of profiles per model.
        """
        profiles = []
        for result in ranked_candidates:
            m = result.metrics
            prof = {
                "task_name": task_name,
                "model_name": m.model_name,
                "rank": result.rank,
                "rmse": m.rmse,
                "mae": m.mae,
                "training_time": m.training_time,
                "prediction_time": m.prediction_time,
                "memory_usage_mb": m.memory_usage_mb,
                "cpu_usage_pct": m.cpu_usage_pct,
                "composite_score": result.composite_score
            }
            profiles.append(prof)
            
        return profiles
