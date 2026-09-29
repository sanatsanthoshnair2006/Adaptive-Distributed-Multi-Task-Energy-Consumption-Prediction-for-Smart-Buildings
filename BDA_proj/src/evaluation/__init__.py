"""
Evaluation Module
=================

Provides post-training evaluation, statistical validation, and reporting.

Exposes the primary Evaluator orchestrator and its constituent classes.
"""

from src.evaluation.evaluator import Evaluator, EvaluatorError
from src.evaluation.regression_metrics import RegressionMetrics, RegressionMetricsResult
from src.evaluation.statistical_tests import StatisticalTests, StatisticalTestResult
from src.evaluation.performance_analyzer import PerformanceAnalyzer
from src.evaluation.evaluation_report import EvaluationReport

__all__ = [
    "Evaluator",
    "EvaluatorError",
    "RegressionMetrics",
    "RegressionMetricsResult",
    "StatisticalTests",
    "StatisticalTestResult",
    "PerformanceAnalyzer",
    "EvaluationReport",
]
