"""
Visualization Module
====================

Provides publication-quality figure generation, structured table creation,
and automated experiment report generation for the adaptive energy prediction
research framework.

Public API
----------
    VisualizationManager   — top-level orchestrator
    PredictionPlots        — actual vs predicted, residuals
    ModelComparisonPlots   — candidate bar charts
    SelectionPlots         — composite score breakdown / ranking
    ResourcePlots          — accuracy-efficiency scatter plots
    MultiTaskPlots         — cross-task comparisons
    ExplainabilityPlots    — feature importance (SHAP optional)
    TableGenerator         — publication-ready tables
    ReportGenerator        — Markdown / HTML experiment reports
"""

from src.visualization.visualization_manager import VisualizationManager
from src.visualization.prediction_plots import PredictionPlots
from src.visualization.model_comparison_plots import ModelComparisonPlots
from src.visualization.selection_plots import SelectionPlots
from src.visualization.resource_plots import ResourcePlots
from src.visualization.multi_task_plots import MultiTaskPlots
from src.visualization.explainability_plots import ExplainabilityPlots
from src.visualization.table_generator import TableGenerator
from src.visualization.report_generator import ReportGenerator

__all__ = [
    "VisualizationManager",
    "PredictionPlots",
    "ModelComparisonPlots",
    "SelectionPlots",
    "ResourcePlots",
    "MultiTaskPlots",
    "ExplainabilityPlots",
    "TableGenerator",
    "ReportGenerator",
]
