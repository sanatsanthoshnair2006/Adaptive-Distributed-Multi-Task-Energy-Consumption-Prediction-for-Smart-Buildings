"""
Regression Metrics Module
=========================

Robust regression metrics computation (RMSE, MAE, R², MAPE) with safety features.

Design Decisions:
    - Independent calculation module, separating math from orchestration.
    - Zero-handling for MAPE via a configurable epsilon to prevent division-by-zero.
    - Safe evaluation flags for instances where MAPE is mathematically inappropriate
      (e.g., highly zero-inflated distributions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator

from src.logging.logger_factory import LoggerFactory


@dataclass
class RegressionMetricsResult:
    """Stores the calculated regression metrics and metadata."""
    rmse: float
    mae: float
    r2: float
    mape: float
    mape_is_reliable: bool
    num_rows: int


class RegressionMetrics:
    """
    Computes robust regression metrics on Spark DataFrames.

    Args:
        config (Dict[str, Any]): Merged application configuration.
        label_col (str): Ground truth column name. Defaults to "label".
        prediction_col (str): Prediction column name. Defaults to "prediction".
    """

    def __init__(
        self,
        config: Dict[str, Any],
        label_col: str = "label",
        prediction_col: str = "prediction"
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._label_col = label_col
        self._prediction_col = prediction_col

        eval_cfg = config.get("evaluation", {})
        metrics_cfg = eval_cfg.get("metrics", {})
        self._mape_epsilon = float(metrics_cfg.get("mape_epsilon", 1e-6))
        
        self._rmse_eval = RegressionEvaluator(
            labelCol=self._label_col,
            predictionCol=self._prediction_col,
            metricName="rmse"
        )
        self._mae_eval = RegressionEvaluator(
            labelCol=self._label_col,
            predictionCol=self._prediction_col,
            metricName="mae"
        )
        self._r2_eval = RegressionEvaluator(
            labelCol=self._label_col,
            predictionCol=self._prediction_col,
            metricName="r2"
        )

    def evaluate(self, predictions_df: DataFrame) -> RegressionMetricsResult:
        """
        Evaluate all standard regression metrics on the provided DataFrame.

        Args:
            predictions_df: DataFrame containing label and prediction columns.

        Returns:
            RegressionMetricsResult: Dataclass containing the computed metrics.
        """
        num_rows = predictions_df.count()
        if num_rows == 0:
            self._logger.warning("Empty DataFrame provided for evaluation. Returning zeros.")
            return RegressionMetricsResult(0.0, 0.0, 0.0, 0.0, False, 0)

        # 1. Standard Spark MLlib Metrics
        rmse = float(self._rmse_eval.evaluate(predictions_df))
        mae = float(self._mae_eval.evaluate(predictions_df))
        r2 = float(self._r2_eval.evaluate(predictions_df))

        # 2. Custom Robust MAPE Calculation
        # MAPE = (1/n) * sum(|(y - y_pred) / y|)
        # To avoid division by zero, we clamp |y| to a minimum of epsilon.
        # We also track how many values were clamped to flag reliability.
        
        abs_y = F.abs(F.col(self._label_col))
        safe_y = F.when(abs_y < self._mape_epsilon, self._mape_epsilon).otherwise(abs_y)
        
        mape_col = F.abs((F.col(self._label_col) - F.col(self._prediction_col)) / safe_y)
        
        # Calculate mean MAPE and count of near-zero targets in one pass
        stats = predictions_df.select(
            F.mean(mape_col).alias("mape"),
            F.sum(F.when(abs_y < self._mape_epsilon, 1).otherwise(0)).alias("near_zero_count")
        ).collect()[0]

        mape = float(stats["mape"]) if stats["mape"] is not None else 0.0
        near_zero_count = int(stats["near_zero_count"]) if stats["near_zero_count"] is not None else 0
        
        # If more than 5% of the data was near zero, MAPE is mathematically questionable
        mape_is_reliable = (near_zero_count / num_rows) < 0.05

        if not mape_is_reliable:
            self._logger.debug(
                "MAPE is flagged as unreliable: %d/%d (%.2f%%) targets were near zero.",
                near_zero_count, num_rows, (near_zero_count/num_rows)*100
            )

        return RegressionMetricsResult(
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            mape_is_reliable=mape_is_reliable,
            num_rows=num_rows
        )
