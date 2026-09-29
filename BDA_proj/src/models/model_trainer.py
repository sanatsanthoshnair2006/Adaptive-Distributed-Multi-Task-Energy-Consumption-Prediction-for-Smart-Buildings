"""
Model Trainer Module
=====================

Handles the execution of Spark ML estimator training and resource monitoring.

Design Decisions:
    - Single Responsibility: Separates the act of training and resource
      measurement from the broader orchestration loops.
    - Graceful Degradation: If ``psutil`` is not installed, resource metrics
      are safely logged as ``None`` rather than crashing the pipeline.
    - Distributed Execution: Relies on Spark ML's native distributed ``fit()``,
      avoiding driver-side memory bottlenecks.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

from pyspark.ml import Estimator, Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.logging.logger_factory import LoggerFactory


class ModelTrainerError(Exception):
    """Raised when training fails."""


class ModelTrainer:
    """
    Executes the training of a Spark ML estimator, capturing execution time
    and resource usage.

    Args:
        features_col (str): Column name for the assembled feature vector.
                            Defaults to ``"features"``.
        label_col (str):    Column name for the target label.
                            Defaults to ``"label"``.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        features_col: str = "features",
        label_col: str = "label",
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        if isinstance(config, str):
            features_col = config
        self._features_col = features_col
        self._label_col = label_col

        try:
            import psutil
            self._psutil = psutil
            self._logger.debug("psutil available for resource monitoring.")
        except ImportError:
            self._psutil = None
            self._logger.warning("psutil not available; resource metrics will be None.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        task_name: str,
        model_name: str,
        estimator: Estimator,
        train_df: DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> Tuple[PipelineModel, Dict[str, Any]]:
        """
        Builds a pipeline, trains the estimator, and records performance.

        Args:
            task_name (str):       Identifier for logging.
            model_name (str):      Model identifier for logging.
            estimator (Estimator): Unfitted Spark ML estimator.
            train_df (DataFrame):  Prepared training dataset.
            feature_cols (list):   Ordered list of feature column names.
            target_col (str):      Target column name.

        Returns:
            Tuple[PipelineModel, Dict[str, Any]]: The fitted model and a
                dictionary of training metadata (time, memory, CPU).

        Raises:
            ModelTrainerError: If training fails.
        """
        self._logger.info("[%s/%s] Starting model training...", task_name, model_name)

        # 1. Prepare pipeline
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=self._features_col,
            handleInvalid="skip",
        )
        pipeline = Pipeline(stages=[assembler, estimator])

        # 2. Prepare labeled dataset
        train_labeled = train_df.withColumn(
            self._label_col, F.col(target_col).cast("double")
        ).dropna(subset=feature_cols + [self._label_col])

        # 3. Capture baseline resources
        mem_before_mb = self._get_memory_mb()

        # 4. Execute distributed training
        t0 = time.perf_counter()
        try:
            fitted_model: PipelineModel = pipeline.fit(train_labeled)
        except Exception as exc:
            msg = f"Training failed for {model_name} on task {task_name}: {exc}"
            self._logger.error(msg, exc_info=True)
            raise ModelTrainerError(msg) from exc
        
        training_time = time.perf_counter() - t0

        # 5. Capture post-training resources
        mem_after_mb = self._get_memory_mb()
        memory_delta = (
            (mem_after_mb - mem_before_mb)
            if (mem_before_mb is not None and mem_after_mb is not None)
            else None
        )
        cpu_usage = self._get_cpu_pct()

        self._logger.info(
            "[%s/%s] Training complete in %.3fs.",
            task_name, model_name, training_time
        )

        metadata = {
            "training_time": training_time,
            "memory_usage_mb": memory_delta,
            "cpu_usage_pct": cpu_usage,
        }

        return fitted_model, metadata

    # ------------------------------------------------------------------
    # Resource Monitoring Helpers
    # ------------------------------------------------------------------

    def _get_memory_mb(self) -> Optional[float]:
        """Return current process RSS memory in MB, or None."""
        if self._psutil is None:
            return None
        try:
            proc = self._psutil.Process(os.getpid())
            return proc.memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    def _get_cpu_pct(self) -> Optional[float]:
        """Return a 1-second CPU usage sample for this process, or None."""
        if self._psutil is None:
            return None
        try:
            proc = self._psutil.Process(os.getpid())
            return proc.cpu_percent(interval=1.0)
        except Exception:
            return None
