"""
Concrete Task Mixin Module
===========================

Provides a default, reusable implementation of the ``BaseTask`` methods that
are *identical* across all four prediction tasks:

    - ``prepare_dataset``      — selects feature + target + Timestamp columns
    - ``train_test_split``     — temporal (non-shuffled) row-based split
    - ``save_predictions``     — writes Parquet to HDFS with overwrite guard
    - ``load_predictions``     — reads Parquet from HDFS

Design Decisions:
    - DRY via Mixin:  Instead of duplicating ~150 lines across four task files,
      a single ``ConcreteTaskMixin`` holds the shared logic.  Each concrete task
      class does a simple ``class TaskFoo(ConcreteTaskMixin, BaseTask)`` and
      only overrides the identity / feature-list methods.
    - MRO ordering:  ``ConcreteTaskMixin`` is placed *before* ``BaseTask`` in
      the MRO so Python resolves the concrete implementations from the mixin
      first, satisfying the ABC requirements of BaseTask.
    - Temporal split via approxQuantile:  Spark's ``approxQuantile`` on the
      monotonically increasing id column gives a deterministic, reproducible
      cut-point without collecting all timestamps to the driver.
    - No shuffle guarantee:  The split never calls ``randomSplit``; it always
      uses an ordered index to maintain temporal integrity.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from pyspark.sql import DataFrame, functions as F

from src.logging.logger_factory import LoggerFactory
from src.tasks.base_task import TaskPersistenceError


_logger = LoggerFactory.get_logger("tasks.concrete_task_mixin")


class ConcreteTaskMixin:
    """
    Shared concrete implementation for all prediction tasks.

    Inheriting classes must still provide:
        - ``get_task_name()``
        - ``get_target_column()``
        - ``get_feature_columns()``

    Everything else is handled here.

    Inheriting classes must also call ``BaseTask.__init__`` (which sets
    ``self._config``, ``self._hdfs_client``, ``self._predictions_base_path``
    and ``self._logger``).
    """

    # ------------------------------------------------------------------
    # Data preparation  (implements BaseTask.prepare_dataset)
    # ------------------------------------------------------------------

    def prepare_dataset(self, features_df: DataFrame) -> DataFrame:
        """
        Select Timestamp + feature columns + target column from the full
        engineered feature DataFrame.

        Rows with a null target value are dropped to guarantee a clean
        training set — nulls can arise from lag features at the head of the
        time series.

        Args:
            features_df (DataFrame):  Full engineered Spark DataFrame.

        Returns:
            DataFrame:  Narrowed, null-free DataFrame ordered by Timestamp.
        """
        task_name = self.get_task_name()          # type: ignore[attr-defined]
        target = self.get_target_column()         # type: ignore[attr-defined]
        features = self.get_feature_columns()     # type: ignore[attr-defined]
        logger = self._logger                     # type: ignore[attr-defined]

        logger.info("=" * 60)
        logger.info("[%s] Preparing dataset...", task_name)

        required_cols = ["Timestamp"] + features + [target]

        # Validate that all required columns are present
        available = set(features_df.columns)
        missing = [c for c in required_cols if c not in available]
        if missing:
            raise ValueError(
                f"[{task_name}] Missing columns in features DataFrame: {missing}"
            )

        df = features_df.select(required_cols)

        # Drop rows where the target is null (e.g. lag warm-up rows)
        before = df.count()
        df = df.dropna(subset=[target])
        after = df.count()

        dropped = before - after
        if dropped > 0:
            logger.warning(
                "[%s] Dropped %d rows with null target '%s'.",
                task_name, dropped, target
            )

        # Ensure chronological order for downstream temporal operations
        df = df.orderBy("Timestamp")

        logger.info(
            "[%s] Dataset ready. Rows: %d | Columns: %d",
            task_name, after, len(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # Train / test split  (implements BaseTask.train_test_split)
    # ------------------------------------------------------------------

    def train_test_split(
        self,
        df: DataFrame,
        test_ratio: float = 0.2,
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Temporal train/test split that preserves chronological order.

        Implementation strategy
        -----------------------
        1. Attach a monotonically increasing integer index (``_row_idx``) that
           reflects the existing temporal ordering of *df*.
        2. Count total rows.
        3. Compute the cut-point index = ``floor(total * (1 - test_ratio))``.
        4. Filter: train ← ``_row_idx < cut_point``,
                   test  ← ``_row_idx >= cut_point``.
        5. Drop the helper column before returning.

        This avoids collecting data to the driver and never shuffles.

        Args:
            df (DataFrame):     Temporally ordered prepared dataset.
            test_ratio (float): Fraction of rows for test set. ``(0, 1)``.

        Returns:
            Tuple[DataFrame, DataFrame]: ``(train_df, test_df)``.

        Raises:
            ValueError: If ``test_ratio`` is not in ``(0, 1)``.
        """
        task_name = self.get_task_name()    # type: ignore[attr-defined]
        logger = self._logger               # type: ignore[attr-defined]

        if not (0.0 < test_ratio < 1.0):
            raise ValueError(
                f"[{task_name}] test_ratio must be in (0, 1), got {test_ratio}."
            )

        logger.info(
            "[%s] Performing temporal train/test split (test_ratio=%.2f)...",
            task_name, test_ratio
        )

        # Attach a sequential row index and cache to lock the ID assignment
        # across subsequent approxQuantile and filtering operations.
        df_indexed = df.withColumn("_row_idx", F.monotonically_increasing_id()).cache()

        total_rows = df_indexed.count()
        cut_index = df_indexed.approxQuantile(
            "_row_idx", [1.0 - test_ratio], 0.001
        )[0]

        train_df = df_indexed.filter(F.col("_row_idx") < cut_index).drop("_row_idx")
        test_df  = df_indexed.filter(F.col("_row_idx") >= cut_index).drop("_row_idx")


        train_count = total_rows - int(total_rows * test_ratio)
        test_count  = total_rows - train_count

        logger.info(
            "[%s] Split complete. ~Train rows: %d | ~Test rows: %d",
            task_name, train_count, test_count
        )
        return train_df, test_df

    # ------------------------------------------------------------------
    # Prediction persistence  (implements BaseTask.save_predictions)
    # ------------------------------------------------------------------

    def save_predictions(
        self,
        predictions_df: DataFrame,
        model_name: str,
        version: str = "v1",
        overwrite: bool = False,
    ) -> str:
        """
        Save predictions DataFrame to HDFS as Parquet.

        Expected schema of ``predictions_df``:
            - Timestamp  (TimestampType)
            - Actual     (DoubleType)
            - Predicted  (DoubleType)
            - Task       (StringType)
            - Model      (StringType)

        Args:
            predictions_df (DataFrame): Predictions to persist.
            model_name (str):           Model identifier.
            version (str):              Version label.
            overwrite (bool):           Allow overwriting existing data.

        Returns:
            str: HDFS path where Parquet was written.

        Raises:
            TaskPersistenceError: On write failure or path conflict.
        """
        task_name = self.get_task_name()    # type: ignore[attr-defined]
        logger = self._logger               # type: ignore[attr-defined]
        hdfs_client = self._hdfs_client     # type: ignore[attr-defined]
        build_path = self._build_predictions_path  # type: ignore[attr-defined]

        output_path = build_path(model_name, version)

        logger.info("=" * 60)
        logger.info(
            "[%s] Saving predictions to: %s", task_name, output_path
        )

        if hdfs_client.exists(output_path):
            if overwrite:
                logger.warning(
                    "[%s] Path exists. Overwriting: %s", task_name, output_path
                )
            else:
                msg = (
                    f"[{task_name}] Predictions already exist at {output_path}. "
                    f"Set overwrite=True to replace."
                )
                logger.error(msg)
                raise TaskPersistenceError(msg)

        mode = "overwrite" if overwrite else "errorifexists"

        try:
            spark_write_path = hdfs_client.spark_path(output_path)
            predictions_df.write.mode(mode).parquet(spark_write_path)
            logger.info(
                "[%s] Predictions saved successfully to %s",
                task_name, spark_write_path
            )
            return output_path
        except Exception as exc:
            msg = (
                f"[{task_name}] Failed to save predictions to {output_path}: {exc}"
            )
            logger.error(msg)
            raise TaskPersistenceError(msg) from exc

    # ------------------------------------------------------------------
    # Load predictions  (implements BaseTask.load_predictions)
    # ------------------------------------------------------------------

    def load_predictions(
        self,
        spark_session: Any,
        model_name: str,
        version: str = "v1",
    ) -> DataFrame:
        """
        Load previously saved predictions from HDFS.

        Args:
            spark_session:  Active SparkSession.
            model_name (str): Model identifier.
            version (str):    Version label.

        Returns:
            DataFrame: Predictions DataFrame.

        Raises:
            TaskPersistenceError: If path not found or read fails.
        """
        task_name = self.get_task_name()    # type: ignore[attr-defined]
        logger = self._logger               # type: ignore[attr-defined]
        hdfs_client = self._hdfs_client     # type: ignore[attr-defined]
        build_path = self._build_predictions_path  # type: ignore[attr-defined]

        input_path = build_path(model_name, version)

        if not hdfs_client.exists(input_path):
            msg = (
                f"[{task_name}] Predictions not found at {input_path}. "
                f"Ensure save_predictions was called for model='{model_name}', "
                f"version='{version}'."
            )
            logger.error(msg)
            raise TaskPersistenceError(msg)

        logger.info(
            "[%s] Loading predictions from: %s", task_name, input_path
        )
        try:
            spark_read_path = hdfs_client.spark_path(input_path)
            df = spark_session.read.parquet(spark_read_path)
            logger.info(
                "[%s] Predictions loaded. Rows: %d", task_name, df.count()
            )
            return df
        except Exception as exc:
            msg = (
                f"[{task_name}] Failed to load predictions from {input_path}: {exc}"
            )
            logger.error(msg)
            raise TaskPersistenceError(msg) from exc
