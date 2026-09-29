"""
Base Task Module
=================

Defines the abstract contract that every prediction task must fulfil.

Design Decisions:
    - ABC enforcement:  Concrete tasks that forget to implement any required
      method will raise TypeError at *import time*, not at runtime.
    - No Spark coupling in the abstract layer:  The base class only imports
      typing helpers and the logger factory, keeping it lightweight and
      testable without a live SparkSession.
    - Strict time-series split:  The abstract train_test_split() signature
      enforces a ratio-based, non-shuffled cut so all tasks behave uniformly.
    - Parquet persistence contract:  save_predictions / load_predictions are
      abstract so each task controls its own schema while the registry can call
      them polymorphically.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame

from src.logging.logger_factory import LoggerFactory


class BaseTask(abc.ABC):
    """
    Abstract base for every multi-task prediction task.

    Concrete implementations must:
        1. Declare a unique task name (``get_task_name``).
        2. Declare their target and feature columns.
        3. Produce a ready-to-train dataset (``prepare_dataset``).
        4. Split that dataset respecting temporal order (``train_test_split``).
        5. Persist and reload predictions in Parquet format.

    Constructor Args:
        config (Dict[str, Any]):  Full merged application configuration dict
            (app + hadoop sections).  Each task reads the keys it needs.
        hdfs_client:  An ``HDFSClient`` instance used for Parquet I/O.
            Typed as ``Any`` here to avoid a hard dependency on the ingestion
            layer inside the abstract base.
    """

    def __init__(self, config: Dict[str, Any], hdfs_client: Any) -> None:
        self._config = config
        self._hdfs_client = hdfs_client
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

        # Resolve the HDFS predictions output directory from config.
        hadoop_cfg = config.get("hadoop", {})
        hdfs_cfg = hadoop_cfg.get("hdfs", {})
        base_path = hdfs_cfg.get("base_path", "/user/energy_prediction")
        predictions_subdir = (
            hdfs_cfg.get("paths", {}).get("predictions", "predictions")
        )
        # e.g.  /user/energy_prediction/predictions
        self._predictions_base_path = f"{base_path.rstrip('/')}/{predictions_subdir.lstrip('/')}"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_task_name(self) -> str:
        """
        Return a unique, human-readable identifier for this task.

        Returns:
            str: Task name (e.g. ``"global_active_power"``).
        """

    @abc.abstractmethod
    def get_target_column(self) -> str:
        """
        Return the name of the column this task predicts.

        Returns:
            str: Column name (e.g. ``"Global_active_power"``).
        """

    @abc.abstractmethod
    def get_feature_columns(self) -> List[str]:
        """
        Return the ordered list of feature column names used for training.

        These columns are read from the engineered feature DataFrame produced
        by ``FeatureEngineer`` — no preprocessing is duplicated here.

        Returns:
            List[str]: Feature column names.
        """

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def prepare_dataset(self, features_df: DataFrame) -> DataFrame:
        """
        Select and finalise the columns required by this specific task.

        Accepts the *full* engineered feature DataFrame and returns a
        narrowed DataFrame containing only the features + target + Timestamp.
        This keeps downstream steps lean and avoids carrying irrelevant columns
        through model training.

        Args:
            features_df (DataFrame):  Engineered Spark DataFrame from the
                feature store.  Must contain ``Timestamp``, the target column,
                and all feature columns returned by ``get_feature_columns()``.

        Returns:
            DataFrame:  Narrowed DataFrame ready for splitting and training.
        """

    @abc.abstractmethod
    def train_test_split(
        self,
        df: DataFrame,
        test_ratio: float = 0.2,
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Perform a **temporal** (non-shuffled) train/test split.

        The split must preserve chronological order — i.e. the first
        ``(1 - test_ratio) * 100 %`` rows form the training set and the
        remaining rows form the test set.

        Args:
            df (DataFrame):  Prepared dataset ordered by ``Timestamp``.
            test_ratio (float):  Fraction of rows reserved for testing.
                Must be in the range ``(0, 1)``.  Defaults to ``0.2``.

        Returns:
            Tuple[DataFrame, DataFrame]:  ``(train_df, test_df)``.

        Raises:
            ValueError:  If ``test_ratio`` is not in ``(0, 1)``.
        """

    # ------------------------------------------------------------------
    # Prediction persistence
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def save_predictions(
        self,
        predictions_df: DataFrame,
        model_name: str,
        version: str = "v1",
        overwrite: bool = False,
    ) -> str:
        """
        Persist a predictions DataFrame to HDFS as Parquet.

        The DataFrame **must** contain the following columns before calling
        this method:
            - ``Timestamp``     — original observation timestamp
            - ``Actual``        — ground-truth target value
            - ``Predicted``     — model output
            - ``Task``          — task name (``get_task_name()``)
            - ``Model``         — model identifier (``model_name`` arg)

        Args:
            predictions_df (DataFrame):  DataFrame with the columns above.
            model_name (str):   Identifier for the model used
                                (e.g. ``"random_forest"``).
            version (str):      Version label for the output path.
                                Defaults to ``"v1"``.
            overwrite (bool):   Whether to overwrite an existing version.
                                Defaults to ``False``.

        Returns:
            str:  The HDFS path where predictions were written.

        Raises:
            TaskPersistenceError:  If the write fails or the path exists and
                ``overwrite=False``.
        """

    @abc.abstractmethod
    def load_predictions(
        self,
        spark_session: Any,
        model_name: str,
        version: str = "v1",
    ) -> DataFrame:
        """
        Load previously saved predictions from HDFS.

        Args:
            spark_session:  Active ``SparkSession``.
            model_name (str):   Identifier for the model whose predictions
                                should be loaded.
            version (str):      Version label to load.  Defaults to ``"v1"``.

        Returns:
            DataFrame:  Predictions DataFrame.

        Raises:
            TaskPersistenceError:  If the path does not exist or read fails.
        """

    # ------------------------------------------------------------------
    # Helpers (non-abstract, shared across all tasks)
    # ------------------------------------------------------------------

    def _build_predictions_path(self, model_name: str, version: str) -> str:
        """
        Construct the HDFS output path for a predictions artifact.

        Path pattern::

            <predictions_base>/<task_name>/<model_name>/<version>

        Args:
            model_name (str): Model identifier.
            version (str):    Version label.

        Returns:
            str: Fully qualified HDFS path.
        """
        task = self.get_task_name()
        return (
            f"{self._predictions_base_path.rstrip('/')}/"
            f"{task}/{model_name}/{version}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"task={self.get_task_name()!r}, "
            f"target={self.get_target_column()!r})"
        )


# ---------------------------------------------------------------------------
# Shared exception
# ---------------------------------------------------------------------------

class TaskPersistenceError(Exception):
    """Raised when saving or loading task predictions fails."""
