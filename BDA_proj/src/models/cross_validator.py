"""
Cross Validator Module
=======================

Time-aware validation splits for hyperparameter tuning.

**Why not standard K-Fold?**

Standard K-fold cross-validation randomly shuffles the dataset before
splitting.  For time-series data this causes severe *data leakage*:
future observations (lag features, rolling statistics) may appear in the
training fold while their source rows are in the validation fold.
Any hyperparameter set selected via random K-fold would be selected on
optimistically biased validation scores.

**Implemented strategies**

Three chronological splitting strategies are supported, all configurable:

1. ``chronological``
   The simplest safe approach.  The training split is divided into
   ``n_folds`` sequential slices.  In each fold, the first
   ``(n_folds - 1)`` slices are training data and the last slice is the
   validation holdout.  A single validation score is produced per
   parameter combination (the score on the last slice).

   This strategy is the default because it integrates cleanly with Spark
   DataFrames without requiring per-row index manipulation that would
   force data collection to the driver.

2. ``expanding_window``
   Each successive fold adds one more slice to the training window.
   Fold k trains on slices [0..k] and validates on slice [k+1].
   This mirrors the realistic scenario where a deployed model is
   periodically retrained on all available historical data.

3. ``rolling_window``
   A fixed-width training window slides forward.  Fold k trains on
   slices [k..k+w-1] and validates on slice [k+w].  This isolates
   how well a model generalises within a fixed historical horizon and
   is appropriate when older data may be less relevant.

**Data leakage prevention**

All strategies guarantee:
- Validation rows always have timestamps strictly greater than all
  training rows within the same fold.
- No row appears in both the training and validation partition of any
  fold.
- The held-out test split (managed by ``BaseTask.train_test_split``)
  is never included in any validation fold.

Design Decisions:
    - Spark-native splits:  Rather than collecting the full DataFrame to
      the driver and slicing as a list, we use ``approxQuantile`` on a
      monotonically increasing index to find chronological cut points and
      then filter distributed DataFrames.  This avoids OOM for large
      datasets.
    - The split-finding action (``approxQuantile``) is performed once per
      strategy instantiation, not once per fold, to minimise Spark jobs.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.logging.logger_factory import LoggerFactory


class CrossValidatorError(Exception):
    """Raised when cross-validation configuration is invalid."""


class TimeSeriesCrossValidator:
    """
    Generates temporally-ordered train/validation fold pairs from a
    Spark DataFrame for use during hyperparameter tuning.

    **Important**: Only the *training split* (from ``BaseTask.train_test_split``)
    should be passed here.  The held-out test set must remain untouched.

    Args:
        strategy (str):   One of ``"chronological"``, ``"expanding_window"``,
                          ``"rolling_window"``.  Defaults to ``"chronological"``.
        n_folds (int):    Number of folds.  Minimum 2.  Defaults to 3.
        val_ratio (float): Fraction of the training data reserved for the
                           validation holdout in the chronological strategy.
                           In expanding/rolling window, each fold's validation
                           slice has width ``total / (n_folds + 1)``.
                           Defaults to 0.15.
    """

    _VALID_STRATEGIES = {"chronological", "expanding_window", "rolling_window"}

    def __init__(
        self,
        strategy: str = "chronological",
        n_folds: int = 3,
        val_ratio: float = 0.15,
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

        if strategy not in self._VALID_STRATEGIES:
            raise CrossValidatorError(
                f"Invalid CV strategy '{strategy}'. "
                f"Supported: {sorted(self._VALID_STRATEGIES)}"
            )
        if n_folds < 2:
            raise CrossValidatorError(
                f"n_folds must be >= 2, got {n_folds}."
            )
        if not (0.0 < val_ratio < 1.0):
            raise CrossValidatorError(
                f"val_ratio must be in (0, 1), got {val_ratio}."
            )

        self._strategy = strategy
        self._n_folds = n_folds
        self._val_ratio = val_ratio

        self._logger.info(
            "TimeSeriesCrossValidator initialised: strategy=%s, n_folds=%d, "
            "val_ratio=%.2f",
            strategy, n_folds, val_ratio,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def split(
        self, train_df: DataFrame
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """
        Generate a list of ``(fold_train_df, fold_val_df)`` pairs.

        Each pair is a pair of Spark DataFrames.  They are *not* cached
        here; the caller (``HyperparameterTuner``) decides when to cache.

        Args:
            train_df (DataFrame):  Temporally ordered training dataset.
                Should already have ``Timestamp`` and ``_row_idx`` absent
                (will be re-attached internally).

        Returns:
            List of (train, validation) DataFrame tuples, one per fold.

        Raises:
            CrossValidatorError:  If the DataFrame cannot be split.
        """
        self._logger.info(
            "Generating %s folds using strategy '%s'...",
            self._n_folds, self._strategy
        )

        # Attach a strictly sequential row index that reflects the
        # existing chronological ordering of train_df.
        from pyspark.sql.window import Window
        df_indexed = train_df.withColumn(
            "_cv_idx", F.row_number().over(Window.orderBy("Timestamp"))
        ).cache()

        # Find cut-point quantiles in a single Spark action
        cut_points = self._compute_cut_points(df_indexed)

        if self._strategy == "chronological":
            folds = self._chronological_folds(df_indexed, cut_points)
        elif self._strategy == "expanding_window":
            folds = self._expanding_window_folds(df_indexed, cut_points)
        else:  # rolling_window
            folds = self._rolling_window_folds(df_indexed, cut_points)

        # Strip the helper index from every fold DataFrame
        cleaned = [
            (f_train.drop("_cv_idx"), f_val.drop("_cv_idx"))
            for f_train, f_val in folds
        ]

        self._logger.info(
            "Generated %d fold pair(s).", len(cleaned)
        )
        return cleaned

    # ------------------------------------------------------------------
    # Cut-point computation (single Spark action)
    # ------------------------------------------------------------------

    def _compute_cut_points(self, df_indexed: DataFrame) -> List[float]:
        """
        Compute ``n_folds + 1`` quantile cut-points from ``_cv_idx``.

        Returns a list of length ``n_folds + 1`` where ``[0]`` is the
        minimum and ``[-1]`` is the maximum of the index column.
        These define ``n_folds`` equal-width temporal slices.
        """
        # We need n_folds + 1 boundaries to define n_folds slices.
        # Quantile probabilities: 0/n, 1/n, ..., n/n
        n = self._n_folds
        probabilities = [i / n for i in range(n + 1)]

        # approxQuantile is a single distributed action
        cuts = df_indexed.approxQuantile("_cv_idx", probabilities, 0.001)

        self._logger.debug(
            "CV cut-points (%d): %s", len(cuts), cuts
        )
        return cuts

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _chronological_folds(
        self,
        df: DataFrame,
        cuts: List[float],
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """
        Single-fold chronological split.

        Training = all rows before the last slice.
        Validation = last slice (the most recent ``val_ratio`` of data).

        For n_folds > 2, the last ``1/n_folds`` slice is the validation
        partition and everything earlier is training.  This gives one
        (train, val) pair.
        """
        # Validation boundary: everything from cuts[-2] onward is validation
        val_boundary = cuts[-2]

        train_fold = df.filter(F.col("_cv_idx") < val_boundary)
        val_fold   = df.filter(F.col("_cv_idx") >= val_boundary)

        self._logger.debug(
            "[chronological] val_boundary=%.0f", val_boundary
        )
        return [(train_fold, val_fold)]

    def _expanding_window_folds(
        self,
        df: DataFrame,
        cuts: List[float],
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """
        Expanding window: fold k trains on slices 0..k, validates on slice k+1.

        Produces ``n_folds - 1`` (train, val) pairs.  The first fold
        trains on just the first slice; the last fold trains on all but
        the final slice and validates on the final slice.
        """
        folds: List[Tuple[DataFrame, DataFrame]] = []
        # cuts = [c0, c1, c2, ..., cn]  →  slices [c0,c1), [c1,c2), ..., [cn-1,cn]
        for k in range(1, len(cuts) - 1):
            train_end = cuts[k]
            val_start = cuts[k]
            val_end   = cuts[k + 1]

            train_fold = df.filter(F.col("_cv_idx") < train_end)
            val_fold   = df.filter(
                (F.col("_cv_idx") >= val_start) & (F.col("_cv_idx") < val_end)
            )
            self._logger.debug(
                "[expanding_window] fold %d: train=[0, %.0f), val=[%.0f, %.0f)",
                k, train_end, val_start, val_end
            )
            folds.append((train_fold, val_fold))

        return folds

    def _rolling_window_folds(
        self,
        df: DataFrame,
        cuts: List[float],
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """
        Rolling window: a fixed-width window slides forward.

        Window width = ``n_folds - 1`` slices.  Fold k trains on slices
        [k, k + width - 1] and validates on slice [k + width].

        Produces ``1`` fold when n_folds == 2, more with larger n_folds.
        """
        width = max(1, self._n_folds - 1)
        folds: List[Tuple[DataFrame, DataFrame]] = []

        # Slide window through available slices
        max_k = len(cuts) - 1 - width
        for k in range(0, max_k):
            train_start = cuts[k]
            train_end   = cuts[k + width]
            val_start   = cuts[k + width]
            val_end     = cuts[k + width + 1] if (k + width + 1) < len(cuts) else float("inf")

            train_fold = df.filter(
                (F.col("_cv_idx") >= train_start) & (F.col("_cv_idx") < train_end)
            )
            val_fold = df.filter(
                (F.col("_cv_idx") >= val_start) & (F.col("_cv_idx") < val_end)
            )
            self._logger.debug(
                "[rolling_window] fold %d: train=[%.0f, %.0f), val=[%.0f, %.0f)",
                k, train_start, train_end, val_start, val_end
            )
            folds.append((train_fold, val_fold))

        # Guarantee at least one fold
        if not folds:
            self._logger.warning(
                "[rolling_window] Could not produce folds with width=%d. "
                "Falling back to chronological split.",
                width
            )
            return self._chronological_folds(df, cuts)

        return folds

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> str:
        """Active validation strategy name."""
        return self._strategy

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return self._n_folds
