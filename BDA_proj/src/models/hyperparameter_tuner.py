"""
Hyperparameter Tuner Module
===========================

Handles grid search and cross-validation across time-series folds.

Design Decisions:
    - Avoids Spark's native CrossValidator/TrainValidationSplit because they
      cannot enforce strictly chronological, leak-free splits. We use our custom
      ``TimeSeriesCrossValidator`` instead.
    - Evaluates multiple hyperparameters and selects the best based on a
      configurable primary metric (e.g. RMSE).
    - Grid search is executed sequentially over parameter combinations to
      prevent cluster resource exhaustion.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml import Estimator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.param import Param
from pyspark.sql import DataFrame

from src.logging.logger_factory import LoggerFactory
from src.models.cross_validator import TimeSeriesCrossValidator


class HyperparameterTunerError(Exception):
    """Raised when tuning fails."""


class HyperparameterTuner:
    """
    Executes time-aware hyperparameter tuning for Spark ML estimators.

    Args:
        config (Dict[str, Any]): Merged application configuration.
        features_col (str): Column name for the feature vector.
        label_col (str): Column name for the target label.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        features_col: str = "features",
        label_col: str = "label",
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._features_col = features_col
        self._label_col = label_col

        tuning_cfg = config.get("model", {}).get("tuning", {})
        self._strategy = tuning_cfg.get("strategy", "grid_search")
        self._primary_metric = tuning_cfg.get("primary_metric", "rmse")
        self._max_combinations = tuning_cfg.get("max_combinations", 12)
        self._seed = tuning_cfg.get("random_seed", 42)

        cv_cfg = tuning_cfg.get("cross_validation", {})
        self._cv_strategy = cv_cfg.get("strategy", "chronological")
        self._n_folds = cv_cfg.get("n_folds", 3)
        self._val_ratio = cv_cfg.get("val_ratio", 0.15)

        self._search_spaces = tuning_cfg.get("search_spaces", {})

        self._logger.info(
            "HyperparameterTuner initialised: strategy=%s, metric=%s",
            self._strategy, self._primary_metric
        )

    def tune(
        self,
        task_name: str,
        model_name: str,
        base_estimator: Estimator,
        train_df: DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Executes hyperparameter tuning and returns the best parameters.

        Args:
            task_name (str): Identifier for logging.
            model_name (str): Identifier for the model family.
            base_estimator (Estimator): The unfitted base Spark ML estimator.
            train_df (DataFrame): The temporally ordered training dataset.
            feature_cols (list): Feature column names.
            target_col (str): Target column name.

        Returns:
            Tuple[Dict[str, Any], float]: A tuple containing the best parameter
                dictionary and its average validation score.
        """
        self._logger.info("[%s/%s] Starting hyperparameter tuning...", task_name, model_name)

        search_space = self._search_spaces.get(model_name)
        if not search_space:
            self._logger.info("[%s/%s] No search space defined. Skipping tuning.", task_name, model_name)
            return {}, 0.0

        # 1. Build Param Grid
        param_grid = self._build_param_grid(base_estimator, search_space)
        self._logger.info("[%s/%s] Generated %d parameter combinations.", task_name, model_name, len(param_grid))

        if not param_grid:
            return {}, 0.0

        # 2. Generate CV Folds
        cv = TimeSeriesCrossValidator(
            strategy=self._cv_strategy,
            n_folds=self._n_folds,
            val_ratio=self._val_ratio,
        )
        folds = cv.split(train_df)

        # We will use ModelTrainer logic internally, or just fit directly
        # to avoid circular overhead. A simple vector assembler + estimator
        # is enough for tuning.
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import functions as F

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=self._features_col,
            handleInvalid="skip",
        )
        
        evaluator = RegressionEvaluator(
            labelCol=self._label_col,
            predictionCol="prediction",
            metricName=self._primary_metric,
        )

        is_higher_better = self._primary_metric.lower() == "r2"
        best_score = float("-inf") if is_higher_better else float("inf")
        best_params = {}

        # 3. Iterate over param grid
        for i, param_map in enumerate(param_grid):
            self._logger.debug("[%s/%s] Evaluating combo %d/%d: %s", task_name, model_name, i + 1, len(param_grid), param_map)
            
            # Apply params to a clone of the base estimator
            cloned_estimator = base_estimator.copy(param_map)
            pipeline = Pipeline(stages=[assembler, cloned_estimator])
            
            fold_scores = []
            
            for fold_idx, (f_train, f_val) in enumerate(folds):
                f_train_labeled = f_train.withColumn(
                    self._label_col, F.col(target_col).cast("double")
                ).dropna(subset=feature_cols + [self._label_col])
                
                f_val_labeled = f_val.withColumn(
                    self._label_col, F.col(target_col).cast("double")
                ).dropna(subset=feature_cols + [self._label_col])

                try:
                    # Train
                    model = pipeline.fit(f_train_labeled)
                    # Predict
                    predictions = model.transform(f_val_labeled)
                    # Evaluate
                    score = evaluator.evaluate(predictions)
                    fold_scores.append(score)
                except Exception as exc:
                    self._logger.warning("[%s/%s] Combo %d failed on fold %d: %s", task_name, model_name, i + 1, fold_idx, exc)
                    break
            
            if len(fold_scores) == len(folds):
                avg_score = sum(fold_scores) / len(fold_scores)
                self._logger.debug("[%s/%s] Combo %d avg_score: %.6f", task_name, model_name, i + 1, avg_score)
                
                if is_higher_better:
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = param_map
                else:
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = param_map

        # Convert Spark Param objects back to string keys for returning
        best_params_dict = {param.name: value for param, value in best_params.items()}
        
        self._logger.info(
            "[%s/%s] Tuning complete. Best %s: %.6f | Best params: %s", 
            task_name, model_name, self._primary_metric, best_score, best_params_dict
        )
        
        return best_params_dict, best_score

    def _build_param_grid(
        self, estimator: Estimator, search_space: Dict[str, List[Any]]
    ) -> List[Dict[Param, Any]]:
        """
        Creates a list of parameter dictionaries from the search space.
        Uses grid search or random search.
        """
        keys = []
        valid_values = []
        
        # Match config keys to Spark Param objects
        for param_name, values in search_space.items():
            if not isinstance(values, list) or not values:
                continue
            
            # Spark params are often camelCase (e.g., maxDepth), but our config 
            # might be snake_case (e.g., max_depth).
            # Let's try direct match first, then camelCase.
            try:
                param = estimator.getParam(param_name)
            except AttributeError:
                # Convert snake_case to camelCase
                parts = param_name.split('_')
                camel_name = parts[0] + ''.join(word.capitalize() for word in parts[1:])
                try:
                    param = estimator.getParam(camel_name)
                except AttributeError:
                    self._logger.warning("Estimator does not have parameter '%s' or '%s'. Ignoring.", param_name, camel_name)
                    continue
            
            keys.append(param)
            valid_values.append(values)
            
        if not keys:
            return []
            
        # Cartesian product
        grid = []
        for combo in itertools.product(*valid_values):
            param_map = dict(zip(keys, combo))
            grid.append(param_map)
            
        if self._strategy == "random_search" or len(grid) > self._max_combinations:
            random.seed(self._seed)
            sample_size = min(len(grid), self._max_combinations)
            grid = random.sample(grid, sample_size)
            self._logger.debug("Sampled %d combinations from total %d", sample_size, len(valid_values[0]) ** len(keys) if valid_values else 0)

        return grid
