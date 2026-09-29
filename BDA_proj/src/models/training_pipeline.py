"""
Training Pipeline Module
=========================

Orchestrates distributed model training and hyperparameter optimization.

Responsibilities:
    - Receive a prediction task and prepared Spark dataset.
    - Retrieve candidate models from ModelFactory.
    - Perform hyperparameter optimization using HyperparameterTuner.
    - Train the optimized candidate models using ModelTrainer.
    - Generate performance metrics on the test split.
    - Pass results to the existing CompositeAdaptiveSelector for final ranking.

Design Decisions:
    - Time-aware splits are preserved (no random shuffles).
    - Data leakage is prevented by isolating the test holdout from tuning.
    - The pipeline handles all training, whereas the Selector handles
      cross-model ranking, persistence, and experiment registration.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.logging.logger_factory import LoggerFactory
from src.models.composite_adaptive_selector import CompositeAdaptiveSelector
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.models.model_factory import ModelFactory
from src.models.model_metrics import (
    CandidateResult,
    FailedCandidateResult,
    ModelMetrics,
)
from src.models.model_trainer import ModelTrainer
from src.tasks.base_task import BaseTask


class TrainingPipelineError(Exception):
    """Raised when the training pipeline encounters an unrecoverable error."""


class TrainingPipeline:
    """
    Executes hyperparameter tuning and final training for all candidates.

    Args:
        config (Dict[str, Any]): Merged application configuration.
        spark (SparkSession): Active Spark session.
        selector (CompositeAdaptiveSelector): Existing selector for ranking.
    """

    _FEATURES_COL = "features"
    _LABEL_COL = "label"
    _PREDICTION_COL = "prediction"

    def __init__(
        self,
        config: Dict[str, Any],
        spark: SparkSession,
        selector: CompositeAdaptiveSelector,
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._spark = spark
        self._selector = selector

        self._model_factory = ModelFactory(
            config=config,
            features_col=self._FEATURES_COL,
            label_col=self._LABEL_COL,
        )
        self._tuner = HyperparameterTuner(
            config=config,
            features_col=self._FEATURES_COL,
            label_col=self._LABEL_COL,
        )
        self._trainer = ModelTrainer(
            features_col=self._FEATURES_COL,
            label_col=self._LABEL_COL,
        )
        
        # We read test_ratio to mirror selector behavior
        pipeline_cfg = config.get("model", {}).get("pipeline", {})
        self._test_ratio = float(pipeline_cfg.get("test_ratio", 0.20))
        self._active_profile = config.get("model", {}).get("active_profile", "balanced")

    def run(
        self,
        tasks: List[BaseTask],
        features_df: DataFrame,
    ) -> Dict[str, CandidateResult]:
        """
        Execute tuning and training for all tasks, then hand off to selector.

        Args:
            tasks (List[BaseTask]): Instantiated tasks.
            features_df (DataFrame): Engineered features.

        Returns:
            Dict[str, CandidateResult]: Mapping of task_name -> winning result.
        """
        self._logger.info("=" * 70)
        self._logger.info("Training Pipeline (with Tuning) started.")
        self._logger.info("=" * 70)

        results: Dict[str, CandidateResult] = {}

        for task in tasks:
            task_name = task.get_task_name()
            self._logger.info(">>> Pipeline for Task: %s", task_name)
            
            try:
                winner = self._process_task(task, features_df)
                results[task_name] = winner
            except Exception as exc:
                self._logger.error(">>> Task [%s] pipeline failed: %s", task_name, exc, exc_info=True)
                raise
                
        return results

    def _process_task(
        self,
        task: BaseTask,
        features_df: DataFrame,
    ) -> CandidateResult:
        task_name = task.get_task_name()
        experiment_id = str(uuid.uuid4())

        self._logger.info("[%s] Preparing dataset...", task_name)
        prepared_df = task.prepare_dataset(features_df)
        prepared_df.cache()

        try:
            train_df, test_df = task.train_test_split(
                prepared_df, test_ratio=self._test_ratio
            )
            train_df.cache()
            test_df.cache()

            candidates = self._model_factory.get_enabled_candidates()
            
            raw_metrics_list: List[ModelMetrics] = []
            failed_list: List[FailedCandidateResult] = []
            trained_models: Dict[str, PipelineModel] = {}

            for model_name, estimator in candidates:
                try:
                    # 1. Tune Hyperparameters on training set
                    best_params, _ = self._tuner.tune(
                        task_name=task_name,
                        model_name=model_name,
                        base_estimator=estimator,
                        train_df=train_df,
                        feature_cols=task.get_feature_columns(),
                        target_col=task.get_target_column(),
                    )
                    
                    # 2. Apply best params to the estimator
                    # The tuner returns {str: value} dicts; estimator.copy()
                    # needs {Param: value} — convert the keys back here.
                    if best_params:
                        param_map = {}
                        for key, val in best_params.items():
                            try:
                                param_obj = estimator.getParam(key)
                                param_map[param_obj] = val
                            except Exception:
                                pass  # skip unrecognised params silently
                        tuned_estimator = estimator.copy(param_map) if param_map else estimator
                    else:
                        tuned_estimator = estimator
                    
                    # 3. Train final model on the full training set
                    fitted_model, metadata = self._trainer.train(
                        task_name=task_name,
                        model_name=model_name,
                        estimator=tuned_estimator,
                        train_df=train_df,
                        feature_cols=task.get_feature_columns(),
                        target_col=task.get_target_column(),
                    )
                    
                    # 4. Evaluate on test set
                    metrics = self._evaluate_test_set(
                        task=task,
                        model_name=model_name,
                        fitted_model=fitted_model,
                        test_df=test_df,
                        metadata=metadata,
                        experiment_id=experiment_id,
                    )
                    
                    raw_metrics_list.append(metrics)
                    trained_models[model_name] = fitted_model
                    
                except Exception as exc:
                    self._logger.error(
                        "[%s/%s] Candidate failed: %s", task_name, model_name, exc
                    )
                    failed_list.append(FailedCandidateResult(
                        task_name=task_name,
                        model_name=model_name,
                        error=str(exc),
                        experiment_id=experiment_id,
                    ))

            # Hand off to the existing selector's new extension method
            winner = self._selector.process_pre_trained(
                task=task,
                test_df=test_df,
                raw_metrics_list=raw_metrics_list,
                failed_list=failed_list,
                trained_models=trained_models,
                experiment_id=experiment_id,
            )
            
            return winner

        finally:
            prepared_df.unpersist()
            train_df.unpersist()
            test_df.unpersist()

    def _evaluate_test_set(
        self,
        task: BaseTask,
        model_name: str,
        fitted_model: PipelineModel,
        test_df: DataFrame,
        metadata: Dict[str, Any],
        experiment_id: str,
    ) -> ModelMetrics:
        """Evaluate the fitted model on the test split."""
        task_name = task.get_task_name()
        target_col = task.get_target_column()
        feature_cols = task.get_feature_columns()

        # Format test set
        test_labeled = test_df.withColumn(
            self._LABEL_COL, F.col(target_col).cast("double")
        ).dropna(subset=feature_cols + [self._LABEL_COL])

        # Predict
        t0_pred = time.perf_counter()
        predictions_df = fitted_model.transform(test_labeled)
        _ = predictions_df.count()  # Materialize
        prediction_time = time.perf_counter() - t0_pred

        # Evaluate
        evaluator = RegressionEvaluator(
            labelCol=self._LABEL_COL,
            predictionCol=self._PREDICTION_COL,
        )
        
        rmse = float(evaluator.evaluate(predictions_df, {evaluator.metricName: "rmse"}))
        mae = float(evaluator.evaluate(predictions_df, {evaluator.metricName: "mae"}))
        r2 = float(evaluator.evaluate(predictions_df, {evaluator.metricName: "r2"}))

        return ModelMetrics(
            task_name=task_name,
            model_name=model_name,
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=metadata.get("training_time", 0.0),
            prediction_time=prediction_time,
            memory_usage_mb=metadata.get("memory_usage_mb"),
            cpu_usage_pct=metadata.get("cpu_usage_pct"),
            experiment_id=experiment_id,
            selection_profile=self._active_profile,
        )
