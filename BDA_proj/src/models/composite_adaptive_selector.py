"""
Composite Adaptive Selector Module
=====================================

Orchestrates the complete model selection workflow for all four
prediction tasks.

Workflow (per task)
--------------------
1.  Retrieve the prepared task dataset via the task framework.
2.  Perform the existing temporal train/test split.
3.  Assemble a Spark ML ``Pipeline`` (VectorAssembler → Estimator).
4.  For each candidate model:
    a. Measure memory baseline (psutil, optional).
    b. Fit the pipeline → record training time.
    c. Generate predictions → record prediction time.
    d. Compute RMSE, MAE, R² via Spark MLlib evaluators.
    e. Measure resource delta (memory, CPU).
    f. Store raw ``ModelMetrics``.
    g. Save trained pipeline as a candidate model via ``ModelManager``.
    h. Save predictions via ``BaseTask.save_predictions``.
5.  Normalise metrics via ``SelectionStrategy``.
6.  Compute Composite Performance Scores.
7.  Rank all candidates deterministically.
8.  Select the winning model.
9.  Save the winning model under the ``selected/`` subtree.
10. Persist all candidate metrics (Parquet).
11. Register the experiment via ``ExperimentManager``.

Error handling
--------------
One failing model does **not** abort the task-level selection.  The
failure is logged and that candidate is excluded from ranking.  If
*all* models fail for a task, ``AllModelsFailedError`` is raised.

Design Decisions:
    - psutil is optional:  Resource measurement degrades gracefully if the
      library is absent.  CPU and memory fields are set to ``None``.
    - No Pandas collection:  Metrics (RMSE, MAE, R²) are computed with Spark
      ML evaluators which remain distributed.
    - Cache strategy:  The prepared dataset (full feature set) is cached
      once per task.  Training and test sets are derived from this cache.
      Both are unpersisted before moving to the next task.
    - Spark ML Pipeline:  Using a ``Pipeline`` (VectorAssembler + Estimator)
      keeps the fitted model self-contained and loadable without needing
      to reassemble features manually.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ingestion.hdfs_client import HDFSClient
from src.logging.logger_factory import LoggerFactory
from src.models.model_factory import ModelFactory
from src.models.model_manager import ModelManager
from src.models.model_metrics import (
    CandidateResult,
    FailedCandidateResult,
    ModelMetrics,
)
from src.models.selection_strategy import SelectionStrategy
from src.tasks.base_task import BaseTask
from src.utils.hdfs_utils import hdfs_path_join


# ---------------------------------------------------------------------------
# Module-level exceptions
# ---------------------------------------------------------------------------

class AllModelsFailedError(Exception):
    """Raised when every candidate model fails for a given prediction task."""


class AdaptiveSelectorError(Exception):
    """General error raised by the adaptive selector."""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class CompositeAdaptiveSelector:
    """
    Orchestrates multi-task, multi-model training, evaluation,
    normalisation, and composite-score-based model selection.

    Args:
        config (Dict[str, Any]):   Full merged configuration dictionary.
        spark (SparkSession):      Active SparkSession.
        hdfs_client (HDFSClient):  Connected HDFS client.
        model_manager (ModelManager): Initialised model manager.
    """

    # Features and label column names used inside the Spark ML Pipeline
    _FEATURES_COL = "features"
    _LABEL_COL = "label"
    _PREDICTION_COL = "prediction"

    def __init__(
        self,
        config: Dict[str, Any],
        spark: SparkSession,
        hdfs_client: HDFSClient,
        model_manager: ModelManager,
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._spark = spark
        self._hdfs_client = hdfs_client
        self._model_manager = model_manager

        model_cfg = config.get("model", {})
        pipeline_cfg = model_cfg.get("pipeline", {})
        self._test_ratio: float = float(pipeline_cfg.get("test_ratio", 0.20))
        self._overwrite: bool = bool(pipeline_cfg.get("overwrite_results", True))
        self._random_seed: int = int(pipeline_cfg.get("random_seed", 42))
        self._tie_tolerance: float = float(pipeline_cfg.get("tie_tolerance", 1e-6))

        # Active selection profile
        self._active_profile: str = model_cfg.get("active_profile", "balanced")
        profiles = model_cfg.get("selection_profiles", {})
        if self._active_profile not in profiles:
            raise AdaptiveSelectorError(
                f"Active profile '{self._active_profile}' not found in "
                f"selection_profiles. Available: {list(profiles.keys())}"
            )
        self._profile_weights: Dict[str, float] = profiles[self._active_profile]
        self._logger.info(
            "Active selection profile: '%s' | Weights: %s",
            self._active_profile, self._profile_weights
        )

        # ExperimentManager (local results storage)
        self._experiment_manager = ExperimentManager(
            config=config,
            hdfs_client=hdfs_client,
        )

        # Model factory
        self._model_factory = ModelFactory(
            config=config,
            features_col=self._FEATURES_COL,
            label_col=self._LABEL_COL,
        )

        # psutil availability
        try:
            import psutil
            self._psutil = psutil
            self._logger.info("psutil available — resource monitoring enabled.")
        except ImportError:
            self._psutil = None
            self._logger.warning(
                "psutil not available — CPU/memory metrics will be None."
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        tasks: List[BaseTask],
        features_df: DataFrame,
    ) -> Dict[str, CandidateResult]:
        """
        Execute the full adaptive selection pipeline for all tasks.

        Args:
            tasks (List[BaseTask]):  Instantiated task objects (one per target).
            features_df (DataFrame): Full engineered feature DataFrame from the
                feature store — shared across all tasks to avoid re-reads.

        Returns:
            Dict[str, CandidateResult]:  Mapping of task name → selected
                model result.
        """
        self._logger.info("=" * 70)
        self._logger.info("Composite Adaptive Model Selection started.")
        self._logger.info(
            "Tasks: %s | Profile: %s",
            [t.get_task_name() for t in tasks],
            self._active_profile,
        )
        self._logger.info("=" * 70)

        selected: Dict[str, CandidateResult] = {}

        for task in tasks:
            task_name = task.get_task_name()
            self._logger.info("")
            self._logger.info(">>> Task: %s", task_name)

            try:
                winner = self._run_task(task, features_df)
                selected[task_name] = winner
                self._logger.info(
                    ">>> Task [%s] complete. Selected model: '%s' "
                    "(composite score: %.6f)",
                    task_name, winner.metrics.model_name, winner.composite_score
                )
            except AllModelsFailedError as exc:
                self._logger.error(
                    ">>> Task [%s] FAILED: All models failed. %s",
                    task_name, exc
                )
                raise
            except Exception as exc:
                self._logger.error(
                    ">>> Task [%s] unexpected error: %s", task_name, exc,
                    exc_info=True
                )
                raise

        self._logger.info("")
        self._logger.info("=" * 70)
        self._logger.info("Composite Adaptive Model Selection completed.")
        for tn, result in selected.items():
            self._logger.info(
                "  %-30s → %s (score=%.6f, RMSE=%.6f, R²=%.6f)",
                tn, result.metrics.model_name,
                result.composite_score, result.metrics.rmse, result.metrics.r2
            )
        self._logger.info("=" * 70)
        return selected

    # ------------------------------------------------------------------
    # Per-task orchestration
    # ------------------------------------------------------------------

    def _run_task(
        self,
        task: BaseTask,
        features_df: DataFrame,
    ) -> CandidateResult:
        """Run the full selection pipeline for one prediction task."""
        task_name = task.get_task_name()
        experiment_id = str(uuid.uuid4())

        self._logger.info("[%s] Preparing dataset...", task_name)
        prepared_df = task.prepare_dataset(features_df)

        # Cache the prepared dataset — reused for every candidate model
        prepared_df.cache()
        self._logger.info("[%s] Dataset cached.", task_name)

        try:
            self._logger.info(
                "[%s] Performing temporal train/test split (test_ratio=%.2f)...",
                task_name, self._test_ratio
            )
            train_df, test_df = task.train_test_split(
                prepared_df, test_ratio=self._test_ratio
            )
            # Cache splits — each model iterates over both
            train_df.cache()
            test_df.cache()

            candidates = self._model_factory.get_enabled_candidates()
            self._logger.info(
                "[%s] Candidate models: %s",
                task_name, [c[0] for c in candidates]
            )

            raw_metrics_list: List[ModelMetrics] = []
            failed_list: List[FailedCandidateResult] = []
            trained_models: Dict[str, PipelineModel] = {}

            for model_name, estimator in candidates:
                self._logger.info(
                    "[%s] ── Candidate: %s ──", task_name, model_name
                )
                try:
                    metrics, fitted_pipeline = self._train_and_evaluate(
                        task=task,
                        train_df=train_df,
                        test_df=test_df,
                        model_name=model_name,
                        estimator=estimator,
                        experiment_id=experiment_id,
                    )
                    raw_metrics_list.append(metrics)
                    trained_models[model_name] = fitted_pipeline
                    self._logger.info(
                        "[%s/%s] Metrics → RMSE=%.6f, MAE=%.6f, "
                        "R²=%.6f, train_t=%.3fs, pred_t=%.3fs",
                        task_name, model_name,
                        metrics.rmse, metrics.mae, metrics.r2,
                        metrics.training_time, metrics.prediction_time,
                    )
                except Exception as exc:
                    self._logger.error(
                        "[%s/%s] FAILED: %s",
                        task_name, model_name, exc,
                        exc_info=True
                    )
                    failed_list.append(FailedCandidateResult(
                        task_name=task_name,
                        model_name=model_name,
                        error=str(exc),
                        experiment_id=experiment_id,
                    ))

            if not raw_metrics_list:
                raise AllModelsFailedError(
                    f"[{task_name}] All {len(candidates)} candidate models "
                    f"failed.  Failures: "
                    f"{[f.model_name for f in failed_list]}"
                )

            # Rank candidates
            self._logger.info(
                "[%s] Normalising metrics and computing composite scores...",
                task_name
            )
            strategy = SelectionStrategy(
                profile_weights=self._profile_weights,
                tie_tolerance=self._tie_tolerance,
                profile_name=self._active_profile,
            )
            ranked: List[CandidateResult] = strategy.rank(raw_metrics_list)

            winner: CandidateResult = ranked[0]
            winning_model_name = winner.metrics.model_name
            self._logger.info(
                "[%s] Selected model: '%s'", task_name, winning_model_name
            )

            # Save all candidate models and their predictions
            version = f"exp_{experiment_id[:8]}"
            for result in ranked:
                mn = result.metrics.model_name
                if mn in trained_models:
                    self._save_candidate_artifacts(
                        task=task,
                        model_name=mn,
                        fitted_pipeline=trained_models[mn],
                        test_df=test_df,
                        result=result,
                        version=version,
                    )

            # Save winning model to the selected/ subtree
            self._logger.info(
                "[%s] Saving selected model '%s'...",
                task_name, winning_model_name
            )
            self._model_manager.save_selected(
                model=trained_models[winning_model_name],
                task_name=task_name,
                metadata={
                    "task_name": task_name,
                    "model_name": winning_model_name,
                    "composite_score": winner.composite_score,
                    "rmse": winner.metrics.rmse,
                    "mae": winner.metrics.mae,
                    "r2": winner.metrics.r2,
                    "selection_profile": self._active_profile,
                    "experiment_id": experiment_id,
                },
                version=version,
                overwrite=self._overwrite,
            )

            # Persist all candidate metrics to Parquet
            self._persist_metrics(
                task_name=task_name,
                ranked_results=ranked,
                failed_results=failed_list,
                experiment_id=experiment_id,
                version=version,
            )

            # Register experiment
            env_info = self._collect_env_info()
            self._experiment_manager.register(
                experiment_id=experiment_id,
                task_name=task_name,
                candidates=[r.metrics.model_name for r in ranked],
                ranked_results=ranked,
                failed_results=failed_list,
                selected_model=winning_model_name,
                selection_profile=self._active_profile,
                profile_weights=self._profile_weights,
                model_config=self._config.get("model", {}),
                env_info=env_info,
            )

            return winner

        finally:
            # Always unpersist — even if an exception was raised
            try:
                prepared_df.unpersist()
                train_df.unpersist()
                test_df.unpersist()
                self._logger.debug("[%s] DataFrames unpersisted.", task_name)
            except Exception:
                pass  # Unpersist failures are non-fatal

    def process_pre_trained(
        self,
        task: BaseTask,
        test_df: DataFrame,
        raw_metrics_list: List[ModelMetrics],
        failed_list: List[FailedCandidateResult],
        trained_models: Dict[str, PipelineModel],
        experiment_id: str,
    ) -> CandidateResult:
        """
        Receives trained models and metrics from the TrainingPipeline,
        ranks them, and persists the winner and artifacts.
        """
        task_name = task.get_task_name()

        if not raw_metrics_list:
            raise AllModelsFailedError(
                f"[{task_name}] All candidate models failed during tuning/training. "
                f"Failures: {[f.model_name for f in failed_list]}"
            )

        self._logger.info(
            "[%s] Normalising metrics and computing composite scores...",
            task_name
        )
        strategy = SelectionStrategy(
            profile_weights=self._profile_weights,
            tie_tolerance=self._tie_tolerance,
            profile_name=self._active_profile,
        )
        ranked: List[CandidateResult] = strategy.rank(raw_metrics_list)

        winner: CandidateResult = ranked[0]
        winning_model_name = winner.metrics.model_name
        self._logger.info(
            "[%s] Selected model: '%s'", task_name, winning_model_name
        )

        version = f"exp_{experiment_id[:8]}"
        for result in ranked:
            mn = result.metrics.model_name
            if mn in trained_models:
                self._save_candidate_artifacts(
                    task=task,
                    model_name=mn,
                    fitted_pipeline=trained_models[mn],
                    test_df=test_df,
                    result=result,
                    version=version,
                )

        self._logger.info(
            "[%s] Saving selected model '%s'...",
            task_name, winning_model_name
        )
        self._model_manager.save_selected(
            model=trained_models[winning_model_name],
            task_name=task_name,
            metadata={
                "task_name": task_name,
                "model_name": winning_model_name,
                "composite_score": winner.composite_score,
                "rmse": winner.metrics.rmse,
                "mae": winner.metrics.mae,
                "r2": winner.metrics.r2,
                "selection_profile": self._active_profile,
                "experiment_id": experiment_id,
            },
            version=version,
            overwrite=self._overwrite,
        )

        self._persist_metrics(
            task_name=task_name,
            ranked_results=ranked,
            failed_results=failed_list,
            experiment_id=experiment_id,
            version=version,
        )

        env_info = self._collect_env_info()
        self._experiment_manager.register(
            experiment_id=experiment_id,
            task_name=task_name,
            candidates=[r.metrics.model_name for r in ranked],
            ranked_results=ranked,
            failed_results=failed_list,
            selected_model=winning_model_name,
            selection_profile=self._active_profile,
            profile_weights=self._profile_weights,
            model_config=self._config.get("model", {}),
            env_info=env_info,
        )

        return winner

    # ------------------------------------------------------------------
    # Training and evaluation for one candidate
    # ------------------------------------------------------------------

    def _train_and_evaluate(
        self,
        task: BaseTask,
        train_df: DataFrame,
        test_df: DataFrame,
        model_name: str,
        estimator: Any,
        experiment_id: str,
    ) -> Tuple[ModelMetrics, PipelineModel]:
        """
        Build pipeline, train, and evaluate one candidate model.

        Returns:
            Tuple of (ModelMetrics, fitted PipelineModel).
        """
        task_name = task.get_task_name()
        feature_cols = task.get_feature_columns()
        target_col = task.get_target_column()

        # Build Spark ML Pipeline
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=self._FEATURES_COL,
            handleInvalid="skip",
        )
        pipeline = Pipeline(stages=[assembler, estimator])

        # Rename target column to the standard label name
        train_labeled = train_df.withColumn(self._LABEL_COL, F.col(target_col).cast("double"))
        test_labeled  = test_df.withColumn(self._LABEL_COL, F.col(target_col).cast("double"))

        # Drop any rows with null features or label
        train_labeled = train_labeled.dropna(
            subset=feature_cols + [self._LABEL_COL]
        )
        test_labeled = test_labeled.dropna(
            subset=feature_cols + [self._LABEL_COL]
        )

        # ---- Resource baseline ----------------------------------------
        mem_before_mb = self._get_memory_mb()

        # ---- Train ----------------------------------------------------
        self._logger.info("[%s/%s] Training started.", task_name, model_name)
        t0_train = time.perf_counter()
        fitted_pipeline: PipelineModel = pipeline.fit(train_labeled)
        training_time = time.perf_counter() - t0_train
        self._logger.info(
            "[%s/%s] Training completed in %.3fs.",
            task_name, model_name, training_time
        )

        # ---- Resource delta -------------------------------------------
        mem_after_mb = self._get_memory_mb()
        memory_delta = (
            (mem_after_mb - mem_before_mb)
            if (mem_before_mb is not None and mem_after_mb is not None)
            else None
        )
        cpu_usage = self._get_cpu_pct()

        # ---- Predict --------------------------------------------------
        self._logger.info("[%s/%s] Prediction started.", task_name, model_name)
        t0_pred = time.perf_counter()
        predictions_df = fitted_pipeline.transform(test_labeled)
        # Force an action to materialise the prediction plan
        _ = predictions_df.count()
        prediction_time = time.perf_counter() - t0_pred
        self._logger.info(
            "[%s/%s] Prediction completed in %.3fs.",
            task_name, model_name, prediction_time
        )

        # ---- Metrics --------------------------------------------------
        self._logger.info("[%s/%s] Computing evaluation metrics...", task_name, model_name)
        rmse = self._evaluate(predictions_df, metric="rmse")
        mae  = self._evaluate(predictions_df, metric="mae")
        r2   = self._evaluate(predictions_df, metric="r2")
        self._logger.info(
            "[%s/%s] Metrics calculated: RMSE=%.6f, MAE=%.6f, R²=%.6f",
            task_name, model_name, rmse, mae, r2
        )

        metrics = ModelMetrics(
            task_name=task_name,
            model_name=model_name,
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=training_time,
            prediction_time=prediction_time,
            memory_usage_mb=memory_delta,
            cpu_usage_pct=cpu_usage,
            experiment_id=experiment_id,
            selection_profile=self._active_profile,
        )

        return metrics, fitted_pipeline

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(self, predictions_df: DataFrame, metric: str) -> float:
        """Compute a regression metric using Spark MLlib evaluator."""
        evaluator = RegressionEvaluator(
            labelCol=self._LABEL_COL,
            predictionCol=self._PREDICTION_COL,
            metricName=metric,
        )
        return float(evaluator.evaluate(predictions_df))

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def _save_candidate_artifacts(
        self,
        task: BaseTask,
        model_name: str,
        fitted_pipeline: PipelineModel,
        test_df: DataFrame,
        result: CandidateResult,
        version: str,
    ) -> None:
        """Save candidate model binary and its predictions to HDFS."""
        task_name = task.get_task_name()
        target_col = task.get_target_column()
        feature_cols = task.get_feature_columns()

        # --- Candidate model binary ---
        try:
            self._model_manager.save_candidate(
                model=fitted_pipeline,
                task_name=task_name,
                model_name=model_name,
                metadata=result.to_flat_dict(),
                version=version,
                overwrite=self._overwrite,
            )
        except Exception as exc:
            self._logger.warning(
                "[%s/%s] Failed to save candidate model binary: %s",
                task_name, model_name, exc
            )

        # --- Predictions Parquet ---
        try:
            test_labeled = test_df.withColumn(
                self._LABEL_COL, F.col(target_col).cast("double")
            ).dropna(subset=feature_cols + [self._LABEL_COL])

            pred_df = fitted_pipeline.transform(test_labeled)

            # Build the standard prediction schema expected by BaseTask
            output_df = pred_df.select(
                F.col("Timestamp"),
                F.col(self._LABEL_COL).alias("Actual"),
                F.col(self._PREDICTION_COL).alias("Predicted"),
                F.lit(task_name).alias("Task"),
                F.lit(model_name).alias("Model"),
            )

            task.save_predictions(
                predictions_df=output_df,
                model_name=model_name,
                version=version,
                overwrite=self._overwrite,
            )
        except Exception as exc:
            self._logger.warning(
                "[%s/%s] Failed to save candidate predictions: %s",
                task_name, model_name, exc
            )

    def _persist_metrics(
        self,
        task_name: str,
        ranked_results: List[CandidateResult],
        failed_results: List[FailedCandidateResult],
        experiment_id: str,
        version: str,
    ) -> None:
        """Write all candidate metrics to Parquet via Spark."""
        self._logger.info(
            "[%s] Persisting candidate metrics...", task_name
        )

        rows = [r.to_flat_dict() for r in ranked_results]
        for f in failed_results:
            rows.append(f.to_flat_dict())

        try:
            metrics_df = self._spark.createDataFrame(rows)

            metrics_path = hdfs_path_join(
                self._hdfs_client.base_path,
                "results",
                "metrics",
                task_name,
                version,
            )
            metrics_df.write.mode("overwrite").parquet(metrics_path)
            self._logger.info(
                "[%s] Metrics persisted to: %s", task_name, metrics_path
            )
        except Exception as exc:
            self._logger.warning(
                "[%s] Failed to persist metrics: %s", task_name, exc
            )

    # ------------------------------------------------------------------
    # Resource monitoring (psutil, optional)
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

    # ------------------------------------------------------------------
    # Environment info
    # ------------------------------------------------------------------

    def _collect_env_info(self) -> Dict[str, Any]:
        """Collect reproducibility metadata about the runtime environment."""
        info: Dict[str, Any] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "selection_profile": self._active_profile,
            "random_seed": self._random_seed,
        }
        try:
            info["spark_version"] = self._spark.version
        except Exception:
            info["spark_version"] = "unknown"

        try:
            sc = self._spark.sparkContext
            hadoop_conf = sc._jsc.hadoopConfiguration()
            info["hadoop_version"] = hadoop_conf.get(
                "hadoop.version", "unknown"
            )
        except Exception:
            info["hadoop_version"] = "unknown"

        # Config hash for reproducibility tracking
        try:
            cfg_str = json.dumps(
                self._config.get("model", {}), sort_keys=True, default=str
            )
            info["config_hash"] = hashlib.md5(
                cfg_str.encode("utf-8"), usedforsecurity=False
            ).hexdigest()
        except Exception:
            info["config_hash"] = "unavailable"

        return info


# ---------------------------------------------------------------------------
# Experiment Manager
# ---------------------------------------------------------------------------

class ExperimentManager:
    """
    Records and persists experiment metadata for every selection run.

    Experiment records are written as JSON files to a local ``results/``
    directory (relative to project root) and also as a Parquet summary
    in HDFS when possible.

    Args:
        config (Dict[str, Any]):   Full merged configuration dictionary.
        hdfs_client (HDFSClient):  Connected HDFS client.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        hdfs_client: HDFSClient,
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._hdfs_client = hdfs_client
        self._config = config

        exp_cfg = config.get("model", {}).get("experiment", {})
        project_root = config.get("project_root", ".")
        results_rel = exp_cfg.get("results_dir", "results/experiments")
        self._results_dir = os.path.join(project_root, results_rel)
        os.makedirs(self._results_dir, exist_ok=True)

        self._logger.info(
            "ExperimentManager initialised. Results dir: %s", self._results_dir
        )

    def register(
        self,
        experiment_id: str,
        task_name: str,
        candidates: List[str],
        ranked_results: List[CandidateResult],
        failed_results: List[FailedCandidateResult],
        selected_model: str,
        selection_profile: str,
        profile_weights: Dict[str, float],
        model_config: Dict[str, Any],
        env_info: Dict[str, Any],
    ) -> str:
        """
        Persist experiment metadata to a local JSON file.

        Args:
            experiment_id (str):      UUID of this experiment run.
            task_name (str):          Prediction task identifier.
            candidates (List[str]):   Names of all evaluated models.
            ranked_results:           Ranked CandidateResult objects.
            failed_results:           FailedCandidateResult objects.
            selected_model (str):     Name of the winning model.
            selection_profile (str):  Active profile name.
            profile_weights:          Weight dict snapshot.
            model_config:             Model configuration snapshot.
            env_info:                 Runtime environment metadata.

        Returns:
            str:  Path to the written JSON file.
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        record = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "task_name": task_name,
            "candidates": candidates,
            "selected_model": selected_model,
            "selection_profile": selection_profile,
            "profile_weights": profile_weights,
            "raw_metrics": [r.metrics.to_dict() for r in ranked_results],
            "normalised_metrics": [
                {
                    "model_name": r.metrics.model_name,
                    "normalised_rmse": r.normalised_rmse,
                    "normalised_mae": r.normalised_mae,
                    "normalised_r2_penalty": r.normalised_r2_penalty,
                    "normalised_training_time": r.normalised_training_time,
                    "normalised_prediction_time": r.normalised_prediction_time,
                    "normalised_memory_mb": r.normalised_memory_mb,
                    "normalised_cpu_pct": r.normalised_cpu_pct,
                }
                for r in ranked_results
            ],
            "composite_scores": [
                {
                    "model_name": r.metrics.model_name,
                    "composite_score": r.composite_score,
                    "rank": r.rank,
                    "is_selected": r.is_selected,
                }
                for r in ranked_results
            ],
            "failed_candidates": [f.to_flat_dict() for f in failed_results],
            "model_config_snapshot": model_config,
            "environment": env_info,
        }

        filename = f"{task_name}_{experiment_id[:8]}_{timestamp[:10]}.json"
        filepath = os.path.join(self._results_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, default=str)
            self._logger.info(
                "[%s] Experiment record written: %s", task_name, filepath
            )
        except Exception as exc:
            self._logger.warning(
                "[%s] Failed to write experiment record: %s", task_name, exc
            )

        return filepath
