"""
Pipeline Orchestration Module
=============================

Coordinates the 20 distinct stages of the experimental workflow.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.logging.logger_factory import LoggerFactory
from src.pipeline.pipeline_result import PipelineResult, StageResult
from src.pipeline.pipeline_stage import PipelineStage, PipelineStageError

# Module Imports
from src.processing.spark_session_manager import SparkSessionManager
from src.ingestion.hdfs_client import HDFSClient
from src.ingestion.data_loader import DataLoader
from src.ingestion.data_validator import DataValidator
from src.processing.data_cleaner import DataCleaner
from src.processing.data_transformer import DataTransformer
from src.energy.energy_estimator import EnergyEstimator
from src.features.feature_engineer import FeatureEngineer
from src.features.feature_store import FeatureStore
from src.tasks.task_registry import TaskRegistry
from src.models.composite_adaptive_selector import CompositeAdaptiveSelector
from src.models.training_pipeline import TrainingPipeline
from src.evaluation.evaluator import Evaluator
from src.visualization.visualization_manager import VisualizationManager


class Pipeline:
    """
    End-to-End Orchestrator for the Adaptive Energy Prediction project.
    """

    def __init__(
        self, 
        config: Dict[str, Any], 
        project_root: str, 
        cli_args: Any
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._project_root = project_root
        self._cli_args = cli_args
        
        self.result = PipelineResult(
            execution_id=str(uuid.uuid4()),
            start_time=datetime.now().isoformat(),
            configuration_profile=config.get("model", {}).get("active_profile"),
            environment={
                "python_version": sys.version,
                "force_recompute": cli_args.force,
                "resume_supported": cli_args.resume,
            }
        )

        # Component placeholders
        self._spark_manager: Optional[SparkSessionManager] = None
        self._hdfs_client: Optional[HDFSClient] = None
        self._tasks: List[Any] = []
        self._features_df: Any = None

    def run(self) -> PipelineResult:
        """Execute the pipeline sequentially."""
        t0 = time.perf_counter()
        self.result.status = "RUNNING"
        
        try:
            self._execute_stages()
            self.result.status = "SUCCESS"
        except PipelineStageError:
            self.result.status = "FAILED"
            self.result.overall_error = "A critical pipeline stage failed."
        except Exception as exc:
            self._logger.critical("Unhandled pipeline exception: %s", exc, exc_info=True)
            self.result.status = "FAILED"
            self.result.overall_error = str(exc)
        finally:
            self._cleanup_resources()
            
            t1 = time.perf_counter()
            self.result.total_duration_seconds = t1 - t0
            self.result.end_time = datetime.now().isoformat()
            
            self._finalize_experiment()
            
        return self.result

    def _execute_stages(self) -> None:
        """Define and execute all 20 stages."""
        
        def run_stage(id: int, name: str, critical: bool, skip: bool, action: Any, *args: Any, **kwargs: Any) -> Any:
            stage = PipelineStage(id, name, is_critical=critical, skip_if=skip)
            stage_result = stage.execute(action, *args, **kwargs)
            self.result.add_stage(stage_result)
            return stage_result.artifacts if hasattr(stage_result, "artifacts") else None

        # STAGE 1 & 2 are handled by main.py prior to pipeline init, but we log them.
        run_stage(1, "Configuration Initialization", True, False, lambda: {"status": "Already done in main"})
        run_stage(2, "Logging Initialization", True, False, lambda: {"status": "Already done in main"})
        
        # STAGE 3
        def init_experiment():
            exp_id = self._cli_args.experiment_name or str(uuid.uuid4())
            self.result.experiment_id = exp_id
            return {"experiment_id": exp_id}
        run_stage(3, "Experiment Initialization", True, False, init_experiment)
        
        # STAGE 4
        def init_resource_monitor():
            self._logger.info("Resource tracking is handled inline by CompositeAdaptiveSelector via psutil. No independent monitor initialized.")
        run_stage(4, "Resource Monitoring Initialization", False, False, init_resource_monitor)
        
        # STAGE 5
        def init_spark():
            self._spark_manager = SparkSessionManager(self._config)
            spark = self._spark_manager.get_session()
            return {"spark_version": spark.version}
        run_stage(5, "Spark Session Initialization", True, False, init_spark)
        spark = self._spark_manager.get_session()
        
        # Init HDFS (needed by many stages)
        # Inject project_root so HDFSClient can resolve local_base_path
        hadoop_cfg = dict(self._config.get("hadoop", {}))
        hadoop_cfg["_project_root"] = self._project_root
        self._hdfs_client = HDFSClient(hadoop_cfg)
        
        # STAGE 6
        def verify_hdfs():
            loader = DataLoader(spark, self._hdfs_client, self._config.get("app", {}), self._config.get("hadoop", {}), self._project_root)
            hdfs_path = loader.upload_if_absent()
            return {"hdfs_path": hdfs_path}
        run_stage(6, "HDFS Dataset Verification", True, False, verify_hdfs)
        
        # STAGE 7 & 8
        def load_and_validate():
            loader = DataLoader(spark, self._hdfs_client, self._config.get("app", {}), self._config.get("hadoop", {}), self._project_root)
            df = loader.load()
            validator = DataValidator(self._config.get("app", {}), self._project_root)
            report = validator.validate(df)
            return df, report
            
        def execute_load():
            df, report = load_and_validate()
            self._raw_df = df
            return report
        
        run_stage(7, "Dataset Validation", True, False, execute_load)
        # Stage 8 is inherently part of Stage 7's DataLoader call. We log it separately for tracking.
        run_stage(8, "Distributed Data Loading", True, False, lambda: {"rows": self._raw_df.count()})
        
        # STAGE 9
        def execute_clean():
            cleaner = DataCleaner(self._config)
            self._clean_df = cleaner.clean(self._raw_df)
        run_stage(9, "Distributed Data Cleaning", True, False, execute_clean)
        
        # STAGE 10
        def execute_transform():
            transformer = DataTransformer(self._config)
            self._transformed_df = transformer.transform(self._clean_df)
        run_stage(10, "Data Transformation", True, False, execute_transform)
        
        # STAGE 11
        def execute_energy():
            estimator = EnergyEstimator(self._hdfs_client, self._config.get("hadoop", {}))
            estimator.estimate(self._transformed_df)
        run_stage(11, "Energy Estimation", False, False, execute_energy)
        
        # STAGE 12
        def execute_features():
            engineer = FeatureEngineer(self._config)
            self._features_df = engineer.engineer(self._transformed_df)
        run_stage(12, "Feature Engineering", True, False, execute_features)
        
        # STAGE 13
        def execute_feature_store():
            store = FeatureStore(self._hdfs_client, self._config.get("hadoop", {}))
            version = f"exp_{self.result.experiment_id[:8]}" if self.result.experiment_id else "v1"
            store.save(self._features_df, version=version, overwrite=self._cli_args.force)
        run_stage(13, "Feature Store Persistence", True, False, execute_feature_store)
        
        # STAGE 14
        def init_tasks():
            factories = TaskRegistry.get_all_tasks()
            self._tasks = [factory(self._config, self._hdfs_client) for factory in factories.values()]
            return {"task_count": len(self._tasks)}
        run_stage(14, "Prediction Task Initialization", True, False, init_tasks)
        
        # STAGE 15
        def run_prediction_pipeline():
            from src.models.model_manager import ModelManager
            model_manager = ModelManager(self._hdfs_client, self._config.get("hadoop", {}))
            selector = CompositeAdaptiveSelector(self._config, spark, self._hdfs_client, model_manager)
            pipeline = TrainingPipeline(self._config, spark, selector)
            winners = pipeline.run(self._tasks, self._features_df)
            
            # Sub-stage: Explainable AI
            if not self._cli_args.skip_xai:
                from src.evaluation.explanation_manager import ExplanationManager
                xai = ExplanationManager(self._config, spark, self._hdfs_client)
                for task_name, winner in winners.items():
                    xai.explain_model(task_name, winner.metrics.model_name, self.result.experiment_id)
            
            return {t: w.metrics.model_name for t, w in winners.items()}
        run_stage(15, "Per-Task Training, Selection, and Predictions", True, False, run_prediction_pipeline)
        
        # STAGE 16
        def execute_eval():
            evaluator = Evaluator(self._config, spark, self._hdfs_client)
            evaluator.evaluate_tasks(self._tasks)
        run_stage(16, "Multi-Task Result Aggregation & Eval", True, False, execute_eval)
        
        # STAGE 17 & 18
        def execute_viz():
            manager = VisualizationManager(self._config)
            task_names = [t.get_task_name() for t in self._tasks]
            manager.generate_all(task_names)
        run_stage(17, "Visualization Generation", False, self._cli_args.skip_visualization, execute_viz)
        run_stage(18, "Research Report Generation", False, self._cli_args.skip_report, lambda: {"status": "Handled by VisualizationManager"})

    def _cleanup_resources(self) -> None:
        """STAGE 20: Safely clean up resources."""
        self._logger.info("=" * 60)
        self._logger.info("[%02d] STARTING STAGE: Resource Cleanup", 20)
        self._logger.info("=" * 60)
        try:
            if self._features_df is not None:
                self._features_df.unpersist()
            if self._spark_manager:
                self._spark_manager.stop()
            self.result.add_stage(StageResult(20, "Resource Cleanup", datetime.now().isoformat(), datetime.now().isoformat(), 0, "SUCCESS"))
        except Exception as exc:
            self._logger.warning("Resource cleanup encountered an error: %s", exc)

    def _finalize_experiment(self) -> None:
        """STAGE 19: Save pipeline execution results."""
        self._logger.info("=" * 60)
        self._logger.info("[%02d] STARTING STAGE: Experiment Finalization", 19)
        self._logger.info("=" * 60)
        
        results_dir = os.path.join(self._project_root, "results", "experiments")
        os.makedirs(results_dir, exist_ok=True)
        
        file_path = os.path.join(results_dir, f"pipeline_run_{self.result.execution_id[:8]}.json")
        self.result.save_json(file_path)
        self._logger.info("Pipeline execution summary saved to %s", file_path)
        self.result.add_stage(StageResult(19, "Experiment Finalization", datetime.now().isoformat(), datetime.now().isoformat(), 0, "SUCCESS"))
