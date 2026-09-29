"""
Explanation Manager Module
===========================

Provides Explainable AI (XAI) capabilities for the selected models.
To prevent massive overhead on the distributed cluster, this module
focuses on extracting native feature importances from tree-based
estimators (e.g., Random Forest, GBT) without calculating expensive
distributed SHAP values unless explicitly configured.

Generated artifacts are saved locally to be consumed by the
VisualizationManager.
"""

import json
import os
from typing import Any, Dict, List, Optional

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from src.logging.logger_factory import LoggerFactory
from src.models.model_manager import ModelManager
from src.ingestion.hdfs_client import HDFSClient


class ExplanationManagerError(Exception):
    pass


class ExplanationManager:
    """
    Extracts interpretability metrics (feature importance) from trained models.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        spark: SparkSession,
        hdfs_client: HDFSClient,
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._spark = spark
        
        # We need ModelManager to load the selected model
        hadoop_cfg = config.get("hadoop", {})
        self._model_manager = ModelManager(hdfs_client, hadoop_cfg)

        project_root = config.get("project_root", ".")
        xai_cfg = config.get("visualization", {}).get("explainability", {})
        self._output_dir = os.path.join(
            project_root, xai_cfg.get("output_dir", "results/explanations")
        )

    def explain_model(
        self,
        task_name: str,
        model_name: str,
        experiment_id: str,
    ) -> None:
        """
        Extract explanations for the selected model of a given task.
        
        Args:
            task_name: The task identifier.
            model_name: The selected model identifier.
            experiment_id: The experiment UUID/version string.
        """
        self._logger.info("[%s] Starting Explanation generation for model: %s", task_name, model_name)
        
        version = f"exp_{experiment_id[:8]}" if experiment_id else "v1"
        
        try:
            pipeline_model = self._model_manager.load_selected(
                self._spark, task_name, version=version
            )
        except Exception as exc:
            self._logger.error("[%s] Could not load selected model for explanation: %s", task_name, exc)
            return

        # The pipeline has [VectorAssembler, Estimator]
        if len(pipeline_model.stages) < 2:
            self._logger.warning("[%s] Pipeline does not have enough stages for feature extraction.", task_name)
            return
            
        assembler = pipeline_model.stages[-2]
        estimator_model = pipeline_model.stages[-1]
        
        feature_names = assembler.getInputCols()
        
        importances = self._extract_native_importance(estimator_model, feature_names)
        
        if importances:
            self._save_feature_importance(task_name, model_name, importances)
        else:
            self._logger.info("[%s] No native feature importance available for model type: %s", task_name, type(estimator_model).__name__)

    def _extract_native_importance(
        self,
        estimator_model: Any,
        feature_names: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract native feature importances if the model supports it.
        """
        if hasattr(estimator_model, "featureImportances"):
            importance_vector = estimator_model.featureImportances
            # Convert SparseVector/DenseVector to list
            imp_array = importance_vector.toArray()
            
            result = []
            for name, imp in zip(feature_names, imp_array):
                result.append({"feature": name, "importance": float(imp)})
                
            return sorted(result, key=lambda x: x["importance"], reverse=True)
            
        # For linear models we could extract coefficients, but skipping for brevity
        return None

    def _save_feature_importance(
        self,
        task_name: str,
        model_name: str,
        importances: List[Dict[str, Any]]
    ) -> None:
        """Save feature importance JSON to the local results directory."""
        target_dir = os.path.join(self._output_dir, task_name, model_name)
        os.makedirs(target_dir, exist_ok=True)
        
        out_path = os.path.join(target_dir, "feature_importance.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(importances, f, indent=2)
            self._logger.info("[%s] Saved feature importance to %s", task_name, out_path)
        except Exception as exc:
            self._logger.error("[%s] Failed to save feature importance: %s", task_name, exc)
