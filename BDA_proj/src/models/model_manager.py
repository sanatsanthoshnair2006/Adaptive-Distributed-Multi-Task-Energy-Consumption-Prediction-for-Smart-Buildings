"""
Model Manager Module
=====================

Handles persistence of trained Spark MLlib models to HDFS.

Design Decisions:
    - Separate paths for selected vs. candidate models:  The winning model
      is written to a dedicated ``selected/`` subtree so downstream consumers
      can reliably load it without knowing the ranking result.
    - Version-based paths:  Every save call includes a version tag so
      multiple experiment runs do not overwrite each other.
    - Metadata sidecar:  A lightweight JSON metadata file is written
      alongside each saved model.  This keeps the model binary and its
      provenance together without coupling to an external database.
    - Overwrite guard:  Matches the pattern established by ``FeatureStore``
      and ``ConcreteTaskMixin``; explicit ``overwrite=True`` required.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from src.ingestion.hdfs_client import HDFSClient
from src.logging.logger_factory import LoggerFactory
from src.utils.hdfs_utils import hdfs_path_join


class ModelManagerError(Exception):
    """Raised when model save / load operations fail."""


class ModelManager:
    """
    Saves, loads, and tracks metadata for trained Spark MLlib models.

    Directory layout in HDFS::

        <models_base>/
            candidates/
                <task_name>/
                    <model_name>/
                        <version>/          ← Spark model binary
                        <version>.meta.json ← Metadata sidecar
            selected/
                <task_name>/
                    <version>/              ← Spark model binary (winner)
                    <version>.meta.json     ← Metadata sidecar

    Args:
        hdfs_client (HDFSClient): Connected HDFS client.
        hadoop_config (Dict[str, Any]): The ``"hadoop"`` section of the
            merged configuration dictionary.
    """

    def __init__(
        self,
        hdfs_client: HDFSClient,
        hadoop_config: Dict[str, Any],
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._hdfs_client = hdfs_client

        hdfs_cfg = hadoop_config.get("hdfs", {})
        self._models_base: str = hdfs_cfg.get("paths", {}).get(
            "models", "/user/energy_prediction/models"
        )

        self._candidates_base = hdfs_path_join(self._models_base, "candidates")
        self._selected_base = hdfs_path_join(self._models_base, "selected")

        # Ensure base directories exist
        self._hdfs_client.create_directory(self._candidates_base)
        self._hdfs_client.create_directory(self._selected_base)

        self._logger.info(
            "ModelManager initialised. Models base: %s", self._models_base
        )

    # ------------------------------------------------------------------
    # Save / Load candidate models
    # ------------------------------------------------------------------

    def save_candidate(
        self,
        model: PipelineModel,
        task_name: str,
        model_name: str,
        metadata: Dict[str, Any],
        version: str = "v1",
        overwrite: bool = False,
    ) -> str:
        """
        Save a trained candidate model to HDFS.

        Args:
            model (PipelineModel):  Fitted Spark ML pipeline.
            task_name (str):        Task identifier.
            model_name (str):       Model identifier.
            metadata (Dict):        Arbitrary metadata dict (metrics, config).
            version (str):          Version tag.  Defaults to ``"v1"``.
            overwrite (bool):       Allow overwrite.  Defaults to ``False``.

        Returns:
            str:  HDFS path where the model binary was saved.

        Raises:
            ModelManagerError:  On write failure or path conflict.
        """
        model_path = hdfs_path_join(
            self._candidates_base, task_name, model_name, version
        )
        return self._save_model(
            model=model,
            path=model_path,
            metadata=metadata,
            overwrite=overwrite,
            label=f"candidate/{task_name}/{model_name}",
        )

    def load_candidate(
        self,
        spark: SparkSession,
        task_name: str,
        model_name: str,
        version: str = "v1",
    ) -> PipelineModel:
        """
        Load a previously saved candidate model from HDFS.

        Args:
            spark (SparkSession): Active SparkSession.
            task_name (str):      Task identifier.
            model_name (str):     Model identifier.
            version (str):        Version tag.

        Returns:
            PipelineModel:  Loaded Spark ML pipeline.

        Raises:
            ModelManagerError:  If the path does not exist or load fails.
        """
        model_path = hdfs_path_join(
            self._candidates_base, task_name, model_name, version
        )
        return self._load_model(spark, model_path)

    # ------------------------------------------------------------------
    # Save / Load selected (winning) model
    # ------------------------------------------------------------------

    def save_selected(
        self,
        model: PipelineModel,
        task_name: str,
        metadata: Dict[str, Any],
        version: str = "v1",
        overwrite: bool = True,
    ) -> str:
        """
        Save the selected (winning) model to the ``selected/`` subtree.

        Args:
            model (PipelineModel):  Fitted Spark ML pipeline.
            task_name (str):        Task identifier.
            metadata (Dict):        Metadata dict (includes model_name, score).
            version (str):          Version tag.
            overwrite (bool):       Allow overwrite.  Defaults to ``True``
                                    because the selected model should always
                                    reflect the latest run.

        Returns:
            str:  HDFS path where the selected model was saved.
        """
        model_path = hdfs_path_join(self._selected_base, task_name, version)
        return self._save_model(
            model=model,
            path=model_path,
            metadata=metadata,
            overwrite=overwrite,
            label=f"selected/{task_name}",
        )

    def load_selected(
        self,
        spark: SparkSession,
        task_name: str,
        version: str = "v1",
    ) -> PipelineModel:
        """
        Load the selected model for a task.

        Args:
            spark (SparkSession): Active SparkSession.
            task_name (str):      Task identifier.
            version (str):        Version tag.

        Returns:
            PipelineModel:  Loaded Spark ML pipeline.
        """
        model_path = hdfs_path_join(self._selected_base, task_name, version)
        return self._load_model(spark, model_path)

    def get_selected_metadata(
        self,
        task_name: str,
        version: str = "v1",
    ) -> Optional[Dict[str, Any]]:
        """
        Read the JSON metadata sidecar for the selected model.

        Args:
            task_name (str): Task identifier.
            version (str):   Version tag.

        Returns:
            Dict or None if the metadata file does not exist.
        """
        meta_path = hdfs_path_join(
            self._selected_base, task_name, f"{version}.meta.json"
        )
        return self._read_metadata(meta_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_model(
        self,
        model: PipelineModel,
        path: str,
        metadata: Dict[str, Any],
        overwrite: bool,
        label: str,
    ) -> str:
        """Core save logic shared by candidate and selected paths."""
        self._logger.info("Saving model [%s] to: %s", label, path)

        if self._hdfs_client.exists(path):
            if overwrite:
                self._logger.warning(
                    "Path exists; overwriting: %s", path
                )
                self._hdfs_client.delete(path, recursive=True)
            else:
                raise ModelManagerError(
                    f"Model path already exists: {path}. "
                    f"Pass overwrite=True to replace."
                )

        try:
            spark_write_path = self._hdfs_client.spark_path(path)
            model.save(spark_write_path)
            self._logger.info("Model binary saved: %s", spark_write_path)
        except Exception as exc:
            raise ModelManagerError(
                f"Failed to save model to {path}: {exc}"
            ) from exc

        # Write JSON metadata sidecar
        meta_path = f"{path}.meta.json"
        metadata["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata["hdfs_path"] = path
        try:
            meta_bytes = json.dumps(metadata, indent=2, default=str).encode("utf-8")
            # Write via PyArrow FS (modern API uses open_output_stream)
            self._hdfs_client._ensure_connected()
            with self._hdfs_client._fs.open_output_stream(meta_path) as fh:
                fh.write(meta_bytes)
            self._logger.info("Metadata sidecar written: %s", meta_path)
        except Exception as exc:
            # Metadata failure is logged but non-fatal — the model binary is
            # already successfully saved.
            self._logger.warning(
                "Failed to write metadata sidecar (%s): %s", meta_path, exc
            )

        return path

    def _load_model(self, spark: SparkSession, path: str) -> PipelineModel:
        """Core load logic shared by candidate and selected paths."""
        if not self._hdfs_client.exists(path):
            raise ModelManagerError(
                f"Model path does not exist in HDFS: {path}"
            )
        self._logger.info("Loading model from: %s", path)
        try:
            spark_read_path = self._hdfs_client.spark_path(path)
            model = PipelineModel.load(spark_read_path)
            self._logger.info("Model loaded successfully from: %s", spark_read_path)
            return model
        except Exception as exc:
            raise ModelManagerError(
                f"Failed to load model from {path}: {exc}"
            ) from exc

    def _read_metadata(self, meta_path: str) -> Optional[Dict[str, Any]]:
        """Read and parse a JSON metadata sidecar file."""
        if not self._hdfs_client.exists(meta_path):
            return None
        try:
            self._hdfs_client._ensure_connected()
            with self._hdfs_client._fs.open_input_stream(meta_path) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except Exception as exc:
            self._logger.warning(
                "Failed to read metadata sidecar %s: %s", meta_path, exc
            )
            return None
