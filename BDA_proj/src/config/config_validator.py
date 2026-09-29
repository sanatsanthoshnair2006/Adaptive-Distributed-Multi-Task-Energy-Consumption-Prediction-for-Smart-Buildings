"""
Configuration Validator Module
===============================

Validates that all required configuration keys are present and
have correct types before the pipeline starts. Fails fast to
prevent runtime errors deep in the pipeline.

Design Decisions:
    - Validates at startup, not at point-of-use.
    - Returns all validation errors at once (not one-at-a-time).
"""

import os
from typing import Any, Dict, List, Optional

from src.logging.logger_factory import LoggerFactory


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


class ConfigValidator:
    """
    Validates the merged configuration dictionary for required
    keys, types, and path existence.
    """

    def __init__(self) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._errors: List[str] = []

    def validate(self, config: Dict[str, Any], project_root: str) -> None:
        """
        Run all validation checks.

        Args:
            config: The merged configuration dictionary from ConfigLoader.
            project_root: Absolute path to the project root.

        Raises:
            ConfigValidationError: If any validation checks fail.
        """
        self._errors = []
        self._logger.info("Starting configuration validation...")

        self._validate_app_config(config)
        self._validate_hadoop_config(config)
        self._validate_spark_config(config)
        self._validate_logging_config(config)
        self._validate_evaluation_config(config)
        self._validate_visualization_config(config)
        self._validate_local_paths(config, project_root)

        if self._errors:
            self._logger.error(
                "Configuration validation failed with %d error(s).",
                len(self._errors),
            )
            raise ConfigValidationError(self._errors)

        self._logger.info("Configuration validation passed successfully.")

    def _validate_app_config(self, config: Dict[str, Any]) -> None:
        """Validate the application configuration section."""
        app = config.get("app")
        if app is None:
            self._errors.append("Missing configuration section: 'app'")
            return

        self._require_key(app, "project", "app")
        self._require_key(app, "dataset", "app")

        project = app.get("project", {})
        self._require_key(project, "name", "app.project")
        self._require_key(project, "execution_mode", "app.project")

        execution_mode = project.get("execution_mode", "")
        if execution_mode not in ("local", "cluster"):
            self._errors.append(
                f"app.project.execution_mode must be 'local' or 'cluster', "
                f"got '{execution_mode}'"
            )

        dataset = app.get("dataset", {})
        self._require_key(dataset, "filename", "app.dataset")
        self._require_key(dataset, "delimiter", "app.dataset")
        self._require_key(dataset, "expected_columns", "app.dataset")

        expected_cols = dataset.get("expected_columns", [])
        if not isinstance(expected_cols, list) or len(expected_cols) == 0:
            self._errors.append(
                "app.dataset.expected_columns must be a non-empty list."
            )

    def _validate_hadoop_config(self, config: Dict[str, Any]) -> None:
        """Validate the Hadoop configuration section."""
        hadoop = config.get("hadoop")
        if hadoop is None:
            self._errors.append("Missing configuration section: 'hadoop'")
            return

        hdfs = hadoop.get("hdfs")
        if hdfs is None:
            self._errors.append("Missing configuration key: hadoop.hdfs")
            return

        self._require_key(hdfs, "namenode_uri", "hadoop.hdfs")
        self._require_key(hdfs, "base_path", "hadoop.hdfs")
        self._require_key(hdfs, "paths", "hadoop.hdfs")

        paths = hdfs.get("paths", {})
        for required_path in ["raw_data", "processed_data", "features", "models"]:
            self._require_key(paths, required_path, "hadoop.hdfs.paths")

    def _validate_spark_config(self, config: Dict[str, Any]) -> None:
        """Validate the Spark configuration section."""
        spark = config.get("spark")
        if spark is None:
            self._errors.append("Missing configuration section: 'spark'")
            return

        spark_inner = spark.get("spark", spark)
        self._require_key(spark_inner, "app_name", "spark")
        self._require_key(spark_inner, "master", "spark")

    def _validate_logging_config(self, config: Dict[str, Any]) -> None:
        """Validate the logging configuration section."""
        logging_cfg = config.get("logging")
        if logging_cfg is None:
            self._errors.append("Missing configuration section: 'logging'")
            return

        logging_inner = logging_cfg.get("logging", logging_cfg)
        self._require_key(logging_inner, "level", "logging")

    def _validate_evaluation_config(self, config: Dict[str, Any]) -> None:
        """Validate the evaluation configuration section."""
        evaluation_cfg = config.get("evaluation")
        if evaluation_cfg is None:
            self._errors.append("Missing configuration section: 'evaluation'")
            return
            
        metrics = evaluation_cfg.get("metrics")
        if metrics is None:
            self._errors.append("Missing configuration key: evaluation.metrics")
        else:
            self._require_key(metrics, "compute", "evaluation.metrics")
            
        statistics = evaluation_cfg.get("statistics")
        if statistics is None:
            self._errors.append("Missing configuration key: evaluation.statistics")
        else:
            self._require_key(statistics, "alpha", "evaluation.statistics")
            self._require_key(statistics, "test_type", "evaluation.statistics")
            
        reporting = evaluation_cfg.get("reporting")
        if reporting is None:
            self._errors.append("Missing configuration key: evaluation.reporting")
        else:
            self._require_key(reporting, "output_dir", "evaluation.reporting")

    def _validate_visualization_config(self, config: Dict[str, Any]) -> None:
        """Validate the visualization configuration section (soft — skipped if absent)."""
        viz_cfg = config.get("visualization")
        if viz_cfg is None:
            # Visualization config is optional; just log a warning
            self._logger.warning(
                "'visualization' configuration section not found. Visualization features disabled."
            )
            return

        figures_cfg = viz_cfg.get("figures")
        if figures_cfg is None:
            self._errors.append("Missing configuration key: visualization.figures")
        else:
            self._require_key(figures_cfg, "output_dir", "visualization.figures")
            self._require_key(figures_cfg, "dpi", "visualization.figures")

    def _validate_local_paths(
        self, config: Dict[str, Any], project_root: str
    ) -> None:
        """Verify that critical local directories can be created."""
        app = config.get("app", {})
        paths = app.get("paths", {})

        for path_key, path_value in paths.items():
            full_path = os.path.join(project_root, path_value)
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.isdir(parent_dir):
                # Not an error — directories will be created at runtime
                self._logger.debug(
                    "Directory will be created at runtime: %s", full_path
                )

    def _require_key(
        self, data: Any, key: str, section: str
    ) -> Optional[Any]:
        """
        Check that a key exists in a dictionary.

        Args:
            data: The dictionary to check.
            key: The required key.
            section: Human-readable section name for error messages.
        """
        if not isinstance(data, dict):
            self._errors.append(
                f"Expected dict for section '{section}', got {type(data).__name__}"
            )
            return None

        if key not in data:
            self._errors.append(
                f"Missing required key '{key}' in section '{section}'."
            )
            return None

        return data[key]
