"""
Pipeline Validator Module
==========================

Pre-flight checks to ensure the pipeline can execute successfully.
"""

import os
from typing import Any, Dict, List

from src.logging.logger_factory import LoggerFactory


class PipelineValidationError(Exception):
    """Raised when pre-flight validation fails."""
    pass


class PipelineValidator:
    """
    Validates configuration, datasets, and environment before starting
    the heavy execution stages of the pipeline.
    """

    def __init__(self, config: Dict[str, Any], project_root: str) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._config = config
        self._project_root = project_root

    def validate_all(self) -> None:
        """
        Run all pre-flight checks.
        
        Raises:
            PipelineValidationError: If any critical check fails.
        """
        self._logger.info("=" * 60)
        self._logger.info("Running Pipeline Pre-flight Validation...")
        self._logger.info("=" * 60)

        errors: List[str] = []

        self._check_config(errors)
        self._check_dataset(errors)
        self._check_directories(errors)
        self._check_models(errors)

        if errors:
            self._logger.error("Pipeline validation failed with %d errors.", len(errors))
            for err in errors:
                self._logger.error("  - %s", err)
            raise PipelineValidationError("\n".join(errors))

        self._logger.info("Pipeline validation passed.")

    def _check_config(self, errors: List[str]) -> None:
        """Verify essential configuration sections exist."""
        required_sections = ["app", "hadoop", "spark", "model", "evaluation", "visualization"]
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required configuration section: '{section}'")
                
        # Validate selection weights if active_profile is present
        model_cfg = self._config.get("model", {})
        active_profile = model_cfg.get("active_profile")
        profiles = model_cfg.get("selection_profiles", {})
        
        if active_profile and active_profile in profiles:
            weights = profiles[active_profile]
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-4:
                errors.append(
                    f"Selection profile '{active_profile}' weights must sum to 1.0. "
                    f"Current sum is {total_weight}."
                )

    def _check_dataset(self, errors: List[str]) -> None:
        """Verify the raw dataset exists locally."""
        app_cfg = self._config.get("app", {})
        dataset_cfg = app_cfg.get("dataset", {})
        filename = dataset_cfg.get("filename", "household_power_consumption.txt")
        
        local_data_dir = os.path.join(
            self._project_root,
            app_cfg.get("paths", {}).get("data_raw_local", "data/raw")
        )
        
        search_paths = [
            os.path.join(local_data_dir, filename),
            os.path.join(self._project_root, filename),
        ]
        
        found = any(os.path.isfile(p) for p in search_paths)
        if not found:
            errors.append(f"Required dataset '{filename}' not found locally.")

    def _check_directories(self, errors: List[str]) -> None:
        """Verify local output directories can be created."""
        paths_cfg = self._config.get("app", {}).get("paths", {})
        for key, rel_path in paths_cfg.items():
            abs_path = os.path.join(self._project_root, rel_path)
            try:
                os.makedirs(abs_path, exist_ok=True)
                if not os.access(abs_path, os.W_OK):
                    errors.append(f"Directory not writable: {abs_path}")
            except Exception as exc:
                errors.append(f"Cannot create directory {abs_path}: {exc}")

    def _check_models(self, errors: List[str]) -> None:
        """Verify at least one model candidate is enabled."""
        candidates = self._config.get("model", {}).get("models", {})

        enabled_count = sum(
            1
            for candidate in candidates.values()
            if isinstance(candidate, dict)
            and candidate.get("enabled", False)
        )

        if enabled_count == 0:
            errors.append("No candidate models are enabled in configuration.")