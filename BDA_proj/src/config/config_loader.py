"""
Configuration Loader Module
============================

Reads and merges all YAML configuration files into a unified
configuration dictionary. Provides typed access to configuration
sections.

Design Decisions:
    - Single Responsibility: Only loads and provides config data.
    - All paths are resolved relative to the project root.
    - Config is loaded once and shared across all modules.
"""

import os
from typing import Any, Dict, Optional

import yaml

from src.logging.logger_factory import LoggerFactory


class ConfigLoader:
    """
    Loads and provides access to all YAML configuration files.

    Attributes:
        project_root: Absolute path to the project root directory.
        _config: Merged configuration dictionary from all YAML files.
    """

    # Configuration file names expected in the config/ directory
    _CONFIG_FILES: Dict[str, str] = {
        "app": "app_config.yaml",
        "hadoop": "hadoop_config.yaml",
        "spark": "spark_config.yaml",
        "model": "model_config.yaml",
        "logging": "logging_config.yaml",
        "evaluation": "evaluation_config.yaml",
        "visualization": "visualization_config.yaml",
    }

    def __init__(self, project_root: Optional[str] = None) -> None:
        """
        Initialize ConfigLoader.

        Args:
            project_root: Absolute path to the project root directory.
                          If None, auto-detects based on this file's location.
        """
        if project_root is None:
            # Auto-detect: this file is at src/config/config_loader.py
            # Project root is two levels up
            self.project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
        else:
            self.project_root = os.path.abspath(project_root)

        self._config: Dict[str, Any] = {}
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        """Load all YAML configuration files into the merged config dict."""
        config_dir = os.path.join(self.project_root, "config")

        if not os.path.isdir(config_dir):
            raise FileNotFoundError(
                f"Configuration directory not found: {config_dir}"
            )

        for section_name, filename in self._CONFIG_FILES.items():
            filepath = os.path.join(config_dir, filename)
            if os.path.isfile(filepath):
                self._config[section_name] = self._load_yaml(filepath)
                self._logger.info(
                    "Loaded configuration: %s from %s", section_name, filename
                )
            else:
                self._logger.warning(
                    "Configuration file not found (skipping): %s", filepath
                )

        # Inject project_root into config for downstream use
        self._config["project_root"] = self.project_root

    def _load_yaml(self, filepath: str) -> Dict[str, Any]:
        """
        Load a single YAML file.

        Args:
            filepath: Absolute path to the YAML file.

        Returns:
            Parsed dictionary from the YAML file.

        Raises:
            yaml.YAMLError: If the file contains invalid YAML.
            FileNotFoundError: If the file does not exist.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                return data
        except yaml.YAMLError as e:
            self._logger.error("Failed to parse YAML file %s: %s", filepath, e)
            raise
        except FileNotFoundError:
            self._logger.error("YAML file not found: %s", filepath)
            raise

    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Retrieve a configuration value.

        Args:
            section: Top-level config section (e.g., 'app', 'hadoop').
            key: Optional nested key within the section.

        Returns:
            The configuration value, or the entire section if key is None.

        Raises:
            KeyError: If the section or key does not exist.
        """
        if section not in self._config:
            raise KeyError(f"Configuration section '{section}' not found.")

        if key is None:
            return self._config[section]

        section_data = self._config[section]
        if isinstance(section_data, dict) and key in section_data:
            return section_data[key]

        raise KeyError(
            f"Key '{key}' not found in configuration section '{section}'."
        )

    def get_nested(self, *keys: str) -> Any:
        """
        Retrieve a deeply nested configuration value.

        Args:
            *keys: Sequence of keys to traverse.
                   Example: get_nested('hadoop', 'hdfs', 'namenode_uri')

        Returns:
            The configuration value at the specified path.

        Raises:
            KeyError: If any key in the path does not exist.
        """
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise KeyError(
                    f"Configuration path not found: {' -> '.join(keys)}"
                )
        return current

    @property
    def config(self) -> Dict[str, Any]:
        """Return the complete merged configuration dictionary."""
        return self._config

    def get_project_root(self) -> str:
        """Return the absolute path to the project root."""
        return self.project_root

    def get_absolute_path(self, relative_path: str) -> str:
        """
        Convert a project-relative path to an absolute path.

        Args:
            relative_path: Path relative to the project root.

        Returns:
            Absolute path.
        """
        return os.path.join(self.project_root, relative_path)
