"""
File Utilities Module
======================

Local filesystem helper functions for directory creation,
JSON/CSV persistence, and path management.
"""

import csv
import json
import os
from typing import Any, Dict, List

from src.logging.logger_factory import LoggerFactory


_logger = LoggerFactory.get_logger("utils.file_utils")


def ensure_local_dir(path: str) -> str:
    """
    Create a local directory if it does not exist.

    Args:
        path: Absolute or relative path to the directory.

    Returns:
        The absolute path to the created/existing directory.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path, exist_ok=True)
        _logger.debug("Created local directory: %s", abs_path)
    return abs_path


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data as a JSON file.

    Args:
        data: Serializable Python object.
        filepath: Absolute path for the output JSON file.
        indent: JSON indentation level.
    """
    ensure_local_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    _logger.debug("Saved JSON: %s", filepath)


def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.

    Args:
        filepath: Absolute path to the JSON file.

    Returns:
        Parsed Python object.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    data: List[Dict[str, Any]], filepath: str, fieldnames: List[str]
) -> None:
    """
    Save a list of dictionaries as a CSV file.

    Args:
        data: List of row dictionaries.
        filepath: Absolute path for the output CSV file.
        fieldnames: Column names for the CSV header.
    """
    ensure_local_dir(os.path.dirname(filepath))
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    _logger.debug("Saved CSV: %s", filepath)
