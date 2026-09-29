"""
Evaluation Report Module
========================

Handles the serialization and persistence of evaluation outcomes.

Design Decisions:
    - Independent from metrics computation to ensure single responsibility.
    - Generates reports in JSON, CSV, and Parquet formats based on config.
    - Connects directly to HDFS for persistence (or local FS via PyArrow).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List
import pandas as pd

from src.logging.logger_factory import LoggerFactory
from src.ingestion.hdfs_client import HDFSClient
from src.utils.hdfs_utils import hdfs_path_join


class EvaluationReport:
    """
    Serializes and persists evaluation reports in multiple formats.

    Args:
        config (Dict[str, Any]): Merged application configuration.
        hdfs_client (HDFSClient): Configured HDFS client.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        hdfs_client: HDFSClient,
    ) -> None:
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._hdfs_client = hdfs_client
        
        eval_cfg = config.get("evaluation", {})
        reporting_cfg = eval_cfg.get("reporting", {})
        
        self._output_dir = reporting_cfg.get("output_dir", "results/evaluation")
        self._formats = reporting_cfg.get("formats", ["json", "csv"])

        # Create base directory if it doesn't exist
        if not self._hdfs_client.exists(self._output_dir):
            self._hdfs_client.create_directory(self._output_dir)

    def save_task_report(
        self,
        task_name: str,
        experiment_id: str,
        report_data: Dict[str, Any]
    ) -> None:
        """
        Saves a comprehensive JSON report for a single task.

        Args:
            task_name: Task identifier.
            experiment_id: Experiment ID to link to the models.
            report_data: Structured dictionary containing metrics, stats, and analysis.
        """
        task_dir = hdfs_path_join(self._output_dir, task_name)
        self._hdfs_client.create_directory(task_dir)

        file_path = hdfs_path_join(task_dir, f"eval_report_{experiment_id[:8]}.json")
        
        self._logger.info("[%s] Saving JSON evaluation report to %s", task_name, file_path)
        
        try:
            self._hdfs_client._ensure_connected()
            report_bytes = json.dumps(report_data, indent=2, default=str).encode("utf-8")
            with self._hdfs_client._fs.open(file_path, "wb") as fh:
                fh.write(report_bytes)
        except Exception as exc:
            self._logger.error("[%s] Failed to save JSON report: %s", task_name, exc)

    def save_flat_comparisons(
        self,
        task_name: str,
        experiment_id: str,
        flat_data: List[Dict[str, Any]],
        prefix: str
    ) -> None:
        """
        Saves tabular data (like accuracy-efficiency profiles or stat tests)
        to CSV or Parquet for easy downstream analysis.

        Args:
            task_name: Task identifier.
            experiment_id: Experiment ID.
            flat_data: List of dictionaries representing rows.
            prefix: Filename prefix (e.g., "acc_eff", "stats").
        """
        if not flat_data:
            return

        task_dir = hdfs_path_join(self._output_dir, task_name)
        self._hdfs_client.create_directory(task_dir)
        
        df = pd.DataFrame(flat_data)
        
        for fmt in self._formats:
            fmt = fmt.lower()
            if fmt == "json":
                continue # JSON is handled by the comprehensive report
                
            file_path = hdfs_path_join(task_dir, f"{prefix}_{experiment_id[:8]}.{fmt}")
            self._logger.debug("[%s] Saving %s to %s", task_name, fmt.upper(), file_path)
            
            try:
                self._hdfs_client._ensure_connected()
                with self._hdfs_client._fs.open(file_path, "wb") as fh:
                    if fmt == "csv":
                        df.to_csv(fh, index=False)
                    elif fmt == "parquet":
                        df.to_parquet(fh, index=False)
                    else:
                        self._logger.warning("[%s] Unsupported format: %s", task_name, fmt)
            except Exception as exc:
                self._logger.error("[%s] Failed to save %s: %s", task_name, file_path, exc)
