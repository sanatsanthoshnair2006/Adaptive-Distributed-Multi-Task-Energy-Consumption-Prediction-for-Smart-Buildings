"""
Data Loader Module
===================

Loads the raw dataset into Spark for distributed processing.
Handles the full flow: detect local file, upload to HDFS if
necessary, and read into a Spark DataFrame.

Design Decisions:
    - Never converts to Pandas — always returns Spark DataFrame.
    - Uploads to HDFS only if the file is not already present.
    - Uses Spark's native HDFS CSV reader for distributed I/O.
    - Delegates all HDFS operations to HDFSClient (Dependency Inversion).
"""

import os
from typing import Any, Dict, Optional

from pyspark.sql import DataFrame, SparkSession

from src.ingestion.hdfs_client import HDFSClient, HDFSOperationError
from src.logging.logger_factory import LoggerFactory
from src.utils.hdfs_utils import hdfs_path_join


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


class DataLoader:
    """
    Loads the household power consumption dataset into a Spark DataFrame.

    Workflow:
        1. Check if dataset exists locally.
        2. Upload to HDFS if not already present.
        3. Read from HDFS using Spark's distributed CSV reader.
        4. Return the Spark DataFrame.

    Attributes:
        _spark: Active SparkSession.
        _hdfs_client: HDFSClient for HDFS operations.
        _config: Application configuration dictionary.
        _dataset_config: Dataset-specific configuration.
    """

    def __init__(
        self,
        spark: SparkSession,
        hdfs_client: HDFSClient,
        app_config: Dict[str, Any],
        hadoop_config: Dict[str, Any],
        project_root: str,
    ) -> None:
        """
        Initialize DataLoader.

        Args:
            spark: Active SparkSession instance.
            hdfs_client: Connected HDFSClient instance.
            app_config: The 'app' section from the merged config.
            hadoop_config: The 'hadoop' section from the merged config.
            project_root: Absolute path to the project root directory.
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._spark = spark
        self._hdfs_client = hdfs_client
        self._project_root = project_root

        # Dataset configuration
        self._dataset_config = app_config.get("dataset", {})
        self._filename: str = self._dataset_config.get(
            "filename", "household_power_consumption.txt"
        )
        self._delimiter: str = self._dataset_config.get("delimiter", ";")
        self._encoding: str = self._dataset_config.get("encoding", "utf-8")
        self._missing_value_marker: str = self._dataset_config.get(
            "missing_value_marker", "?"
        )

        # Local paths
        self._local_data_dir: str = os.path.join(
            project_root,
            app_config.get("paths", {}).get("data_raw_local", "data/raw"),
        )

        # HDFS / local-mode paths
        hdfs_cfg = hadoop_config.get("hdfs", {})
        self._hdfs_raw_path: str = hdfs_cfg.get("paths", {}).get(
            "raw_data", "/user/energy_prediction/raw"
        )
        self._namenode_uri: str = hdfs_cfg.get(
            "namenode_uri", "hdfs://localhost:9000"
        )
        self._local_mode: bool = bool(hdfs_cfg.get("local_mode", False))
        _local_base_rel: str = hdfs_cfg.get("local_base_path", "hdfs_local")
        self._local_base: str = os.path.abspath(
            os.path.join(project_root, _local_base_rel)
        )

        self._logger.info(
            "DataLoader initialized: filename=%s, delimiter='%s', local_mode=%s",
            self._filename,
            self._delimiter,
            self._local_mode,
        )

    def load(self) -> DataFrame:
        """
        Execute the complete data loading pipeline.

        Steps:
            1. Locate the dataset on the local filesystem.
            2. Upload to HDFS if not already present.
            3. Read from HDFS into a Spark DataFrame.

        Returns:
            Spark DataFrame containing the raw dataset.

        Raises:
            DataLoadError: If the dataset cannot be found or loaded.
        """
        self._logger.info("=" * 60)
        self._logger.info("Starting data loading pipeline...")
        self._logger.info("=" * 60)

        # Step 1: Locate local dataset
        local_path = self._find_local_dataset()

        # Step 2: Upload to HDFS if necessary
        hdfs_file_path = self._upload_to_hdfs(local_path)

        # Step 3: Read from HDFS using Spark
        df = self._read_from_hdfs(hdfs_file_path)

        self._logger.info(
            "Data loading complete: %d rows, %d columns",
            df.count(),
            len(df.columns),
        )
        self._logger.info("Columns: %s", df.columns)

        return df

    def _find_local_dataset(self) -> str:
        """
        Locate the dataset file on the local filesystem.

        Searches in the following order:
            1. Configured local data directory (data/raw/)
            2. Project root directory
            3. Parent directory of the project

        Returns:
            Absolute path to the local dataset file.

        Raises:
            DataLoadError: If the dataset file is not found anywhere.
        """
        search_paths = [
            # Primary: configured data directory
            os.path.join(self._local_data_dir, self._filename),
            # Secondary: project root
            os.path.join(self._project_root, self._filename),
            # Tertiary: parent directory (common for downloaded datasets)
            os.path.join(
                os.path.dirname(self._project_root),
                "individual+household+electric+power+consumption",
                self._filename,
            ),
        ]

        for path in search_paths:
            if os.path.isfile(path):
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                self._logger.info(
                    "Dataset found locally: %s (%.2f MB)", path, file_size_mb
                )
                return path
            else:
                self._logger.debug("Dataset not at: %s", path)

        error_msg = (
            f"Dataset file '{self._filename}' not found in any of the "
            f"expected locations:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
        )
        self._logger.error(error_msg)
        raise DataLoadError(error_msg)

    def _upload_to_hdfs(self, local_path: str) -> str:
        """
        Upload the dataset to HDFS if it is not already there.

        Args:
            local_path: Absolute path to the local dataset file.

        Returns:
            HDFS path where the dataset is stored.

        Raises:
            DataLoadError: If upload fails.
        """
        hdfs_file_path = hdfs_path_join(self._hdfs_raw_path, self._filename)

        try:
            # Check if already in HDFS
            if self._hdfs_client.exists(hdfs_file_path):
                self._logger.info(
                    "Dataset already exists in HDFS: %s (skipping upload)",
                    hdfs_file_path,
                )
                return hdfs_file_path

            # Ensure HDFS directory exists
            self._hdfs_client.create_directory(self._hdfs_raw_path)

            # Upload
            self._logger.info(
                "Uploading dataset to HDFS: %s → %s",
                local_path,
                hdfs_file_path,
            )
            self._hdfs_client.upload_file(local_path, hdfs_file_path)
            self._logger.info(
                "Dataset successfully uploaded to HDFS: %s", hdfs_file_path
            )

            return hdfs_file_path

        except HDFSOperationError as e:
            error_msg = f"Failed to upload dataset to HDFS: {e}"
            self._logger.error(error_msg)
            raise DataLoadError(error_msg) from e

    def _read_from_hdfs(self, hdfs_file_path: str) -> DataFrame:
        """
        Read the dataset from HDFS into a Spark DataFrame.

        Uses Spark's native CSV reader for distributed I/O.
        Handles the semicolon delimiter and missing value marker.

        Args:
            hdfs_file_path: HDFS path to the dataset file.

        Returns:
            Spark DataFrame with the raw dataset.

        Raises:
            DataLoadError: If Spark cannot read the file.
        """
        try:
            if self._local_mode:
                # In local mode: Spark reads directly from the local copy
                # that was "uploaded" to the local hdfs_local directory.
                rel = hdfs_file_path.lstrip("/").lstrip("\\")
                local_file = os.path.join(self._local_base, rel)
                # Use forward slashes for Spark file:// URI on Windows
                spark_path = "file:///" + local_file.replace("\\", "/")
                self._logger.info(
                    "LOCAL MODE: Reading dataset via Spark from: %s", spark_path
                )
            else:
                full_hdfs_path = hdfs_file_path
                if not full_hdfs_path.startswith("hdfs://"):
                    namenode = self._hdfs_client.namenode_uri.rstrip("/")
                    full_hdfs_path = f"{namenode}/{hdfs_file_path.lstrip('/')}"
                spark_path = full_hdfs_path
                self._logger.info(
                    "Reading dataset from HDFS using Spark: %s", spark_path
                )

            df = (
                self._spark.read.format("csv")
                .option("header", "true")
                .option("delimiter", self._delimiter)
                .option("inferSchema", "false")   # Keep all columns as strings; DataCleaner handles casting
                .option("nullValue", self._missing_value_marker)
                .option("encoding", self._encoding)
                .option("mode", "PERMISSIVE")
                .option("columnNameOfCorruptRecord", "_corrupt_record")
                .load(spark_path)
            )

            self._logger.info(
                "Spark DataFrame created: schema=%s",
                df.schema.simpleString(),
            )

            # Log basic statistics
            row_count = df.count()
            col_count = len(df.columns)
            self._logger.info(
                "DataFrame dimensions: %d rows × %d columns",
                row_count,
                col_count,
            )

            return df

        except Exception as e:
            error_msg = f"Failed to read dataset: {e}"
            self._logger.error(error_msg)
            raise DataLoadError(error_msg) from e

    def upload_if_absent(self) -> str:
        """
        Public convenience method: upload dataset to HDFS only if absent.

        Returns:
            HDFS path of the dataset.

        Raises:
            DataLoadError: If the local file cannot be found or upload fails.
        """
        local_path = self._find_local_dataset()
        return self._upload_to_hdfs(local_path)
