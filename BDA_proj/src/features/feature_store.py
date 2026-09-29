"""
Feature Store Module
=====================

Handles persistence of engineered features into HDFS.

Architectural Decisions:
    - Format: Parquet, which provides columnar compression and is optimized
      for analytical queries and Spark reads.
    - Versioning: Provides versioned paths to avoid accidentally overwriting
      valuable processed features.
    - Overwrite Protection: Throws an error if attempting to write over an
      existing version unless explicitly permitted.
"""

from typing import Any, Dict
from pyspark.sql import DataFrame

from src.ingestion.hdfs_client import HDFSClient, HDFSOperationError
from src.logging.logger_factory import LoggerFactory
from src.utils.hdfs_utils import hdfs_path_join


class FeatureStoreError(Exception):
    """Raised when feature store operations fail."""
    pass


class FeatureStore:
    """
    Persists engineered feature DataFrames to HDFS as Parquet files,
    managing versions and preventing accidental overwrites.
    """

    def __init__(self, hdfs_client: HDFSClient, hadoop_config: Dict[str, Any]):
        """
        Initialize FeatureStore.
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._hdfs_client = hdfs_client
        
        hdfs_cfg = hadoop_config.get("hdfs", {})
        self._features_path = hdfs_cfg.get("paths", {}).get(
            "features", "/user/energy_prediction/features"
        )
        self._hdfs_client.create_directory(self._features_path)

    def save(
        self, 
        df: DataFrame, 
        version: str = "v1", 
        overwrite: bool = False
    ) -> str:
        """
        Save engineered features to HDFS.

        Args:
            df: Spark DataFrame with all features.
            version: Version identifier (e.g., 'v1', 'v20231015').
            overwrite: If True, overwrite existing version directory.

        Returns:
            HDFS path where features were saved.

        Raises:
            FeatureStoreError: If path exists and overwrite=False, or write fails.
        """
        output_path = hdfs_path_join(self._features_path, version)
        
        self._logger.info("=" * 60)
        self._logger.info("Starting feature store write to: %s", output_path)
        self._logger.info("=" * 60)
        
        if self._hdfs_client.exists(output_path):
            if overwrite:
                self._logger.warning("Version '%s' exists. Overwriting...", version)
            else:
                msg = f"Feature version '{version}' already exists at {output_path}."
                self._logger.error(msg)
                raise FeatureStoreError(msg)
                
        mode = "overwrite" if overwrite else "errorifexists"
        
        try:
            # Writing as Parquet.
            # Note: We do not partitionBy here because the dataset size is small
            # Partitioning would create too many tiny files.
            spark_write_path = self._hdfs_client.spark_path(output_path)
            df.write.mode(mode).parquet(spark_write_path)
            self._logger.info("Successfully saved features to %s", spark_write_path)
            return output_path
            
        except Exception as e:
            msg = f"Failed to save features to HDFS: {e}"
            self._logger.error(msg)
            raise FeatureStoreError(msg) from e

    def load(self, spark_session: Any, version: str = "v1") -> DataFrame:
        """
        Load a previously saved feature version from HDFS.

        Args:
            spark_session: Active SparkSession.
            version: Version identifier to load.

        Returns:
            Spark DataFrame.

        Raises:
            FeatureStoreError: If version does not exist or read fails.
        """
        input_path = hdfs_path_join(self._features_path, version)
        
        if not self._hdfs_client.exists(input_path):
            msg = f"Feature version '{version}' not found at {input_path}."
            self._logger.error(msg)
            raise FeatureStoreError(msg)
            
        self._logger.info("Loading features from %s", input_path)
        
        try:
            spark_read_path = self._hdfs_client.spark_path(input_path)
            df = spark_session.read.parquet(spark_read_path)
            self._logger.info("Successfully loaded features (Columns: %d)", len(df.columns))
            return df
        except Exception as e:
            msg = f"Failed to load features from HDFS: {e}"
            self._logger.error(msg)
            raise FeatureStoreError(msg) from e
