"""
Spark Session Manager Module
=============================

Singleton manager for Apache Spark sessions. Handles configuration,
initialization, and graceful shutdown.

Architectural Decisions:
    - Singleton Pattern: Ensures only one Spark context exists per JVM,
      preventing resource leaks and conflicts, particularly in cluster mode.
    - Adaptive Query Execution (AQE): Enabled to dynamically optimize shuffle
      partitions and join strategies during execution.
    - Timezone: Set to UTC to prevent implicit timezone conversions from
      corrupting timestamps.
"""

try:
    import py4j
    _py4j_available = True
except ImportError:
    _py4j_available = False

from typing import Any, Dict, Optional
from pyspark.sql import SparkSession

# ----------------------------------------------------------------------
# PySpark 4.2.0 Compatibility Patches for Local JVM Mode
# ----------------------------------------------------------------------
try:
    import pyspark.errors.utils as _err_utils
    _err_utils.is_debugging_enabled = lambda: False
    _err_utils._with_origin = lambda func: func
except Exception:
    pass

try:
    import pyspark.errors.exceptions.captured as _cap
    def _safe_get_query_context(self):
        return []

    for _cls_name in dir(_cap):
        _cls = getattr(_cap, _cls_name)
        if isinstance(_cls, type) and issubclass(_cls, _cap.CapturedException):
            _cls.getQueryContext = _safe_get_query_context
except Exception:
    pass


def _fixed_get_j_spark_session_class(jvm):
    res = getattr(jvm, "org.apache.spark.sql.classic.SparkSession")
    if _py4j_available and isinstance(res, py4j.java_gateway.JavaPackage):
        res = getattr(jvm, "org.apache.spark.sql.SparkSession")
    return res


def _fixed_get_j_spark_session_module(jvm):
    res = getattr(getattr(jvm, "org.apache.spark.sql.classic.SparkSession$"), "MODULE$")
    if _py4j_available and isinstance(res, py4j.java_gateway.JavaPackage):
        res = getattr(getattr(jvm, "org.apache.spark.sql.SparkSession$"), "MODULE$")
    return res


SparkSession._get_j_spark_session_class = staticmethod(_fixed_get_j_spark_session_class)
SparkSession._get_j_spark_session_module = staticmethod(_fixed_get_j_spark_session_module)

from src.logging.logger_factory import LoggerFactory


class SparkSessionManager:
    """
    Singleton manager for the SparkSession.

    Reads configuration from spark_config.yaml and sets up
    the Spark environment with optimized parameters.
    """

    _instance: Optional["SparkSessionManager"] = None
    _spark: Optional[SparkSession] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SparkSessionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, spark_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SparkSessionManager.
        
        Args:
            spark_config: The 'spark' section from the configuration.
                          Required on first initialization.
        """
        # Only initialize once
        if getattr(self, "_initialized", False):
            return

        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        
        if spark_config is None:
            raise ValueError("spark_config is required for first initialization")
            
        self._config = spark_config.get("spark", spark_config)
        self._initialized = True

    def get_session(self) -> SparkSession:
        """
        Get or create the SparkSession.

        Returns:
            The active SparkSession.
        """
        if self._spark is not None:
            return self._spark

        self._logger.info("Initializing SparkSession...")

        # -----------------------------------------------------------------
        # Windows: Spark requires HADOOP_HOME + winutils.exe to write files.
        # Auto-detect the standard installation path and inject it so the
        # pipeline works without manual shell environment setup.
        # -----------------------------------------------------------------
        import platform
        import os
        if platform.system() == "Windows":
            hadoop_home_candidates = [
                os.environ.get("HADOOP_HOME", ""),
                r"C:\hadoop",
                r"C:\winutils",
            ]
            for candidate in hadoop_home_candidates:
                if candidate and os.path.isfile(
                    os.path.join(candidate, "bin", "winutils.exe")
                ):
                    os.environ["HADOOP_HOME"] = candidate
                    os.environ.setdefault("hadoop.home.dir", candidate)
                    # py4j picks up java.library.path for native libs
                    hadoop_bin = os.path.join(candidate, "bin")
                    if hadoop_bin not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
                    self._logger.info("HADOOP_HOME auto-set to: %s", candidate)
                    break
            else:
                self._logger.warning(
                    "winutils.exe not found. Parquet writes may fail on Windows. "
                    "Install winutils: https://github.com/cdarlint/winutils"
                )

        app_name = self._config.get("app_name", "AdaptiveEnergyPrediction")
        master = self._config.get("master", "local[*]")
        default_fs = self._config.get("default_fs", "hdfs://localhost:9000")
        executor_memory = self._config.get("executor_memory", "4g")
        driver_memory = self._config.get("driver_memory", "2g")
        executor_cores = str(self._config.get("executor_cores", "2"))
        shuffle_partitions = str(self._config.get("shuffle_partitions", "200"))
        default_parallelism = str(self._config.get("default_parallelism", "8"))
        log_level = self._config.get("log_level", "WARN")

        builder = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.hadoop.fs.defaultFS", default_fs)
            .config("spark.executor.memory", executor_memory)
            .config("spark.driver.memory", driver_memory)
            .config("spark.executor.cores", executor_cores)
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .config("spark.default.parallelism", default_parallelism)
            # Enable Adaptive Query Execution for performance optimization
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # Force UTC timezone for consistent timestamp processing
            .config("spark.sql.session.timeZone", "UTC")
            # Optimize joins
            .config("spark.sql.autoBroadcastJoinThreshold", "10485760") # 10MB
        )

        self._spark = builder.getOrCreate()
        self._spark.sparkContext.setLogLevel(log_level)

        self._logger.info(
            "SparkSession initialized: master=%s, app_name=%s", master, app_name
        )
        return self._spark

    def stop(self) -> None:
        """
        Gracefully stop the SparkSession.
        """
        if self._spark is not None:
            self._logger.info("Stopping SparkSession...")
            self._spark.stop()
            self._spark = None
            self._logger.info("SparkSession stopped.")
