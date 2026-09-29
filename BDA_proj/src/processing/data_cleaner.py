"""
Data Cleaner Module
====================

Distributed data cleaning using Spark DataFrames.

Architectural Decisions:
    - Distributed Processing: Uses native Spark SQL functions to clean data in parallel.
    - No Dropping Rows (where possible): Time-series datasets require continuity.
      Forward-fill is used for missing values to preserve the temporal sequence.
    - Negative Power Fixes: Negative values (impossible in this context) are capped at 0.
    - Outlier Treatment: Uses the Interquartile Range (IQR) method. Outliers are capped
      to prevent extreme distortions in downstream lag features and rolling aggregations.
"""

from typing import Any, Dict, List
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.logging.logger_factory import LoggerFactory


class DataCleaner:
    """
    Cleans raw dataset by handling missing values, duplicates,
    negative values, and outliers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataCleaner.
        
        Args:
            config: Application configuration.
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        dataset_cfg = config.get("app", {}).get("dataset", {})
        self._expected_columns = dataset_cfg.get(
            "expected_columns", 
            ["Date", "Time", "Global_active_power", "Global_reactive_power", 
             "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        )

    def clean(self, df: DataFrame) -> DataFrame:
        """
        Execute the data cleaning pipeline.

        Args:
            df: Raw Spark DataFrame.

        Returns:
            Cleaned Spark DataFrame.
        """
        self._logger.info("=" * 60)
        self._logger.info("Starting distributed data cleaning...")
        self._logger.info("=" * 60)

        initial_count = df.count()

        df = self._cast_columns(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df)
        df = self._cap_negative_values(df)
        df = self._treat_outliers(df)

        final_count = df.count()
        self._logger.info(
            "Data cleaning completed. Rows: %d (initial) -> %d (final)",
            initial_count,
            final_count
        )
        
        return df

    def _cast_columns(self, df: DataFrame) -> DataFrame:
        """Cast columns to appropriate datatypes."""
        numeric_cols = [c for c in self._expected_columns if c not in ("Date", "Time")]
        for col_name in numeric_cols:
            if col_name in df.columns:
                # Cast string "?" to null during casting implicitly or explicitly
                df = df.withColumn(col_name, F.col(col_name).cast("double"))
        self._logger.debug("Cast numeric columns to double.")
        return df

    def _remove_duplicates(self, df: DataFrame) -> DataFrame:
        """Remove exact duplicate rows."""
        count_before = df.count()
        df = df.dropDuplicates()
        count_after = df.count()
        if count_before > count_after:
            self._logger.info("Removed %d duplicate rows.", count_before - count_after)
        return df

    def _handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values using forward fill (last observation carried forward).
        This requires an ordered window, so we combine Date and Time temporarily for ordering.
        """
        self._logger.info("Handling missing values via forward fill...")
        
        # Create a temporary monotonically increasing ID or timestamp for ordering
        # Since Date/Time are strings like '16/12/2006' and '17:24:00', we concat them
        df = df.withColumn(
            "_temp_dt", 
            F.to_timestamp(F.concat_ws(" ", F.col("Date"), F.col("Time")), "d/M/yyyy H:m:s")
        )
        
        window_spec = Window.orderBy("_temp_dt").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        
        numeric_cols = [c for c in self._expected_columns if c not in ("Date", "Time")]
        
        for col_name in numeric_cols:
            if col_name in df.columns:
                # forward fill
                df = df.withColumn(col_name, F.last(F.col(col_name), ignorenulls=True).over(window_spec))
        
        # Drop temp column
        df = df.drop("_temp_dt")
        
        # If there are leading nulls, fill them with 0
        df = df.fillna(0.0, subset=numeric_cols)
        
        return df

    def _cap_negative_values(self, df: DataFrame) -> DataFrame:
        """Cap physically impossible negative power values to 0."""
        numeric_cols = [c for c in self._expected_columns if c not in ("Date", "Time", "Voltage")]
        
        for col_name in numeric_cols:
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    F.when(F.col(col_name) < 0, 0.0).otherwise(F.col(col_name))
                )
        self._logger.debug("Capped negative power values at 0.")
        return df

    def _treat_outliers(self, df: DataFrame) -> DataFrame:
        """
        Treat outliers using the IQR method. Outliers are capped to the upper/lower bounds.
        Computed in a distributed way using approxQuantile.
        """
        self._logger.info("Treating outliers (IQR capping)...")
        numeric_cols = [c for c in self._expected_columns if c not in ("Date", "Time")]
        
        for col_name in numeric_cols:
            if col_name not in df.columns:
                continue
                
            # Calculate Q1 and Q3
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
            if len(quantiles) < 2:
                continue
                
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Since power cannot be negative (and already capped), we ensure lower_bound >= 0
            if "power" in col_name.lower() or "metering" in col_name.lower():
                lower_bound = max(0.0, lower_bound)
                
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name) > upper_bound, upper_bound)
                 .when(F.col(col_name) < lower_bound, lower_bound)
                 .otherwise(F.col(col_name))
            )
            
        return df
