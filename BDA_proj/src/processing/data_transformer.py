"""
Data Transformer Module
========================

Distributed data transformation for temporal context.

Architectural Decisions:
    - Distributed Datetime Parsing: Uses Spark's `to_timestamp` instead of 
      pandas `to_datetime` to maintain distributed parallelism.
    - Window Functions for Resampling: Instead of moving to Pandas for `resample()`, 
      data is truncated to desired time intervals using `date_trunc` and aggregated 
      distributively if needed. However, this module mainly enriches row-level features.
    - Temporal Logic: Peak vs Off-peak indicators help models understand cyclical daily load.
"""

from typing import Any, Dict
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.logging.logger_factory import LoggerFactory


class DataTransformer:
    """
    Transforms clean dataset by extracting temporal context features
    (hour, weekday, season, peak indicators).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataTransformer.
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Execute the transformation pipeline.

        Args:
            df: Cleaned Spark DataFrame.

        Returns:
            Transformed Spark DataFrame with temporal features.
        """
        self._logger.info("=" * 60)
        self._logger.info("Starting distributed data transformation...")
        self._logger.info("=" * 60)

        df = self._generate_timestamp(df)
        df = self._extract_time_components(df)
        df = self._add_business_indicators(df)
        
        self._logger.info("Data transformation completed. Total columns: %d", len(df.columns))
        return df

    def _generate_timestamp(self, df: DataFrame) -> DataFrame:
        """Merge Date and Time columns into a unified Timestamp."""
        # Dataset format: Date = d/M/yyyy, Time = H:m:s
        df = df.withColumn(
            "Timestamp", 
            F.to_timestamp(F.concat_ws(" ", F.col("Date"), F.col("Time")), "d/M/yyyy H:m:s")
        )
        # Drop old columns
        df = df.drop("Date", "Time")
        self._logger.debug("Generated unified 'Timestamp' column.")
        return df

    def _extract_time_components(self, df: DataFrame) -> DataFrame:
        """Extract standard time components from Timestamp."""
        df = df.withColumn("Year", F.year("Timestamp"))
        df = df.withColumn("Hour", F.hour("Timestamp"))
        df = df.withColumn("Minute", F.minute("Timestamp"))
        df = df.withColumn("Day", F.dayofmonth("Timestamp"))
        df = df.withColumn("Weekday", F.dayofweek("Timestamp")) # 1=Sunday, 7=Saturday
        df = df.withColumn("DayOfWeek", F.col("Weekday"))
        df = df.withColumn("Month", F.month("Timestamp"))
        df = df.withColumn("Quarter", F.quarter("Timestamp"))
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall - Northern Hemisphere approximation)
        df = df.withColumn(
            "Season",
            F.when(F.col("Month").isin([12, 1, 2]), 1)
             .when(F.col("Month").isin([3, 4, 5]), 2)
             .when(F.col("Month").isin([6, 7, 8]), 3)
             .otherwise(4)
        )
        
        # Time since midnight in minutes (useful continuous time feature)
        df = df.withColumn("Minutes_Since_Midnight", F.col("Hour") * 60 + F.col("Minute"))
        
        return df

    def _add_business_indicators(self, df: DataFrame) -> DataFrame:
        """Add categorical business indicators (Weekend, Peak Hour)."""
        # Weekend: 1=Sunday, 7=Saturday
        df = df.withColumn(
            "Is_Weekend",
            F.when(F.col("Weekday").isin([1, 7]), 1).otherwise(0)
        )
        df = df.withColumn("IsWeekend", F.col("Is_Weekend"))
        
        # Peak Hour Indicator: assume 18:00 (6 PM) to 22:00 (10 PM) is peak residential load
        df = df.withColumn(
            "Is_Peak_Hour",
            F.when((F.col("Hour") >= 18) & (F.col("Hour") <= 22), 1).otherwise(0)
        )
        
        return df

