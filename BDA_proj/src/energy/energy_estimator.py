"""
Energy Estimator Module
========================

Estimates energy consumption from power measurements over time windows.

Architectural Decisions:
    - Distributed Aggregation: Uses Spark SQL window grouping (`window`) to
      aggregate minute-level power measurements into hourly, daily, weekly, 
      and monthly energy estimates.
    - Physics Integration: Energy (kWh) = Power (kW) * Time (hours). Since
      measurements are per minute, each row represents 1/60th of an hour.
      Therefore, Energy = SUM(Power) / 60 for the time window.
    - Result Persistence: Saves estimates directly to HDFS to serve as
      historical analytics and potential cross-validation baselines, independently
      of the main ML feature pipeline.
"""

import os
from typing import Any, Dict
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.ingestion.hdfs_client import HDFSClient
from src.logging.logger_factory import LoggerFactory
from src.utils.hdfs_utils import hdfs_path_join


class EnergyEstimator:
    """
    Computes hourly, daily, weekly, and monthly energy consumption.
    """

    def __init__(self, hdfs_client: HDFSClient, hadoop_config: Dict[str, Any]):
        """
        Initialize EnergyEstimator.
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._hdfs_client = hdfs_client
        
        hdfs_cfg = hadoop_config.get("hdfs", {})
        self._output_base = hdfs_cfg.get("paths", {}).get(
            "energy_estimates", "/user/energy_prediction/energy_estimates"
        )
        self._hdfs_client.create_directory(self._output_base)

    def estimate(self, df: DataFrame) -> None:
        """
        Estimate and save energy consumption at multiple granularities.
        
        Note: This is a side-effect pipeline branch. It takes the main
        DataFrame, computes aggregates, and saves them to HDFS.

        Args:
            df: Transformed Spark DataFrame with 'Timestamp' and power columns.
        """
        self._logger.info("=" * 60)
        self._logger.info("Starting distributed energy estimation...")
        self._logger.info("=" * 60)

        # We estimate energy for the main power columns
        power_cols = ["Global_active_power", "Global_reactive_power"]
        
        # 1. Hourly Energy
        self._compute_and_save(df, "1 hour", power_cols, "hourly")
        
        # 2. Daily Energy
        self._compute_and_save(df, "1 day", power_cols, "daily")
        
        # 3. Weekly Energy
        self._compute_and_save(df, "1 week", power_cols, "weekly")
        
        # 4. Monthly Energy
        # Since '1 month' can be tricky with exact intervals in some Spark versions,
        # we group by year and month extracted from Timestamp.
        self._compute_monthly_and_save(df, power_cols)
        
        # 5. Seasonal Energy
        self._compute_seasonal_and_save(df, power_cols)
        
        self._logger.info("Energy estimation completed.")

    def _compute_and_save(self, df: DataFrame, window_duration: str, cols: list, name: str) -> None:
        """Group by tumbling window, integrate power, and save to Parquet."""
        self._logger.info("Computing %s energy estimates...", name)
        
        # Group by tumbling time window
        grouped = df.groupBy(F.window(F.col("Timestamp"), window_duration))
        
        # Energy (kWh) = Sum(Power_kW) * (1 min / 60 mins)
        # Carbon Emission (kgCO2) = Energy (kWh) * Emission Factor (e.g., 0.5 kgCO2/kWh)
        CARBON_EMISSION_FACTOR = 0.5
        
        agg_exprs = []
        for c in cols:
            energy_expr = (F.sum(F.col(c)) / 60.0)
            agg_exprs.append(energy_expr.alias(f"Energy_{c}"))
            if c == "Global_active_power":
                agg_exprs.append((energy_expr * CARBON_EMISSION_FACTOR).alias("Carbon_Emission_kgCO2"))
        
        result_df = grouped.agg(*agg_exprs)
        
        # Flatten window struct for saving
        select_cols = [
            F.col("window.start").alias("Window_Start"),
            F.col("window.end").alias("Window_End")
        ]
        for c in cols:
            select_cols.append(f"Energy_{c}")
        if "Global_active_power" in cols:
            select_cols.append("Carbon_Emission_kgCO2")
            
        result_df = result_df.select(*select_cols).orderBy("Window_Start")
        
        output_path = hdfs_path_join(self._output_base, name)
        
        # Write to storage (using spark_path for local mode compatibility)
        spark_write_path = self._hdfs_client.spark_path(output_path)
        result_df.write.mode("overwrite").parquet(spark_write_path)
        self._logger.info("Saved %s estimates to %s", name, spark_write_path)

    def _compute_monthly_and_save(self, df: DataFrame, cols: list) -> None:
        """Group by Year and Month for monthly estimates."""
        self._logger.info("Computing monthly energy estimates...")
        
        grouped = df.groupBy(
            F.year("Timestamp").alias("Year"),
            F.month("Timestamp").alias("Month")
        )
        
        CARBON_EMISSION_FACTOR = 0.5
        
        agg_exprs = []
        for c in cols:
            energy_expr = (F.sum(F.col(c)) / 60.0)
            agg_exprs.append(energy_expr.alias(f"Energy_{c}"))
            if c == "Global_active_power":
                agg_exprs.append((energy_expr * CARBON_EMISSION_FACTOR).alias("Carbon_Emission_kgCO2"))
        
        result_df = grouped.agg(*agg_exprs).orderBy("Year", "Month")
        
        output_path = hdfs_path_join(self._output_base, "monthly")
        spark_write_path = self._hdfs_client.spark_path(output_path)
        result_df.write.mode("overwrite").parquet(spark_write_path)
        self._logger.info("Saved monthly estimates to %s", spark_write_path)

    def _compute_seasonal_and_save(self, df: DataFrame, cols: list) -> None:
        """Group by Year and Season for seasonal estimates."""
        self._logger.info("Computing seasonal energy estimates...")
        
        # Define seasons based on months
        # Winter: Dec (12), Jan (1), Feb (2)
        # Spring: Mar (3), Apr (4), May (5)
        # Summer: Jun (6), Jul (7), Aug (8)
        # Fall: Sep (9), Oct (10), Nov (11)
        
        month_col = F.month("Timestamp")
        season_col = F.when(month_col.isin(12, 1, 2), "Winter") \
                      .when(month_col.isin(3, 4, 5), "Spring") \
                      .when(month_col.isin(6, 7, 8), "Summer") \
                      .otherwise("Fall")
        
        grouped = df.groupBy(
            F.year("Timestamp").alias("Year"),
            season_col.alias("Season")
        )
        
        CARBON_EMISSION_FACTOR = 0.5
        
        agg_exprs = []
        for c in cols:
            energy_expr = (F.sum(F.col(c)) / 60.0)
            agg_exprs.append(energy_expr.alias(f"Energy_{c}"))
            if c == "Global_active_power":
                agg_exprs.append((energy_expr * CARBON_EMISSION_FACTOR).alias("Carbon_Emission_kgCO2"))
        
        result_df = grouped.agg(*agg_exprs).orderBy("Year", "Season")
        
        output_path = hdfs_path_join(self._output_base, "seasonal")
        spark_write_path = self._hdfs_client.spark_path(output_path)
        result_df.write.mode("overwrite").parquet(spark_write_path)
        self._logger.info("Saved seasonal estimates to %s", spark_write_path)
