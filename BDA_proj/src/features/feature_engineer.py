"""
Feature Engineer Module
========================

Generates temporal lags, rolling statistics, and domain-specific
electrical features.

Architectural Decisions:
    - Spark Windowing without Partitions: Since the dataset tracks a single 
      household, there's no natural partition key (like user_id or device_id). 
      Applying a Window ordered by Timestamp without a partitionBy clause moves 
      all data to a single executor, which limits parallelism. However, for 2 million 
      rows (~130MB), this easily fits in a single node's memory. This is an intentional
      trade-off to compute exact lag/rolling features for a single time series.
    - Mathematical Features: Cyclical temporal features (sin/cos) and electrical
      ratios are computed distributively across partitions before the windowing step.
    - Caching: The dataframe can be cached before heavy windowing, but that's
      managed by the orchestrating pipeline, not here.
"""

import math
from typing import Any, Dict
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.logging.logger_factory import LoggerFactory


class FeatureEngineer:
    """
    Engineers distributed features: lags, rolling statistics,
    cyclical time features, and electrical domain ratios.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureEngineer.
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        dataset_cfg = config.get("app", {}).get("dataset", {})
        self._expected_columns = dataset_cfg.get(
            "expected_columns", 
            ["Global_active_power", "Global_reactive_power", "Voltage", 
             "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        )

    def engineer(self, df: DataFrame) -> DataFrame:
        """
        Execute the feature engineering pipeline.

        Args:
            df: Transformed Spark DataFrame.

        Returns:
            Spark DataFrame with engineered features.
        """
        self._logger.info("=" * 60)
        self._logger.info("Starting distributed feature engineering...")
        self._logger.info("=" * 60)
        
        initial_cols = len(df.columns)

        df = self._add_cyclical_features(df)
        df = self._add_electrical_ratios(df)
        df = self._add_lag_and_rolling_features(df)

        final_cols = len(df.columns)
        self._logger.info(
            "Feature engineering completed. Columns: %d (initial) -> %d (final)",
            initial_cols,
            final_cols
        )
        return df

    def _add_cyclical_features(self, df: DataFrame) -> DataFrame:
        """Add trigonometric representations of cyclical time variables."""
        self._logger.info("Adding cyclical time features (sin/cos)...")
        # Hour: 0-23
        df = df.withColumn("Hour_sin", F.sin(2 * math.pi * F.col("Hour") / 24))
        df = df.withColumn("Hour_cos", F.cos(2 * math.pi * F.col("Hour") / 24))
        
        # Month: 1-12
        df = df.withColumn("Month_sin", F.sin(2 * math.pi * F.col("Month") / 12))
        df = df.withColumn("Month_cos", F.cos(2 * math.pi * F.col("Month") / 12))
        
        return df

    def _add_electrical_ratios(self, df: DataFrame) -> DataFrame:
        """Add domain-specific ratios and differences."""
        self._logger.info("Adding electrical ratios and domain features...")
        
        # Reactive/Active Ratio (safeguard against division by zero)
        df = df.withColumn(
            "Reactive_Active_Ratio",
            F.when(F.col("Global_active_power") > 0, 
                   F.col("Global_reactive_power") / F.col("Global_active_power")).otherwise(0.0)
        )
        
        # Intensity/Active Ratio
        df = df.withColumn(
            "Intensity_Active_Ratio",
            F.when(F.col("Global_active_power") > 0, 
                   F.col("Global_intensity") / F.col("Global_active_power")).otherwise(0.0)
        )
        
        # Appliance contribution ratios
        # Active power is in kW (measured every minute).
        # Sub_metering is in watt-hour of active energy.
        # To convert Global_active_power to watt-hour per minute: (kW * 1000) / 60
        df = df.withColumn("Total_Power_Wh", (F.col("Global_active_power") * 1000) / 60.0)
        
        for i in range(1, 4):
            sub_col = f"Sub_metering_{i}"
            if sub_col in df.columns:
                df = df.withColumn(
                    f"{sub_col}_Ratio",
                    F.when(F.col("Total_Power_Wh") > 0,
                           F.col(sub_col) / F.col("Total_Power_Wh")).otherwise(0.0)
                )
                
        # Remainder (unmetered power)
        if all(f"Sub_metering_{i}" in df.columns for i in range(1, 4)):
            df = df.withColumn(
                "Sub_metering_Remainder",
                F.col("Total_Power_Wh") - (F.col("Sub_metering_1") + F.col("Sub_metering_2") + F.col("Sub_metering_3"))
            )
            df = df.withColumn(
                "Sub_metering_Remainder_Ratio",
                F.when(F.col("Total_Power_Wh") > 0,
                       F.col("Sub_metering_Remainder") / F.col("Total_Power_Wh")).otherwise(0.0)
            )
            
        return df

    def _add_lag_and_rolling_features(self, df: DataFrame) -> DataFrame:
        """
        Add lag features and rolling window statistics.
        Note: Requires global sorting.
        """
        self._logger.info("Adding lag and rolling statistics features...")
        
        window_spec = Window.orderBy("Timestamp")
        
        # Targets for lag/rolling features
        targets = ["Global_active_power"]
        
        lags = [1, 5, 15, 30, 60]
        
        for target in targets:
            if target not in df.columns:
                continue
                
            # 1. Lags
            for lag in lags:
                df = df.withColumn(f"{target}_Lag_{lag}", F.lag(target, lag).over(window_spec))
                
            # Power Difference (Current - Lag 1)
            df = df.withColumn(f"{target}_Diff_1", F.col(target) - F.col(f"{target}_Lag_1"))
            
            # 2. Rolling Statistics
            # We'll use row-based windows corresponding to past N minutes (since 1 row = 1 min)
            rolling_windows = [15, 60]
            
            for rw in rolling_windows:
                # rowsBetween(-rw, -1) excludes the current row from the statistic, preventing data leakage
                rw_spec = Window.orderBy("Timestamp").rowsBetween(-rw, -1)
                
                df = df.withColumn(f"{target}_Rolling_{rw}_Mean", F.mean(target).over(rw_spec))
                df = df.withColumn(f"{target}_Rolling_{rw}_Std", F.stddev(target).over(rw_spec))
                df = df.withColumn(f"{target}_Rolling_{rw}_Max", F.max(target).over(rw_spec))
                df = df.withColumn(f"{target}_Rolling_{rw}_Min", F.min(target).over(rw_spec))
                
                # Approximate Median: percentile_approx is NOT usable as a Spark window function.
                # Use (max + min) / 2 as a robust rolling median approximation over the window.
                df = df.withColumn(
                    f"{target}_Rolling_{rw}_Median",
                    (F.max(target).over(rw_spec) + F.min(target).over(rw_spec)) / 2.0
                )

        return df
