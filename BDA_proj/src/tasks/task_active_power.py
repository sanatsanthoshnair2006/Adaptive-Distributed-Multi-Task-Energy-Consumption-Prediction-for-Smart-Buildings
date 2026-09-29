"""
Task: Global Active Power Prediction
======================================

Predicts ``Global_active_power`` (kilowatts) — the total active electrical
power drawn by the household at each one-minute interval.

Feature set
-----------
This task uses the full set of engineered features produced by
``FeatureEngineer``:
    - Cyclical time features (Hour_sin, Hour_cos, Month_sin, Month_cos)
    - Electrical ratios (Reactive_Active_Ratio, Intensity_Active_Ratio,
      Total_Power_Wh, Sub_metering_*_Ratio, Sub_metering_Remainder_Ratio)
    - Lag features      (Global_active_power_Lag_1/5/15/30/60)
    - Difference        (Global_active_power_Diff_1)
    - Rolling stats     (mean, std, max, min, median over 15 and 60 minutes)
    - Raw sensor readings (Global_reactive_power, Voltage, Global_intensity,
      Sub_metering_1, Sub_metering_2, Sub_metering_3)
    - Temporal integers (Year, Month, Day, Hour, DayOfWeek, IsWeekend, Minute)

The lag and rolling features are computed on the target itself, so this task
benefits most from them.  Sub-metering tasks use a subset (see their modules).
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.tasks._concrete_task_mixin import ConcreteTaskMixin
from src.tasks.base_task import BaseTask


class TaskActivePower(ConcreteTaskMixin, BaseTask):
    """
    Prediction task for ``Global_active_power``.

    Inherits all shared logic from ``ConcreteTaskMixin`` and only declares
    its identity and feature set here.

    Args:
        config (Dict[str, Any]):  Full application configuration dict.
        hdfs_client:              ``HDFSClient`` instance for Parquet I/O.
    """

    def __init__(self, config: Dict[str, Any], hdfs_client: Any) -> None:
        super().__init__(config=config, hdfs_client=hdfs_client)
        self._logger.info("Task initialised: %s", self.get_task_name())

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_task_name(self) -> str:
        """Return unique task identifier."""
        return "global_active_power"

    def get_target_column(self) -> str:
        """Return the prediction target column name."""
        return "Global_active_power"

    def get_feature_columns(self) -> List[str]:
        """
        Return the ordered feature column list for this task.

        All lag and rolling features are derived from ``Global_active_power``
        itself, so they are included.  The raw target column is excluded
        from the feature list (it is only in the label position).

        Returns:
            List[str]: Feature columns.
        """
        return [
            # --- Temporal integers ---
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "DayOfWeek",
            "IsWeekend",
            # --- Cyclical time features ---
            "Hour_sin",
            "Hour_cos",
            "Month_sin",
            "Month_cos",
            # --- Raw sensor readings (correlated predictors) ---
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
            # --- Electrical domain ratios ---
            "Reactive_Active_Ratio",
            "Intensity_Active_Ratio",
            "Total_Power_Wh",
            "Sub_metering_1_Ratio",
            "Sub_metering_2_Ratio",
            "Sub_metering_3_Ratio",
            "Sub_metering_Remainder",
            "Sub_metering_Remainder_Ratio",
            # --- Lag features ---
            "Global_active_power_Lag_1",
            "Global_active_power_Lag_5",
            "Global_active_power_Lag_15",
            "Global_active_power_Lag_30",
            "Global_active_power_Lag_60",
            # --- Difference ---
            "Global_active_power_Diff_1",
            # --- Rolling statistics (15-minute window) ---
            "Global_active_power_Rolling_15_Mean",
            "Global_active_power_Rolling_15_Std",
            "Global_active_power_Rolling_15_Max",
            "Global_active_power_Rolling_15_Min",
            "Global_active_power_Rolling_15_Median",
            # --- Rolling statistics (60-minute window) ---
            "Global_active_power_Rolling_60_Mean",
            "Global_active_power_Rolling_60_Std",
            "Global_active_power_Rolling_60_Max",
            "Global_active_power_Rolling_60_Min",
            "Global_active_power_Rolling_60_Median",
        ]
