"""
Task: Sub-metering 3 Prediction
=================================

Predicts ``Sub_metering_3`` — the active energy consumed (Wh) by the electric
water-heater and air-conditioner circuit every minute.

Feature set
-----------
Sub-metering 3 often shows strong seasonal and temperature-driven patterns
(heating in winter, cooling in summer), making the cyclical month and hour
features particularly valuable.  As with the other sub-metering tasks, the
feature set prioritises inter-appliance channels, global electrical
measurements, and time-based features, while omitting lag/rolling features
that were computed specifically on ``Global_active_power``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.tasks._concrete_task_mixin import ConcreteTaskMixin
from src.tasks.base_task import BaseTask


class TaskSubMetering3(ConcreteTaskMixin, BaseTask):
    """
    Prediction task for ``Sub_metering_3`` (water-heater / AC circuit).

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
        return "sub_metering_3"

    def get_target_column(self) -> str:
        """Return the prediction target column name."""
        return "Sub_metering_3"

    def get_feature_columns(self) -> List[str]:
        """
        Return the ordered feature column list for Sub-metering 3.

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
            # --- Cyclical time features (month captures seasonal HVAC usage) ---
            "Hour_sin",
            "Hour_cos",
            "Month_sin",
            "Month_cos",
            # --- Global electrical measurements ---
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            # --- Other sub-metering channels (inter-appliance correlation) ---
            "Sub_metering_1",
            "Sub_metering_2",
            # --- Electrical domain ratios ---
            "Reactive_Active_Ratio",
            "Intensity_Active_Ratio",
            "Total_Power_Wh",
            "Sub_metering_1_Ratio",
            "Sub_metering_2_Ratio",
            "Sub_metering_3_Ratio",
            "Sub_metering_Remainder",
            "Sub_metering_Remainder_Ratio",
        ]
