"""
Task: Sub-metering 2 Prediction
=================================

Predicts ``Sub_metering_2`` — the active energy consumed (Wh) by the laundry
room circuit (washing machine, tumble-dryer, refrigerator, light) every minute.

Feature set
-----------
Sub-metering 2 has episodic, longer-duration appliance runs (washing cycles)
contrasted with shorter on/off patterns.  The feature set mirrors
``Sub_metering_1`` — inter-appliance channels, global measurements, temporal
features, and electrical ratios — with the target and predictor roles swapped
for the sub-metering columns.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.tasks._concrete_task_mixin import ConcreteTaskMixin
from src.tasks.base_task import BaseTask


class TaskSubMetering2(ConcreteTaskMixin, BaseTask):
    """
    Prediction task for ``Sub_metering_2`` (laundry room circuit).

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
        return "sub_metering_2"

    def get_target_column(self) -> str:
        """Return the prediction target column name."""
        return "Sub_metering_2"

    def get_feature_columns(self) -> List[str]:
        """
        Return the ordered feature column list for Sub-metering 2.

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
            # --- Global electrical measurements ---
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            # --- Other sub-metering channels (inter-appliance correlation) ---
            "Sub_metering_1",
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
        ]
