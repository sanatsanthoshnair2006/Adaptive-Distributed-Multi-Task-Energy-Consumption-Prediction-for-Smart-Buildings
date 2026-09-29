"""
Tasks Package
=============

Multi-task prediction framework.

Importing this package automatically registers all four built-in tasks via
``task_registry._auto_register()``.

Public API
----------
- ``BaseTask``          — abstract contract every task must implement
- ``TaskActivePower``   — Global_active_power prediction
- ``TaskSubMetering1``  — Sub_metering_1 prediction
- ``TaskSubMetering2``  — Sub_metering_2 prediction
- ``TaskSubMetering3``  — Sub_metering_3 prediction
- ``TaskRegistry``      — centralised registry (get / register / remove tasks)
- ``TaskRegistryError`` — raised for invalid registry operations
- ``TaskPersistenceError`` — raised for save/load failures
"""

from src.tasks.base_task import BaseTask, TaskPersistenceError
from src.tasks.task_active_power import TaskActivePower
from src.tasks.task_sub_metering_1 import TaskSubMetering1
from src.tasks.task_sub_metering_2 import TaskSubMetering2
from src.tasks.task_sub_metering_3 import TaskSubMetering3
# Importing task_registry triggers _auto_register() which populates the registry
from src.tasks.task_registry import TaskRegistry, TaskRegistryError

__all__ = [
    "BaseTask",
    "TaskPersistenceError",
    "TaskActivePower",
    "TaskSubMetering1",
    "TaskSubMetering2",
    "TaskSubMetering3",
    "TaskRegistry",
    "TaskRegistryError",
]
