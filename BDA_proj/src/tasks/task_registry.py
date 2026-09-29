"""
Task Registry Module
=====================

Centralised registry for all prediction tasks.

Design Decisions:
    - Auto-registration on import:  The four built-in tasks are registered
      when this module is first imported — no manual wiring required.
    - Factory-based registration:  Each entry is a *factory callable* rather
      than a pre-instantiated object.  This allows the registry to be imported
      before a ``SparkSession`` or HDFS client exists, and lets callers create
      as many independent instances as they need.
    - Thread-safety note:  For the current single-driver use-case, standard
      dict operations are sufficient.  If distributed registration is ever
      required, a threading.Lock can be added around mutations.
    - Extensibility (Open/Closed):  Third parties can call ``register_task``
      to add new tasks without modifying this file.

Usage
-----
::

    from src.tasks.task_registry import TaskRegistry

    # Retrieve all registered task factories
    all_tasks = TaskRegistry.get_all_tasks()

    # Instantiate a specific task
    task = TaskRegistry.get_task("global_active_power")(config, hdfs_client)

    # Register a custom task
    TaskRegistry.register_task("my_task", MyTask)

    # Remove a task
    TaskRegistry.remove_task("my_task")
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Type

from src.logging.logger_factory import LoggerFactory
from src.tasks.base_task import BaseTask


# Type alias: a task factory is any callable that accepts (config, hdfs_client)
# and returns a BaseTask instance.  Typically this will just be the class itself.
TaskFactory = Callable[..., BaseTask]


class TaskRegistryError(Exception):
    """Raised for invalid registry operations."""


class TaskRegistry:
    """
    Centralised, thread-safe registry for prediction task factories.

    All methods are class-level (no instantiation required).

    Attributes:
        _registry (Dict[str, TaskFactory]):  Mapping of task name → factory.
        _lock (threading.Lock):              Guards mutations.
        _logger:                             Shared logger.
    """

    _registry: Dict[str, TaskFactory] = {}
    _lock: threading.Lock = threading.Lock()
    _logger = LoggerFactory.get_logger("TaskRegistry")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register_task(
        cls,
        task_name: str,
        task_factory: TaskFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a task factory under ``task_name``.

        Args:
            task_name (str):            Unique task identifier.  Must match
                                        the value returned by
                                        ``BaseTask.get_task_name()``.
            task_factory (TaskFactory): Callable (typically the class itself)
                                        that produces a ``BaseTask`` instance.
            overwrite (bool):           Allow replacing an existing entry.
                                        Defaults to ``False``.

        Raises:
            TaskRegistryError:  If ``task_name`` is already registered and
                                ``overwrite=False``, or if ``task_factory``
                                is not callable.
        """
        if not callable(task_factory):
            raise TaskRegistryError(
                f"task_factory for '{task_name}' must be callable, "
                f"got {type(task_factory).__name__}."
            )

        with cls._lock:
            if task_name in cls._registry and not overwrite:
                raise TaskRegistryError(
                    f"Task '{task_name}' is already registered. "
                    f"Pass overwrite=True to replace it."
                )
            cls._registry[task_name] = task_factory
            cls._logger.info("Registered task: '%s'", task_name)

    @classmethod
    def remove_task(cls, task_name: str) -> None:
        """
        Remove a task from the registry.

        Args:
            task_name (str):  Task identifier to remove.

        Raises:
            TaskRegistryError:  If ``task_name`` is not registered.
        """
        with cls._lock:
            if task_name not in cls._registry:
                raise TaskRegistryError(
                    f"Cannot remove '{task_name}': task is not registered."
                )
            del cls._registry[task_name]
            cls._logger.info("Removed task: '%s'", task_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @classmethod
    def get_task(cls, task_name: str) -> TaskFactory:
        """
        Retrieve the factory for a specific task.

        Args:
            task_name (str):  Task identifier.

        Returns:
            TaskFactory:  The registered factory callable.

        Raises:
            TaskRegistryError:  If ``task_name`` is not registered.

        Example::

            task = TaskRegistry.get_task("global_active_power")(config, hdfs_client)
        """
        with cls._lock:
            factory = cls._registry.get(task_name)

        if factory is None:
            available = cls.get_task_names()
            raise TaskRegistryError(
                f"Task '{task_name}' is not registered. "
                f"Available tasks: {available}"
            )
        return factory

    @classmethod
    def get_all_tasks(cls) -> Dict[str, TaskFactory]:
        """
        Return a snapshot of all registered task factories.

        Returns:
            Dict[str, TaskFactory]:  Copy of the registry mapping.
        """
        with cls._lock:
            return dict(cls._registry)

    @classmethod
    def get_task_names(cls) -> List[str]:
        """
        Return the names of all currently registered tasks.

        Returns:
            List[str]:  Sorted list of task names.
        """
        with cls._lock:
            return sorted(cls._registry.keys())

    @classmethod
    def instantiate_all(
        cls,
        config: Dict[str, Any],
        hdfs_client: Any,
    ) -> List[BaseTask]:
        """
        Instantiate every registered task with the given config and HDFS client.

        This is a convenience method for pipelines that need to run all tasks.

        Args:
            config (Dict[str, Any]):  Full application configuration dict.
            hdfs_client:              ``HDFSClient`` instance.

        Returns:
            List[BaseTask]:  One instance per registered task, in registration
                             order (alphabetical after auto-registration).
        """
        cls._logger.info(
            "Instantiating all registered tasks: %s", cls.get_task_names()
        )
        instances: List[BaseTask] = []
        for name, factory in cls.get_all_tasks().items():
            try:
                instance = factory(config=config, hdfs_client=hdfs_client)
                instances.append(instance)
                cls._logger.debug("Instantiated task: '%s'", name)
            except Exception as exc:
                cls._logger.error(
                    "Failed to instantiate task '%s': %s", name, exc
                )
                raise
        return instances

    @classmethod
    def is_registered(cls, task_name: str) -> bool:
        """
        Check whether a task name is currently registered.

        Args:
            task_name (str):  Task identifier.

        Returns:
            bool:  ``True`` if registered, ``False`` otherwise.
        """
        with cls._lock:
            return task_name in cls._registry

    def __repr__(cls) -> str:  # pragma: no cover
        return f"TaskRegistry(tasks={cls.get_task_names()})"


# ---------------------------------------------------------------------------
# Auto-register all built-in tasks when this module is imported
# ---------------------------------------------------------------------------

def _auto_register() -> None:
    """
    Register the four built-in prediction tasks.

    This function is called once at module import time.  Importing
    ``task_registry`` is the only step required to make all tasks available.
    """
    # Late imports to avoid circular dependency at parse time
    from src.tasks.task_active_power import TaskActivePower
    from src.tasks.task_sub_metering_1 import TaskSubMetering1
    from src.tasks.task_sub_metering_2 import TaskSubMetering2
    from src.tasks.task_sub_metering_3 import TaskSubMetering3

    _built_in_tasks: List[Type[BaseTask]] = [
        TaskActivePower,
        TaskSubMetering1,
        TaskSubMetering2,
        TaskSubMetering3,
    ]

    logger = LoggerFactory.get_logger("TaskRegistry._auto_register")
    logger.info("=" * 60)
    logger.info("Auto-registering built-in prediction tasks...")

    for task_class in _built_in_tasks:
        # Instantiate a temporary "dummy" to read the task name without
        # needing a live config or HDFS client.  We use None placeholders —
        # task classes must not access self._config or self._hdfs_client
        # during __init__ beyond storing them.
        try:
            # Use the class-level get_task_name via a temporary instance
            # that bypasses BaseTask.__init__ — safer: just hardcode the name
            # mapping from the class attribute if available, or probe via a
            # minimal init.  Since our __init__ only stores references and
            # logs, passing None is safe here.
            dummy = task_class.__new__(task_class)
            task_name = dummy.get_task_name()
            TaskRegistry.register_task(task_name, task_class)
        except Exception as exc:
            logger.error(
                "Failed to auto-register %s: %s", task_class.__name__, exc
            )
            raise

    logger.info(
        "Auto-registration complete. Tasks: %s", TaskRegistry.get_task_names()
    )
    logger.info("=" * 60)


_auto_register()
