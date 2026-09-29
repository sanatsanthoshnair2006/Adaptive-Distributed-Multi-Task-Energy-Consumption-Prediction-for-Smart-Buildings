"""
Logger Factory Module
======================

Centralized logger creation for all modules. All loggers share
the same format, handlers, and configuration.

Design Decisions:
    - Static factory pattern — no instantiation required.
    - Loggers are configured once, then retrieved by name.
    - Supports console + rotating file handlers.
    - Must be initialized BEFORE ConfigLoader to avoid circular dependency,
      so it uses a bootstrap mode with defaults that are later reconfigured.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional


class LoggerFactory:
    """
    Centralized factory for creating and managing named loggers.

    Usage:
        # At startup (before config is loaded), use defaults:
        logger = LoggerFactory.get_logger("MyModule")

        # After config is loaded, reconfigure with settings:
        LoggerFactory.configure(logging_config, project_root)
    """

    _initialized: bool = False
    _project_root: str = ""
    _log_format: str = (
        "%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s"
    )
    _date_format: str = "%Y-%m-%d %H:%M:%S"
    _global_level: int = logging.INFO
    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a named logger.

        Args:
            name: Logger name (typically module or class name).

        Returns:
            Configured logging.Logger instance.
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(cls._global_level)

        # Prevent duplicate handlers when logger already exists
        if not logger.handlers:
            # Add console handler as bootstrap default
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls._global_level)
            formatter = logging.Formatter(
                cls._log_format, datefmt=cls._date_format
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Prevent propagation to root logger (avoids duplicate output)
        logger.propagate = False

        cls._loggers[name] = logger
        return logger

    @classmethod
    def configure(
        cls, logging_config: Dict, project_root: str
    ) -> None:
        """
        Reconfigure all loggers with settings from logging_config.yaml.

        Args:
            logging_config: The 'logging' section of the config dictionary.
            project_root: Absolute path to the project root.
        """
        cls._project_root = project_root
        config = logging_config.get("logging", logging_config)

        # Update global settings
        level_str = config.get("level", "INFO").upper()
        cls._global_level = getattr(logging, level_str, logging.INFO)
        cls._log_format = config.get("format", cls._log_format)
        cls._date_format = config.get("date_format", cls._date_format)

        formatter = logging.Formatter(
            cls._log_format, datefmt=cls._date_format
        )

        # Configure file handlers
        file_configs = config.get("file", {})
        file_handlers = {}

        for handler_name, handler_cfg in file_configs.items():
            if not handler_cfg.get("enabled", False):
                continue

            log_path = os.path.join(
                project_root, handler_cfg.get("path", f"logs/{handler_name}.log")
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            handler_level = getattr(
                logging,
                handler_cfg.get("level", "DEBUG").upper(),
                logging.DEBUG,
            )
            max_bytes = handler_cfg.get("max_bytes", 10485760)
            backup_count = handler_cfg.get("backup_count", 5)

            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(handler_level)
            file_handler.setFormatter(formatter)
            file_handlers[handler_name] = file_handler

        # Reconfigure all existing loggers
        console_config = config.get("console", {})
        console_enabled = console_config.get("enabled", True)
        console_level = getattr(
            logging,
            console_config.get("level", "INFO").upper(),
            logging.INFO,
        )

        for logger_name, logger in cls._loggers.items():
            # Clear existing handlers
            logger.handlers.clear()
            logger.setLevel(cls._global_level)

            # Re-add console handler if enabled
            if console_enabled:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(console_level)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            # Add the app file handler to all loggers
            if "app" in file_handlers:
                logger.addHandler(file_handlers["app"])

        cls._initialized = True

        main_logger = cls.get_logger("LoggerFactory")
        main_logger.info(
            "Logging configured: level=%s, file_handlers=%s",
            level_str,
            list(file_handlers.keys()),
        )

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if LoggerFactory has been configured."""
        return cls._initialized
