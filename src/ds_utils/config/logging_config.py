"""Logging configuration for ds_utils.

This module provides standardized logging configuration with console and file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Literal

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Package logger name
PACKAGE_LOGGER_NAME = "ds_utils"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    console_output: bool = True,
) -> logging.Logger:
    """Set up logging configuration for the ds_utils package.

    Args:
        level: Logging level. Can be int (e.g., logging.INFO) or string (e.g., 'INFO').
        log_file: Optional path to a log file. If provided, logs will also be written to file.
        log_format: Log message format string.
        date_format: Date format string for log timestamps.
        console_output: Whether to output logs to console.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging(level='DEBUG', log_file='ds_utils.log')
        >>> logger.info("Starting analysis")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get or create the package logger
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional name for the logger. If None, returns the package logger.
            If provided, creates a child logger under the package namespace.

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger("plotting")
        >>> logger.debug("Creating figure")
    """
    if name is None:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def set_log_level(
    level: int | Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    """Set the logging level for the package logger.

    Args:
        level: Logging level to set.

    Example:
        >>> set_log_level('DEBUG')
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging() -> None:
    """Disable all logging for the package."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.disabled = True


def enable_logging() -> None:
    """Enable logging for the package (if previously disabled)."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.disabled = False
