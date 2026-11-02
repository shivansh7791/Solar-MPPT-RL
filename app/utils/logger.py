"""Centralized Logging Infrastructure for RL-Based Solar MPPT Project.

Provides a unified, reusable logging interface across all modules with configurable
formatting, console streaming, timestamping, and hierarchical logger resolution.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# Standard format for engineering and simulation logs
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "mppt_project",
    level: int | str = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a standardized logger instance.

    Parameters
    ----------
    name : str, optional
        Logger namespace or component identifier, by default "mppt_project".
    level : int | str, optional
        Logging severity threshold (e.g. logging.INFO, "DEBUG"), by default logging.INFO.
    log_file : Optional[str], optional
        Optional filesystem path for appending log messages, by default None.

    Returns
    -------
    logging.Logger
        Configured standard library Logger instance.
    """
    logger = logging.getLogger(name)

    # Normalize log level if provided as a string
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    logger.setLevel(numeric_level)

    # Prevent attaching multiple handlers if setup_logger is called repeatedly
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "mppt_project") -> logging.Logger:
    """Retrieve an existing logger or create one with default project settings.

    Parameters
    ----------
    name : str, optional
        Logger name, by default "mppt_project".

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return setup_logger(name)
