"""
utils/logger.py
===============

Centralized logging utilities for the BSMA (BMKG Strong Motion Analyzer).

This module provides a production-ready logging configuration using:

- RotatingFileHandler
- Console logging
- Singleton logger initialization
- Automatic log directory creation
- Thread-safe initialization
- Configurable log level and file size

Author
------
BSMA Development Team

Python
------
>= 3.12

Example
-------

from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Application started.")
logger.warning("QC warning.")
logger.error("Waveform could not be loaded.")

"""

from __future__ import annotations

import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_LOG_DIRECTORY = Path("logs")
DEFAULT_LOG_FILENAME = "bsma.log"

DEFAULT_MAX_BYTES = 10 * 1024 * 1024      # 10 MB
DEFAULT_BACKUP_COUNT = 5

DEFAULT_LEVEL = logging.INFO

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(filename)s:%(lineno)d | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# Internal State
# =============================================================================

_lock = threading.Lock()

_initialized = False


# =============================================================================
# Logger Configuration
# =============================================================================

def configure_logging(
    *,
    log_directory: Path | str = DEFAULT_LOG_DIRECTORY,
    filename: str = DEFAULT_LOG_FILENAME,
    level: int = DEFAULT_LEVEL,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """
    Configure the global BSMA logging system.

    This function is thread-safe and only performs initialization once.

    Parameters
    ----------
    log_directory :
        Directory where log files are stored.

    filename :
        Log filename.

    level :
        Root logging level.

    max_bytes :
        Maximum size before rotating.

    backup_count :
        Number of rotated files retained.
    """

    global _initialized

    if _initialized:
        return

    with _lock:

        if _initialized:
            return

        log_directory = Path(log_directory)
        log_directory.mkdir(parents=True, exist_ok=True)

        logfile = log_directory / filename

        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )

        file_handler = RotatingFileHandler(
            filename=logfile,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()

        root_logger.setLevel(level)

        # Prevent duplicated handlers
        root_logger.handlers.clear()

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        _initialized = True

        root_logger.info("========================================")
        root_logger.info(" BSMA Logging Initialized")
        root_logger.info(" Log File : %s", logfile)
        root_logger.info("========================================")


# =============================================================================
# Public API
# =============================================================================

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name :
        Usually use:

            logger = get_logger(__name__)

    Returns
    -------
    logging.Logger
    """

    if not _initialized:
        configure_logging()

    return logging.getLogger(name)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":

    logger = get_logger(__name__)

    logger.debug("Debug message")

    logger.info("Application started")

    logger.warning("This is a warning")

    logger.error("This is an error")

    logger.critical("Critical failure example")