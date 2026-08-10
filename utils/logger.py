"""
BMKG Strong Motion Analyzer (BSMA)
Module: utils/logger.py

Description
-----------
Centralized logging utilities for BSMA.

Features
--------
- Thread-safe singleton-style initialization.
- RotatingFileHandler for persistent log storage.
- Console logging.
- Automatic log-directory creation.
- Configurable log level and rotation policy.
- Prevention of duplicate BSMA handlers.
- Compatible ``setup_logger()`` and ``get_logger()`` APIs.

Python
------
>= 3.12
"""

from __future__ import annotations

import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

# =============================================================================
# Public Configuration
# =============================================================================

DEFAULT_LOG_DIRECTORY = Path("logs")
DEFAULT_LOG_FILENAME = "bsma.log"

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
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

# Internal handler identification.
_HANDLER_MARKER = "_bsma_handler"

# =============================================================================
# Internal State
# =============================================================================

_lock = threading.RLock()
_initialized = False

# =============================================================================
# Validation
# =============================================================================


def _validate_configuration(
    level: int,
    max_bytes: int,
    backup_count: int,
) -> None:
    """Validate logging configuration parameters."""

    if not isinstance(level, int):
        raise TypeError(
            f"level must be an int logging level, got {type(level).__name__}."
        )

    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero.")

    if backup_count < 0:
        raise ValueError("backup_count must be greater than or equal to zero.")


# =============================================================================
# Handler Management
# =============================================================================


def _remove_existing_bsma_handlers(
    logger: logging.Logger,
) -> None:
    """Remove handlers previously created by BSMA."""

    for handler in logger.handlers[:]:
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def _create_file_handler(
    logfile: Path,
    formatter: logging.Formatter,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> RotatingFileHandler:
    """Create and configure the rotating BSMA file handler."""

    handler = RotatingFileHandler(
        filename=logfile,
        mode="a",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    handler.setLevel(level)
    handler.setFormatter(formatter)

    setattr(handler, _HANDLER_MARKER, True)

    return handler


def _create_console_handler(
    formatter: logging.Formatter,
    level: int,
) -> logging.StreamHandler:
    """Create and configure the BSMA console handler."""

    handler = logging.StreamHandler()

    handler.setLevel(level)
    handler.setFormatter(formatter)

    setattr(handler, _HANDLER_MARKER, True)

    return handler


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

    Initialization is thread-safe and idempotent.

    Parameters
    ----------
    log_directory:
        Directory where BSMA log files are stored.

    filename:
        Name of the main log file.

    level:
        Logging threshold, e.g. ``logging.DEBUG`` or ``logging.INFO``.

    max_bytes:
        Maximum log-file size before rotation.

    backup_count:
        Number of rotated log files retained.

    Notes
    -----
    Only handlers created by BSMA are managed by this function.
    Existing third-party/application handlers are preserved.
    """

    global _initialized

    with _lock:
        if _initialized:
            return

        _validate_configuration(
            level=level,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

        log_directory = Path(log_directory)

        if not filename:
            raise ValueError("filename must not be empty.")

        if Path(filename).name != filename:
            raise ValueError(
                "filename must contain only a file name, not a directory path."
            )

        log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logfile = log_directory / filename

        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )

        root_logger = logging.getLogger()

        root_logger.setLevel(level)

        # Remove only handlers previously installed by BSMA.
        _remove_existing_bsma_handlers(root_logger)

        file_handler = _create_file_handler(
            logfile=logfile,
            formatter=formatter,
            level=level,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

        console_handler = _create_console_handler(
            formatter=formatter,
            level=level,
        )

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        _initialized = True

        root_logger.info("========================================")
        root_logger.info(" BSMA Logging Initialized")
        root_logger.info(" Log File : %s", logfile)
        root_logger.info(" Log Level: %s", logging.getLevelName(level))
        root_logger.info(" Max Size : %s bytes", max_bytes)
        root_logger.info(" Backups  : %s", backup_count)
        root_logger.info("========================================")


# =============================================================================
# Public Logger API
# =============================================================================


def setup_logger(
    name: str | None = None,
) -> logging.Logger:
    """
    Return a configured BSMA logger.

    Parameters
    ----------
    name:
        Logger name. Normally use ``__name__``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("Application started.")
    """

    if not _initialized:
        configure_logging()

    return logging.getLogger(name)


# Backward-compatible alias.
get_logger = setup_logger


# =============================================================================
# Runtime Status
# =============================================================================


def is_logging_configured() -> bool:
    """
    Return whether the BSMA logging system has been initialized.
    """

    with _lock:
        return _initialized


# =============================================================================
# Example
# =============================================================================


if __name__ == "__main__":
    logger = get_logger(__name__)

    logger.debug("Debug message.")
    logger.info("Application started.")
    logger.warning("This is a warning.")
    logger.error("This is an error.")
    logger.critical("Critical failure example.")