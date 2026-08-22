"""Logging setup shared by terminal and future experiment runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOGGER_NAME = "react_reproduction"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_level: str = "INFO",
    *,
    log_file: Path | None = None,
    quiet: bool = False,
) -> logging.Logger:
    """Configure immediate terminal output and an optional UTF-8 log file."""
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level!r}")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not quiet:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        resolved_log_file = log_file.resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            resolved_log_file,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if quiet and log_file is None:
        logger.addHandler(logging.NullHandler())

    return logger
