from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    log_file: str | None = None,
    logger_name: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger.

    Call once at the beginning of a script, e.g.:

        logger = setup_logging(level="DEBUG", log_file="logs/dev_run.log")
        logger.info("Starting run...")
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level))

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Stream handler for console output
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
