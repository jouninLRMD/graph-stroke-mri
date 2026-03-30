"""Structured logging setup for the stroke-GAT pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure logging with console and optional file output.

    Args:
        log_file: Path to log file. If None, logs only to console.
        level: Logging level.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(path)))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
