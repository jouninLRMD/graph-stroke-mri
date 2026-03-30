"""Memory tracking utilities for large-scale MRI processing."""

from __future__ import annotations

import gc
import logging

import psutil
import torch

logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> float:
    """Return current process RSS memory usage in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def log_memory(step: str) -> float:
    """Log current memory usage with a descriptive step label.

    Returns:
        Memory usage in MB.
    """
    mem_mb = get_memory_usage_mb()
    logger.info("Memory usage after %s: %.2f MB", step, mem_mb)
    return mem_mb


def clear_gpu_memory() -> None:
    """Force GPU memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
