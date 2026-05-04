"""Processing checkpoint management for resumable pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingCheckpoint:
    """Track which subjects have been processed for resumable batch operations."""

    def __init__(self, checkpoint_path: str | Path):
        self.path = Path(checkpoint_path)
        self._processed: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._processed = set(data.get("processed_subjects", []))
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt checkpoint file at %s, starting fresh", self.path)
                self._processed = set()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"processed_subjects": sorted(self._processed)}, indent=2))

    def is_processed(self, subject_id: str) -> bool:
        return subject_id in self._processed

    def mark_processed(self, subject_id: str) -> None:
        self._processed.add(subject_id)
        self._save()

    @property
    def processed_subjects(self) -> set[str]:
        return self._processed.copy()

    @property
    def count(self) -> int:
        return len(self._processed)
