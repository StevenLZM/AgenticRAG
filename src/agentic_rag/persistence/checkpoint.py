"""Single-writer SQLite checkpoint backend for LangGraph graphs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from agentic_rag.config import Settings


@dataclass(frozen=True, slots=True)
class CheckpointBackend:
    """Open independently configured query and ingestion checkpoint savers."""

    settings: Settings
    busy_timeout_ms: int = 5_000

    def __post_init__(self) -> None:
        if (
            self.settings.query_worker_count != 1
            or self.settings.ingestion_worker_count != 1
        ):
            raise ValueError("SQLite checkpoint databases require exactly one worker")
        if self.busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")

    @contextmanager
    def open_query(self) -> Iterator[BaseCheckpointSaver[str]]:
        """Open the query checkpoint saver for the duration of graph use."""

        with self._open(self.settings.query_checkpoint_path) as saver:
            yield saver

    @contextmanager
    def open_ingestion(self) -> Iterator[BaseCheckpointSaver[str]]:
        """Open the ingestion checkpoint saver for the duration of graph use."""

        with self._open(self.settings.ingestion_checkpoint_path) as saver:
            yield saver

    @contextmanager
    def _open(self, path: Path) -> Iterator[SqliteSaver]:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            path,
            check_same_thread=False,
            timeout=self.busy_timeout_ms / 1_000,
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            connection.execute("PRAGMA journal_mode = WAL")
            saver = SqliteSaver(connection)
            saver.setup()
            yield saver
        finally:
            connection.close()
