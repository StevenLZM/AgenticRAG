"""Single-writer async SQLite checkpoint backend for LangGraph graphs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentic_rag.config import Settings


@dataclass(frozen=True, slots=True)
class CheckpointBackend:
    """Open independently configured query and ingestion checkpoint savers."""

    settings: Settings
    busy_timeout_ms: int = 5_000
    _query_path: Path = field(init=False, repr=False)
    _ingestion_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.settings.query_worker_count != 1
            or self.settings.ingestion_worker_count != 1
        ):
            raise ValueError("SQLite checkpoint databases require exactly one worker")
        if self.busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")

        query_path = self.settings.query_checkpoint_path.expanduser().resolve()
        ingestion_path = self.settings.ingestion_checkpoint_path.expanduser().resolve()
        if query_path == ingestion_path:
            raise ValueError(
                "query and ingestion checkpoints require distinct database paths"
            )
        object.__setattr__(self, "_query_path", query_path)
        object.__setattr__(self, "_ingestion_path", ingestion_path)

    @asynccontextmanager
    async def open_query(self) -> AsyncIterator[BaseCheckpointSaver[str]]:
        """Open the query checkpoint saver for the duration of graph use."""

        async with self._open(self._query_path) as saver:
            yield saver

    @asynccontextmanager
    async def open_ingestion(self) -> AsyncIterator[BaseCheckpointSaver[str]]:
        """Open the ingestion checkpoint saver for the duration of graph use."""

        async with self._open(self._ingestion_path) as saver:
            yield saver

    @asynccontextmanager
    async def _open(self, path: Path) -> AsyncIterator[AsyncSqliteSaver]:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(
            path,
            timeout=self.busy_timeout_ms / 1_000,
        ) as connection:
            busy_timeout_cursor = await connection.execute(
                f"PRAGMA busy_timeout = {self.busy_timeout_ms}"
            )
            await busy_timeout_cursor.close()
            journal_mode_cursor = await connection.execute("PRAGMA journal_mode = WAL")
            await journal_mode_cursor.close()
            saver = AsyncSqliteSaver(connection)
            await saver.setup()
            yield saver
