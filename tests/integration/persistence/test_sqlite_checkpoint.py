"""Integration tests for the real LangGraph SQLite checkpoint saver."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic_rag.config import Settings
from agentic_rag.persistence.checkpoint import CheckpointBackend


class ReplayState(TypedDict):
    steps: Annotated[list[str], operator.add]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        mysql_dsn="mysql+asyncmy://rag:rag@127.0.0.1/rag",
        deepseek_base_url="https://models.example.invalid/v1",
        qwen_embedding_base_url="https://embeddings.example.invalid/v1",
        query_checkpoint_path=tmp_path / "query/checkpoints.sqlite",
        ingestion_checkpoint_path=tmp_path / "ingestion/checkpoints.sqlite",
    )


def _graph() -> StateGraph[ReplayState]:
    graph = StateGraph(ReplayState)
    graph.add_node("record_first", lambda _state: {"steps": ["first"]})
    graph.add_node("record_second", lambda _state: {"steps": ["second"]})
    graph.add_edge(START, "record_first")
    graph.add_edge("record_first", "record_second")
    graph.add_edge("record_second", END)
    return graph


async def _pragma(saver: AsyncSqliteSaver, name: str) -> str | int:
    cursor = await saver.conn.execute(f"PRAGMA {name}")
    try:
        row = await cursor.fetchone()
    finally:
        await cursor.close()
    assert row is not None
    return row[0]


@pytest.mark.integration
async def test_checkpoint_resumes_interrupted_graph_after_database_reopen(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = CheckpointBackend(settings)
    config = {"configurable": {"thread_id": "user-1:thread-1"}}

    async with backend.open_query() as saver:
        interrupted = _graph().compile(checkpointer=saver, interrupt_after=["record_first"])
        assert await interrupted.ainvoke({"steps": []}, config=config) == {
            "steps": ["first"]
        }

    async with backend.open_query() as saver:
        resumed = _graph().compile(checkpointer=saver)
        assert await resumed.ainvoke(None, config=config) == {
            "steps": ["first", "second"]
        }


@pytest.mark.integration
async def test_query_and_ingestion_use_independent_configured_databases(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = CheckpointBackend(settings, busy_timeout_ms=7_500)

    async with backend.open_query() as query_saver:
        assert isinstance(query_saver, AsyncSqliteSaver)
        await query_saver.setup()
        query_journal_mode = await _pragma(query_saver, "journal_mode")
        query_timeout = await _pragma(query_saver, "busy_timeout")

    async with backend.open_ingestion() as ingestion_saver:
        assert isinstance(ingestion_saver, AsyncSqliteSaver)
        await ingestion_saver.setup()
        ingestion_journal_mode = await _pragma(ingestion_saver, "journal_mode")
        ingestion_timeout = await _pragma(ingestion_saver, "busy_timeout")

    assert settings.query_checkpoint_path.is_file()
    assert settings.ingestion_checkpoint_path.is_file()
    assert settings.query_checkpoint_path != settings.ingestion_checkpoint_path
    assert query_journal_mode == ingestion_journal_mode == "wal"
    assert query_timeout == ingestion_timeout == 7_500


@pytest.mark.integration
def test_checkpoint_backend_rejects_equal_database_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={"ingestion_checkpoint_path": settings.query_checkpoint_path}
    )

    with pytest.raises(ValueError, match="distinct database paths"):
        CheckpointBackend(settings)


@pytest.mark.integration
def test_checkpoint_backend_rejects_normalized_database_path_alias(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "shared/checkpoints.sqlite"
    aliased_path = tmp_path / "shared/nested/../checkpoints.sqlite"
    settings = _settings(tmp_path).model_copy(
        update={
            "query_checkpoint_path": checkpoint_path,
            "ingestion_checkpoint_path": aliased_path,
        }
    )

    with pytest.raises(ValueError, match="distinct database paths"):
        CheckpointBackend(settings)


@pytest.mark.integration
@pytest.mark.parametrize("worker_field", ("query_worker_count", "ingestion_worker_count"))
def test_checkpoint_backend_rejects_multiple_sqlite_writers(
    tmp_path: Path, worker_field: str
) -> None:
    settings = _settings(tmp_path).model_copy(update={worker_field: 2})

    with pytest.raises(ValueError, match="exactly one worker"):
        CheckpointBackend(settings)
