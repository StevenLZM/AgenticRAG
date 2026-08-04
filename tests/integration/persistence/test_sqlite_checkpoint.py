"""Integration tests for the real LangGraph SQLite checkpoint saver."""

# ruff: noqa: E402 - optional checkpoint API must be checked before adapter import

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, TypedDict

import pytest

sqlite_checkpoint = pytest.importorskip(
    "langgraph.checkpoint.sqlite",
    reason="declared langgraph-checkpoint-sqlite package is unavailable",
)
if not hasattr(sqlite_checkpoint, "SqliteSaver"):
    pytest.skip(
        "declared langgraph-checkpoint-sqlite saver API is unavailable",
        allow_module_level=True,
    )

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


@pytest.mark.integration
def test_checkpoint_resumes_interrupted_graph_after_database_reopen(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = CheckpointBackend(settings)
    config = {"configurable": {"thread_id": "user-1:thread-1"}}

    with backend.open_query() as saver:
        interrupted = _graph().compile(checkpointer=saver, interrupt_after=["record_first"])
        assert interrupted.invoke({"steps": []}, config=config) == {"steps": ["first"]}

    with backend.open_query() as saver:
        resumed = _graph().compile(checkpointer=saver)
        assert resumed.invoke(None, config=config) == {"steps": ["first", "second"]}


@pytest.mark.integration
def test_query_and_ingestion_use_independent_configured_databases(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = CheckpointBackend(settings, busy_timeout_ms=7_500)

    with backend.open_query() as query_saver:
        query_saver.setup()
        query_journal_mode = query_saver.conn.execute("PRAGMA journal_mode").fetchone()[0]
        query_timeout = query_saver.conn.execute("PRAGMA busy_timeout").fetchone()[0]

    with backend.open_ingestion() as ingestion_saver:
        ingestion_saver.setup()
        ingestion_journal_mode = ingestion_saver.conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
        ingestion_timeout = ingestion_saver.conn.execute(
            "PRAGMA busy_timeout"
        ).fetchone()[0]

    assert settings.query_checkpoint_path.is_file()
    assert settings.ingestion_checkpoint_path.is_file()
    assert settings.query_checkpoint_path != settings.ingestion_checkpoint_path
    assert query_journal_mode == ingestion_journal_mode == "wal"
    assert query_timeout == ingestion_timeout == 7_500


@pytest.mark.integration
@pytest.mark.parametrize("worker_field", ("query_worker_count", "ingestion_worker_count"))
def test_checkpoint_backend_rejects_multiple_sqlite_writers(
    tmp_path: Path, worker_field: str
) -> None:
    settings = _settings(tmp_path).model_copy(update={worker_field: 2})

    with pytest.raises(ValueError, match="exactly one worker"):
        CheckpointBackend(settings)
