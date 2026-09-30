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
async def test_research_action_resumes_after_sqlite_reopen_without_repeating_plan(tmp_path):
    from dataclasses import replace
    from agentic_rag.query.graph import build_query_graph, query_checkpoint_config
    from tests.unit.query.test_graph import _deps
    from tests.unit.query.test_research_dag import loop
    from tests.unit.query.test_research_loop import _state_without_research_todos

    deps, _memory, retrieval, _events = _deps(route="research", grades=["sufficient"])
    agent = loop([
        {"action": "create_todos", "items": [
            {"key": "a", "title": "notice"},
            {"key": "b", "title": "check notice", "blocked_by": ["a"]},
        ]},
        {"action": "retrieve_evidence", "todo_id": "todo-1", "query": "notice"},
        {"action": "retrieve_evidence", "todo_id": "todo-2", "query": "check notice"},
        {"action": "submit_evidence"},
    ], retrieval)
    backend = CheckpointBackend(_settings(tmp_path))
    state = _state_without_research_todos()
    config = query_checkpoint_config(state)
    state["runtime_config_snapshot"]["max_research_rounds"] = 4
    async with backend.open_query() as saver:
        graph = build_query_graph(replace(deps, research_loop=agent), checkpointer=saver)
        await graph.ainvoke(state, config=config, interrupt_after=["research_agent_loop"])
        saved = await graph.aget_state(config)
        assert saved.values["research_attempt_count"] == 1
        assert saved.values["research"]["todos"][1]["blocked_by"] == ["todo-1"]
        assert saved.next == ("research_agent_loop",)
    async with backend.open_query() as saver:
        graph = build_query_graph(replace(deps, research_loop=agent), checkpointer=saver)
        result = await graph.ainvoke(None, config=config)
        assert result["research_attempt_count"] == 4
        assert result["research"]["submitted"] is True
        assert len(result["research"]["todos"]) == 2
        assert retrieval.calls == 2


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
