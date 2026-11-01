"""Runnable Query Worker entrypoint contracts."""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import create_async_engine

from scripts.run_query_worker import run
from agentic_rag.config import Settings
from agentic_rag.persistence.elasticsearch import (
    ACTIVE_CHILD_INDEX_ALIAS,
    ChildIndexWriteError,
    ElasticsearchChildIndexStore,
)
from agentic_rag.persistence.repositories import agent_runs, chat_sessions
from agentic_rag.persistence.context_release import context_gate, release_journal
from agentic_rag.runtime.query_composition import build_query_dependencies
from tests.unit.persistence.test_elasticsearch_staging import _Client


def test_query_worker_run_uses_production_composition_by_default() -> None:
    parameter = inspect.signature(run).parameters["dependencies_factory"]
    assert parameter.default is build_query_dependencies


async def test_worker_rejects_legacy_schema_before_alias_or_model_setup(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from sqlalchemy.ext.asyncio import create_async_engine
    import scripts.run_query_worker as entrypoint
    from agentic_rag.persistence.schema_readiness import ChatSchemaUnavailable
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    closed = []

    async def close():
        closed.append(True)
        await engine.dispose()

    async def unexpected(*args):
        pytest.fail("schema must be checked before alias/model setup")

    container = SimpleNamespace(mysql_engine=engine, close=close)
    monkeypatch.setattr(entrypoint, "build_container", lambda _: container)
    monkeypatch.setattr(entrypoint, "ensure_active_child_alias", unexpected)
    with pytest.raises(ChatSchemaUnavailable):
        await entrypoint.run(SimpleNamespace(), dependencies_factory=unexpected)
    assert closed == [True]


async def _startup_schema(engine, *, execution_fence):
    schema = MetaData()
    for table in (agent_runs, chat_sessions):
        table.to_metadata(schema)
    for name in ("0010_context_builds", "0011_context_release"):
        spec = importlib.util.spec_from_file_location(
            name, Path(f"alembic/versions/{name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for table in module.metadata.tables.values():
            if execution_fence and table.name in {
                "context_release_gate", "context_release_journal"
            }:
                continue
            table.to_metadata(schema)
    if execution_fence:
        for table in (context_gate, release_journal):
            table.to_metadata(schema)
    async with engine.begin() as connection:
        await connection.run_sync(schema.create_all)


def _settings(**kwargs):
    return Settings(
        mysql_dsn="sqlite+aiosqlite:///:memory:",
        deepseek_base_url="http://localhost/v1",
        qwen_embedding_base_url="http://localhost/v1",
        index_generation="index-v2",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("execution_fence", "alias_target", "error", "message"),
    [
        (False, "agenticrag-children-index-v2", ValueError, "0012"),
        (False, None, ChildIndexWriteError, "alias"),
        (True, None, ChildIndexWriteError, "alias"),
        (True, "agenticrag-children-other", ChildIndexWriteError, "alias"),
    ],
    ids=["pre0012", "pre0012-missing-alias", "missing-alias", "wrong-alias"],
)
async def test_controlled_startup_rejects_without_index_or_alias_mutations(
    monkeypatch, execution_fence, alias_target, error, message
):
    import scripts.run_query_worker as entrypoint

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await _startup_schema(engine, execution_fence=execution_fence)
    client = _Client()
    await ElasticsearchChildIndexStore(client).ensure_index("index-v2")
    # Fixture setup is complete; only real startup mutations count below.
    client.indices.create_calls = 0
    if alias_target:
        client.indices.aliases[ACTIVE_CHILD_INDEX_ALIAS] = {alias_target}
    before_aliases = {key: set(value) for key, value in client.indices.aliases.items()}
    settings = _settings(context_release_enabled=True, context_instance_id="query-1")
    closed = []
    dependencies_loaded = []

    async def close():
        closed.append(True)
        await engine.dispose()

    async def dependencies_factory(*_):
        dependencies_loaded.append(True)
        return SimpleNamespace(concurrency=None)

    async def close_dependencies(_):
        pass

    container = SimpleNamespace(
        mysql_engine=engine, elasticsearch=client, settings=settings, close=close
    )
    monkeypatch.setattr(entrypoint, "build_container", lambda _: container)
    monkeypatch.setattr(entrypoint, "close_query_dependencies", close_dependencies)
    try:
        with pytest.raises(error, match=message):
            await entrypoint.run(settings, dependencies_factory=dependencies_factory)
    finally:
        assert closed == [True]
        assert client.indices.create_calls == 0
        assert client.indices.put_alias_calls == []
        assert client.indices.alias_updates == []
        assert client.indices.aliases == before_aliases
    if alias_target != "agenticrag-children-index-v2":
        assert dependencies_loaded == []


async def test_legacy_startup_still_repairs_missing_alias_before_dependencies(monkeypatch):
    import scripts.run_query_worker as entrypoint

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await _startup_schema(engine, execution_fence=False)
    client = _Client()
    await ElasticsearchChildIndexStore(client).ensure_index("index-v2")
    settings = _settings(context_release_enabled=False)

    async def close():
        await engine.dispose()

    async def dependencies_factory(*_):
        assert client.indices.aliases == {
            ACTIVE_CHILD_INDEX_ALIAS: {"agenticrag-children-index-v2"}
        }
        raise RuntimeError("stop before model construction")

    container = SimpleNamespace(mysql_engine=engine, elasticsearch=client, close=close)
    monkeypatch.setattr(entrypoint, "build_container", lambda _: container)
    with pytest.raises(RuntimeError, match="stop before model construction"):
        await entrypoint.run(settings, dependencies_factory=dependencies_factory)
    assert client.indices.put_alias_calls == [
        {"index": "agenticrag-children-index-v2", "name": ACTIVE_CHILD_INDEX_ALIAS}
    ]
    assert client.indices.alias_updates == []
