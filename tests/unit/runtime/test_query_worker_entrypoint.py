"""Runnable Query Worker entrypoint contracts."""

from __future__ import annotations

import inspect

from scripts.run_query_worker import run
from agentic_rag.runtime.query_composition import build_query_dependencies


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
