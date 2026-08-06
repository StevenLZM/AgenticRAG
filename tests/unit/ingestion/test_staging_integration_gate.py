"""Regression tests for the opt-in staging integration-test gate."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tests.integration.ingestion import test_staging_index


class _FailingConnection:
    async def __aenter__(self) -> Any:
        raise OSError("configured MySQL is unavailable")

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FailingEngine:
    disposed = False

    def connect(self) -> _FailingConnection:
        return _FailingConnection()

    async def dispose(self) -> None:
        self.disposed = True


class _Elasticsearch:
    closed = False

    async def info(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_configured_infrastructure_failure_is_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _FailingEngine()
    elasticsearch = _Elasticsearch()
    monkeypatch.setenv("AGENTIC_RAG_TEST_MYSQL_DSN", "mysql+asyncmy://configured")
    monkeypatch.setenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", "http://configured:9200")
    monkeypatch.setattr(test_staging_index, "create_mysql_engine", lambda *_a, **_k: engine)
    monkeypatch.setattr(test_staging_index, "AsyncElasticsearch", lambda *_a, **_k: elasticsearch)

    fixture = cast(Any, test_staging_index.infrastructure).__wrapped__()
    try:
        with pytest.raises(OSError, match="configured MySQL"):
            await anext(fixture)
    except pytest.skip.Exception:
        pytest.fail("configured infrastructure errors must fail, not skip")

    assert engine.disposed is True
    assert elasticsearch.closed is True
