"""Fail/skip classification for the opt-in real-provider fixture."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures import query_services


def test_mixed_missing_and_explicitly_disabled_provider_configuration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A missing base URL may not hide an explicit Mem0 disablement as a skip."""
    monkeypatch.setenv("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E", "1")
    monkeypatch.setenv("AGENTIC_RAG_TEST_MYSQL_DSN", "mysql+asyncmy://u:p@127.0.0.1:3306/t")
    monkeypatch.setenv("AGENTIC_RAG_TEST_REDIS_DSN", "redis://127.0.0.1:6379/15")
    monkeypatch.setenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", "http://127.0.0.1:9200")
    monkeypatch.setenv("AGENTIC_RAG_MEM0_ENABLED", "false")

    def missing_settings(**kwargs: object) -> object:
        del kwargs
        raise ValueError("deepseek_base_url is missing")

    monkeypatch.setattr(query_services, "Settings", missing_settings)

    with pytest.raises(pytest.fail.Exception, match="invalid DeepSeek/Qwen/Mem0"):
        query_services._require_real_provider_services(tmp_path)


def test_explicit_non_loopback_service_configuration_fails_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A supplied unsafe service endpoint is invalid configuration, not an absent opt-in."""
    monkeypatch.setenv("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E", "1")
    monkeypatch.setenv("AGENTIC_RAG_TEST_MYSQL_DSN", "mysql+asyncmy://u:p@db.example:3306/t")
    monkeypatch.setenv("AGENTIC_RAG_TEST_REDIS_DSN", "redis://127.0.0.1:6379/15")
    monkeypatch.setenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", "http://127.0.0.1:9200")

    with pytest.raises(pytest.fail.Exception, match="local MySQL DSN"):
        query_services._require_real_provider_services(tmp_path)


def test_real_runtime_exposes_the_seeded_parent_for_api_evidence_assertions() -> None:
    runtime = object.__new__(query_services.RealQueryRuntime)
    runtime._parent_id = "seeded-parent"  # type: ignore[attr-defined]

    assert runtime.seeded_parent_id == "seeded-parent"


class _CleanupResult:
    def scalars(self) -> list[str]:
        return []


class _CleanupSession:
    async def execute(self, statement: object) -> _CleanupResult:
        del statement
        return _CleanupResult()


class _CleanupContext:
    async def __aenter__(self) -> _CleanupSession:
        return _CleanupSession()

    async def __aexit__(self, *args: object) -> None:
        del args


class _CleanupFactory:
    def begin(self) -> _CleanupContext:
        return _CleanupContext()


class _Redis:
    async def delete(self, *keys: str) -> None:
        del keys


class _FailingIndices:
    async def delete(self, *, index: str, ignore_unavailable: bool) -> None:
        del index, ignore_unavailable
        raise OSError("elasticsearch cleanup failed")


class _NoopClient:
    async def aclose(self) -> None:
        return None


class _NoopLifespan:
    async def __aexit__(self, *args: object) -> None:
        del args


class _NoopContainerClose:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    async def close(self, *, raise_on_error: bool = False) -> None:
        self.calls.append(raise_on_error)
        return None


@pytest.mark.asyncio
async def test_real_fixture_cleanup_surfaces_elasticsearch_failures() -> None:
    container = SimpleNamespace(
        repositories=SimpleNamespace(session_factory=_CleanupFactory()),
        redis=_Redis(),
        elasticsearch=SimpleNamespace(indices=_FailingIndices()),
    )
    settings = SimpleNamespace(
        default_user_id="fixture-user",
        index_generation="fixture-index",
        mem0_collection="fixture-memory",
    )

    failures: list[tuple[str, BaseException]] = []
    await query_services._cleanup_real_provider_boundaries(
        failures, container, settings, broker=None  # type: ignore[arg-type]
    )
    with pytest.raises(query_services.FixtureTeardownError, match="fixture query index: OSError"):
        query_services._raise_fixture_teardown_failures(failures)


@pytest.mark.asyncio
async def test_shared_real_query_fixture_does_not_suppress_elasticsearch_cleanup_errors() -> None:
    fixture: Any = object.__new__(query_services.RealQueryFixture)
    closer = _NoopContainerClose()
    fixture.settings = SimpleNamespace(index_generation="fixture-index", default_user_id="user")
    fixture.container = SimpleNamespace(
        elasticsearch=SimpleNamespace(indices=_FailingIndices()),
        repositories=SimpleNamespace(session_factory=_CleanupFactory()),
        close=closer.close,
    )
    fixture.client = _NoopClient()
    fixture._app_lifespan = _NoopLifespan()

    async def stop_worker() -> None:
        return None

    fixture.stop_worker = stop_worker

    with pytest.raises(query_services.FixtureTeardownError, match="fixture Elasticsearch index: OSError"):
        await fixture.close()

    assert closer.calls == [True]
