"""Fail/skip classification for the opt-in real-provider fixture."""

from __future__ import annotations

import asyncio
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


def test_seeded_runtime_fixture_labels_its_summary_as_contract_only() -> None:
    """A direct seed or deterministic embedding cannot become a quality score."""
    summary = query_services._contract_only_fixture_summary("fixture-snapshot")

    assert summary["fixture_kind"] == "contract_only"
    assert summary["evaluation_mode"] == "contract"
    assert summary["client_provenance"] == "contract_fixture"
    assert summary["quality_measurement"] is False
    assert summary["real_query_count"] == 0
    assert summary["completed_cases"] == 0
    assert summary["requested_cases"] == 0


class _CleanupResult:
    def __init__(self, values: list[str] | None = None) -> None:
        self._values = values or []

    def scalars(self) -> list[str]:
        return self._values


class _CleanupSession:
    def __init__(self) -> None:
        self.statements: list[object] = []

    async def execute(self, statement: object) -> _CleanupResult:
        self.statements.append(statement)
        return _CleanupResult(["owned-run"]) if getattr(statement, "is_select", False) else _CleanupResult()


class _CleanupContext:
    def __init__(self, session: _CleanupSession | None = None) -> None:
        self.session = session or _CleanupSession()

    async def __aenter__(self) -> _CleanupSession:
        return self.session

    async def __aexit__(self, *args: object) -> None:
        del args


class _CleanupFactory:
    def __init__(self) -> None:
        self.session = _CleanupSession()

    def begin(self) -> _CleanupContext:
        return _CleanupContext(self.session)


class _Redis:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, ...]] = []

    async def delete(self, *keys: str) -> None:
        self.deleted.append(keys)


@pytest.mark.asyncio
async def test_shared_fixture_cleanup_keeps_foreign_mysql_rows_and_redis_keys() -> None:
    """Cleanup predicates must be fixture-owned even when services are shared."""
    factory = _CleanupFactory()
    redis = _Redis()
    container = SimpleNamespace(
        repositories=SimpleNamespace(session_factory=factory),
        redis=redis,
    )
    settings = SimpleNamespace(default_user_id="fixture-user")
    broker = SimpleNamespace(
        query_stream="agenticrag:e2e:fixture-a:query",
        cleanup_keys=(
            "agenticrag:e2e:fixture-a:query",
            "agenticrag:e2e:fixture-a:query:dedupe",
            "agenticrag:e2e:fixture-a:query:dead",
            "agenticrag:e2e:fixture-a:query:dead:dedupe",
        ),
    )

    await query_services._cleanup_local_fixture_mysql(
        container, settings, broker  # type: ignore[arg-type]
    )
    await query_services._cleanup_local_fixture_redis(
        container, broker  # type: ignore[arg-type]
    )

    statements = factory.session.statements
    outbox_delete = next(
        statement
        for statement in statements
        if getattr(getattr(statement, "table", None), "name", None) == "task_outbox"
    )
    outbox_sql = str(outbox_delete)
    assert "task_outbox.aggregate_id IN" in outbox_sql
    assert "task_outbox.stream_name" in outbox_sql
    for statement in statements:
        table_name = getattr(getattr(statement, "table", None), "name", None)
        if table_name in {"agent_events", "agent_runs", "parent_chunks", "documents"}:
            assert f"{table_name}.user_id" in str(statement)
    assert redis.deleted == [broker.cleanup_keys]
    assert all("agenticrag:jobs:query" not in key for key in redis.deleted[0])


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


class _TrackingCleanupContext(_CleanupContext):
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self) -> _CleanupSession:
        self._events.append("mysql")
        return await super().__aenter__()


class _TrackingCleanupFactory:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def begin(self) -> _TrackingCleanupContext:
        return _TrackingCleanupContext(self._events)


class _TrackingRedis:
    def __init__(self, events: list[str], *, error: BaseException | None = None) -> None:
        self._events = events
        self._error = error

    async def delete(self, *keys: str) -> None:
        assert keys
        self._events.append("redis")
        if self._error is not None:
            raise self._error


class _TrackingIndices:
    def __init__(self, events: list[str], *, failing_index: str | None = None) -> None:
        self._events = events
        self._failing_index = failing_index

    async def delete(self, *, index: str, ignore_unavailable: bool) -> None:
        assert ignore_unavailable is True
        self._events.append(f"elasticsearch:{index}")
        if index == self._failing_index:
            raise OSError("elasticsearch cleanup failed")


class _TrackingCheckpointContext:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aexit__(self, *args: object) -> None:
        del args
        self._events.append("checkpoint")


class _TrackingContainerClose:
    def __init__(self, events: list[str], *, error: BaseException | None = None) -> None:
        self._events = events
        self._error = error

    async def close(self, *, raise_on_error: bool = False) -> None:
        self._events.append(f"container:{raise_on_error}")
        if self._error is not None:
            raise self._error


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
        redis=_Redis(),
        close=closer.close,
    )
    fixture.client = _NoopClient()
    fixture._app_lifespan = _NoopLifespan()
    fixture._isolated_broker = SimpleNamespace(
        query_stream="agenticrag:e2e:fixture:query",
        cleanup_keys=("agenticrag:e2e:fixture:query",),
    )

    async def stop_worker() -> None:
        return None

    fixture.stop_worker = stop_worker

    with pytest.raises(query_services.FixtureTeardownError, match="fixture Elasticsearch index: OSError"):
        await fixture.close()

    assert closer.calls == [True]


@pytest.mark.asyncio
async def test_shared_fixture_worker_failure_still_closes_checkpoint_mysql_and_redis() -> None:
    events: list[str] = []
    fixture: Any = object.__new__(query_services.RealQueryFixture)
    closer = _TrackingContainerClose(events)
    fixture.settings = SimpleNamespace(
        index_generation="fixture-index",
        default_user_id="fixture-user",
    )
    fixture.container = SimpleNamespace(
        elasticsearch=SimpleNamespace(indices=_TrackingIndices(events)),
        repositories=SimpleNamespace(session_factory=_TrackingCleanupFactory(events)),
        redis=_TrackingRedis(events),
        close=closer.close,
    )
    fixture.client = _NoopClient()
    fixture._app_lifespan = _NoopLifespan()
    fixture._stop = asyncio.Event()
    fixture._checkpoint_context = _TrackingCheckpointContext(events)
    fixture._isolated_broker = SimpleNamespace(
        query_stream="agenticrag:e2e:fixture:query",
        cleanup_keys=("agenticrag:e2e:fixture:query",),
    )

    async def fail_worker() -> None:
        raise RuntimeError("worker task failed")

    fixture._worker_task = asyncio.create_task(fail_worker())

    with pytest.raises(query_services.FixtureTeardownError, match="Query Worker: RuntimeError"):
        await fixture.close()

    assert {"checkpoint", "mysql", "redis", "container:True"} <= set(events)


@pytest.mark.asyncio
async def test_real_runtime_es_failure_still_cleans_redis_and_aggregates_close_errors() -> None:
    events: list[str] = []
    runtime: Any = object.__new__(query_services.RealQueryRuntime)
    closer = _TrackingContainerClose(events, error=RuntimeError("container close failed"))
    query_index = "agenticrag-children-runtime-index"
    runtime.settings = SimpleNamespace(
        index_generation="runtime-index",
        default_user_id="runtime-user",
        mem0_collection="runtime-memory",
    )
    runtime.container = SimpleNamespace(
        elasticsearch=SimpleNamespace(
            indices=_TrackingIndices(events, failing_index=query_index)
        ),
        repositories=SimpleNamespace(session_factory=_TrackingCleanupFactory(events)),
        redis=_TrackingRedis(events, error=ConnectionError("redis cleanup failed")),
        close=closer.close,
    )
    runtime.client = _NoopClient()
    runtime._app_lifespan = _NoopLifespan()
    runtime._checkpoint_context = None
    runtime._dependencies = None
    runtime._isolated_broker = SimpleNamespace(cleanup_keys=("runtime-query",))

    async def stop_worker() -> None:
        return None

    runtime.stop_worker = stop_worker

    with pytest.raises(query_services.FixtureTeardownError) as caught:
        await runtime.close()

    message = str(caught.value)
    assert "fixture query index: OSError" in message
    assert "fixture Redis state: ConnectionError" in message
    assert "fixture container: RuntimeError" in message
    assert events.index(f"elasticsearch:{query_index}") < events.index("redis")
    assert events.index("redis") < events.index("container:True")
