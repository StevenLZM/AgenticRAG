"""Pure fail-closed gates for the live console acceptance report."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scripts.run_real_query_acceptance import (
    AcceptanceTeardownError,
    _cleanup_durable_boundaries,
    _invalidate_previous_summary,
    _raise_teardown_failures,
    _record_teardown_error,
    console_acceptance_passed,
    run,
)
from scripts.real_acceptance_evidence import (
    RealAcceptanceEvidenceError,
    exercise_query_worker_recovery,
    require_service_backup_admin_dsn,
    service_backup_configuration_issue,
)


def _summary() -> dict[str, object]:
    return {
        "console_page_status": 200,
        "sse_last_event_id": 7,
        "client_provenance": "real_query_api",
        "runtime_config_snapshot_id": "snapshot-current",
        "console_run_snapshot_id": "snapshot-current",
        "memory_provider_available": True,
        "memory_boundary": {"read": True, "write": True},
        "degradation_events": [
            {
                "event_type": "CIRCUIT_OPEN",
                "attributes": {
                    "attempt": 1,
                    "component": "retrieval",
                    "outcome": "degraded",
                    "reason": "circuit_open",
                    "retryable": True,
                },
            }
        ],
        "citation_coverage": 1.0,
        "user_leak_count": 0,
        "unaudited_answer_count": 0,
        "recovery_drill_passed": True,
        "backup_restore_passed": True,
        "recovery_evidence": {
            "provider_e2e": True,
            "source_run_id": "run-recovery",
            "runtime_config_snapshot_id": "snapshot-current",
            "worker_restarted": True,
            "duplicate_delivery_injected": True,
            "duplicate_delivery_acked": True,
            "terminal_status": "completed",
            "terminal_event_count": 1,
            "final_answer_audited": True,
        },
        "backup_restore_evidence": {
            "provider_e2e": True,
            "source_run_id": "run-recovery",
            "runtime_config_snapshot_id": "snapshot-current",
            "source_scope": {
                "mysql_database": "agentic_rag_acceptance_source",
                "redis_prefix": "agenticrag:e2e:source",
                "index_generation": "real-query-source",
                "checkpoint_thread_id": "query:user:recovery",
                "artifact_marker": "artifact://acceptance/run-recovery.json",
            },
            "restore_scope": {
                "mysql_database": "agentic_rag_acceptance_restore",
                "redis_prefix": "agenticrag:e2e:restore",
                "index_generation": "real-query-restore",
            },
            "services": {
                "mysql": True,
                "redis": True,
                "elasticsearch": True,
                "artifacts": True,
                "checkpoints": True,
            },
        },
    }


def test_console_acceptance_gate_rejects_missing_or_unsafe_live_evidence() -> None:
    summary = _summary()

    assert console_acceptance_passed(summary) is True
    for field, value in (
        ("console_page_status", 202),
        ("sse_last_event_id", 0),
        ("client_provenance", "fixture"),
        ("console_run_snapshot_id", "snapshot-old"),
        ("memory_provider_available", False),
        ("citation_coverage", 0.99),
        ("user_leak_count", 1),
        ("unaudited_answer_count", 1),
        ("recovery_drill_passed", False),
        ("backup_restore_passed", False),
    ):
        altered = {**summary, field: value}
        assert console_acceptance_passed(altered) is False

    # Bare booleans from local-only drills can never satisfy a live gate.
    assert console_acceptance_passed({
        key: value
        for key, value in summary.items()
        if key not in {"recovery_evidence", "backup_restore_evidence"}
    }) is False

    bad_recovery = {
        **summary,
        "recovery_evidence": {
            **summary["recovery_evidence"],  # type: ignore[arg-type]
            "terminal_event_count": 2,
        },
    }
    assert console_acceptance_passed(bad_recovery) is False

    bad_backup = {
        **summary,
        "backup_restore_evidence": {
            **summary["backup_restore_evidence"],  # type: ignore[arg-type]
            "services": {
                "mysql": True,
                "redis": True,
                "elasticsearch": True,
                "artifacts": True,
                "checkpoints": False,
            },
        },
    }
    assert console_acceptance_passed(bad_backup) is False

    unsafe_event = {
        **summary,
        "degradation_events": [
            {
                "event_type": "CIRCUIT_OPEN",
                "attributes": {
                    "attempt": 1,
                    "component": "retrieval",
                    "outcome": "degraded",
                    "prompt": "raw prompt must not be accepted",
                    "reason": "circuit_open",
                    "retryable": True,
                },
            }
        ],
    }
    assert console_acceptance_passed(unsafe_event) is False


def test_invalidate_previous_summary_removes_stale_pass_before_a_live_attempt(tmp_path) -> None:
    """A failed rerun must not leave a previously generated PASS as current."""
    output = tmp_path / "real-query"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text('{"gate_passed": true}', encoding="utf-8")

    _invalidate_previous_summary(output)

    assert not summary.exists()
    stale = list(output.glob("summary.stale-*.json"))
    assert len(stale) == 1
    assert stale[0].read_text(encoding="utf-8") == '{"gate_passed": true}'


def test_real_backup_configuration_is_explicitly_unavailable_without_isolation_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE", raising=False)
    monkeypatch.delenv("AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN", raising=False)

    assert "AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE" in str(
        service_backup_configuration_issue()
    )
    with pytest.raises(RealAcceptanceEvidenceError, match="unavailable"):
        require_service_backup_admin_dsn()


@pytest.mark.asyncio
async def test_recovery_evidence_comes_from_current_api_worker_and_duplicate_ack() -> None:
    calls: list[str] = []

    class Response:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    class Client:
        async def post(self, path: str, *, json: dict[str, object]) -> Response:
            assert path == "/v1/query-runs"
            assert json["wait_seconds"] == 0
            calls.append("api-create")
            return Response(202, {"run_id": "run-recovery"})

        async def get(self, path: str) -> Response:
            assert path == "/v1/query-runs/run-recovery"
            return Response(200, {
                "status": "completed",
                "runtime_config_snapshot_id": "snapshot-current",
                "answer": {
                    "audited": True,
                    "evidence_parent_ids": ["parent-1"],
                },
            })

    class Broker:
        query_stream = "agenticrag:e2e:test:query"
        query_group = "agenticrag-e2e-test"

        def __init__(self) -> None:
            self.messages = 0

        async def publish(self, *args: object, **kwargs: object) -> str:
            del args, kwargs
            self.messages += 1
            calls.append(f"publish-{self.messages}")
            return f"{self.messages}-0"

    class Redis:
        async def xinfo_groups(self, stream: str) -> list[dict[str, object]]:
            assert stream == Broker.query_stream
            return [{
                "name": Broker.query_group,
                "last-delivered-id": "2-0",
                "pending": 0,
            }]

    class Events:
        async def list_after(self, *args: object) -> list[object]:
            del args
            return [SimpleNamespace(event_type="RUN_COMPLETED")]

    async def stop_worker() -> None:
        calls.append("worker-stop")

    async def start_worker() -> None:
        calls.append("worker-start")

    evidence, checkpoint_thread_id = await exercise_query_worker_recovery(
        client=Client(),
        container=SimpleNamespace(redis=Redis(), event_repository=Events()),
        broker=Broker(),
        user_id="user-1",
        snapshot_id="snapshot-current",
        parent_id="parent-1",
        stop_worker=stop_worker,
        start_worker=start_worker,
    )

    assert calls[:4] == ["worker-stop", "api-create", "publish-1", "worker-start"]
    assert evidence["source_run_id"] == "run-recovery"
    assert evidence["duplicate_delivery_acked"] is True
    assert evidence["terminal_event_count"] == 1
    assert checkpoint_thread_id.startswith("query:user-1:recovery-")


@pytest.mark.asyncio
async def test_live_setup_failure_cannot_leave_a_current_pass_report(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "real-query"
    output.mkdir()
    (output / "summary.json").write_text('{"gate_passed": true}', encoding="utf-8")
    monkeypatch.delenv("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E", raising=False)

    with pytest.raises(RuntimeError, match="AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E"):
        await run(output)

    assert not (output / "summary.json").exists()
    assert len(list(output.glob("summary.stale-*.json"))) == 1


@pytest.mark.asyncio
async def test_teardown_failure_is_a_fail_closed_acceptance_gate() -> None:
    async def broken_close() -> None:
        raise OSError("test close failure")

    failures: list[tuple[str, BaseException]] = []
    await _record_teardown_error(failures, "memory client", broken_close())

    with pytest.raises(AcceptanceTeardownError, match="memory client: OSError"):
        _raise_teardown_failures(failures)


@pytest.mark.asyncio
async def test_teardown_continues_after_cancellation_and_reraises_it_after_cleanup() -> None:
    """Cancellation must not strand later owned acceptance boundaries."""
    completed: list[str] = []
    failures: list[tuple[str, BaseException]] = []

    async def cancelled_worker() -> None:
        raise asyncio.CancelledError()

    async def close_http_client() -> None:
        completed.append("http client")

    await _record_teardown_error(failures, "Query Worker", cancelled_worker())
    await _record_teardown_error(failures, "Query API client", close_http_client())

    assert completed == ["http client"]
    with pytest.raises(asyncio.CancelledError):
        _raise_teardown_failures(failures)


class _BrokenMysqlContext:
    async def __aenter__(self) -> object:
        raise OSError("mysql cleanup failed")

    async def __aexit__(self, *args: object) -> None:
        del args


class _BrokenMysqlFactory:
    def begin(self) -> _BrokenMysqlContext:
        return _BrokenMysqlContext()


class _RecordingRedis:
    def __init__(self) -> None:
        self.keys: tuple[str, ...] = ()

    async def delete(self, *keys: str) -> None:
        self.keys = keys


class _RecordingIndices:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, *, index: str, ignore_unavailable: bool) -> None:
        assert ignore_unavailable is True
        self.deleted.append(index)


@pytest.mark.asyncio
async def test_durable_cleanup_continues_after_mysql_failure() -> None:
    """One failed durable boundary cannot strand Redis or either isolated index."""
    redis = _RecordingRedis()
    indices = _RecordingIndices()
    container = SimpleNamespace(
        repositories=SimpleNamespace(session_factory=_BrokenMysqlFactory()),
        redis=redis,
        elasticsearch=SimpleNamespace(indices=indices),
    )
    settings = SimpleNamespace(
        default_user_id="acceptance-user",
        index_generation="acceptance-index",
        mem0_collection="acceptance-memory",
    )
    broker = SimpleNamespace(cleanup_keys=("private-query", "private-dead"))
    failures: list[tuple[str, BaseException]] = []

    await _cleanup_durable_boundaries(
        failures,
        container=container,  # type: ignore[arg-type]
        settings=settings,  # type: ignore[arg-type]
        broker=broker,  # type: ignore[arg-type]
    )

    assert redis.keys == ("private-query", "private-dead")
    assert indices.deleted == [
        "agenticrag-children-acceptance-index",
        "acceptance-memory",
    ]
    with pytest.raises(AcceptanceTeardownError, match="acceptance MySQL state: OSError"):
        _raise_teardown_failures(failures)
