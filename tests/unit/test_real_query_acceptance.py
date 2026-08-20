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
