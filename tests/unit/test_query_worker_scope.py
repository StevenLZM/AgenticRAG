"""Scope propagation for the disposable live Query Outbox adapter."""

from __future__ import annotations

import pytest

from scripts import run_query_worker


class _Context:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> None:
        del args


class _Factory:
    def __call__(self) -> _Context:
        return _Context()

    def begin(self) -> _Context:
        return _Context()


class _RecordingRepository:
    calls: list[tuple[object, ...]] = []

    def __init__(self, session: object) -> None:
        del session

    async def list_pending(
        self,
        limit: int,
        *,
        aggregate_type: str | None = None,
        user_id: str | None = None,
        stream_name: str | None = None,
    ) -> list[object]:
        self.calls.append(("list", limit, aggregate_type, user_id, stream_name))
        return []

    async def claim_pending(
        self,
        limit: int,
        *,
        aggregate_type: str | None = None,
        user_id: str | None = None,
        stream_name: str | None = None,
    ) -> list[object]:
        self.calls.append(("claim", limit, aggregate_type, user_id, stream_name))
        return []

    async def mark_dispatched(
        self,
        outbox_id: str,
        *,
        user_id: str | None = None,
        stream_name: str | None = None,
    ) -> None:
        self.calls.append(("mark", outbox_id, user_id, stream_name))

    async def schedule_retry(
        self,
        outbox_id: str,
        *,
        user_id: str | None = None,
        stream_name: str | None = None,
    ) -> None:
        self.calls.append(("retry", outbox_id, user_id, stream_name))


@pytest.mark.asyncio
async def test_transactional_query_outbox_adapter_forwards_an_optional_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingRepository.calls = []
    monkeypatch.setattr(run_query_worker, "SqlAlchemyOutboxRepository", _RecordingRepository)
    adapter = run_query_worker.TransactionalQueryOutboxAdapter(
        _Factory(),  # type: ignore[arg-type]
        user_id="acceptance-user",
        stream_name="agenticrag:e2e:real-query-a1b2:query",
    )

    await adapter.list_pending(2, aggregate_type="query_run")
    await adapter.claim_pending(3, aggregate_type="query_run")
    await adapter.mark_dispatched("outbox-1")
    await adapter.schedule_retry("outbox-1")

    assert _RecordingRepository.calls == [
        (
            "list",
            2,
            "query_run",
            "acceptance-user",
            "agenticrag:e2e:real-query-a1b2:query",
        ),
        (
            "claim",
            3,
            "query_run",
            "acceptance-user",
            "agenticrag:e2e:real-query-a1b2:query",
        ),
        ("mark", "outbox-1", "acceptance-user", "agenticrag:e2e:real-query-a1b2:query"),
        ("retry", "outbox-1", "acceptance-user", "agenticrag:e2e:real-query-a1b2:query"),
    ]


@pytest.mark.asyncio
async def test_default_query_worker_dispatcher_is_fenced_to_the_global_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal worker cannot claim a live acceptance row on a private stream."""
    _RecordingRepository.calls = []
    monkeypatch.setattr(run_query_worker, "SqlAlchemyOutboxRepository", _RecordingRepository)
    adapter = run_query_worker.TransactionalQueryOutboxAdapter(_Factory())  # type: ignore[arg-type]

    await adapter.claim_pending(1, aggregate_type="query_run")

    assert _RecordingRepository.calls == [
        ("claim", 1, "query_run", None, "agenticrag:jobs:query")
    ]


def test_private_acceptance_stream_requires_a_user_scope() -> None:
    """A private stream alone is not sufficient fencing for a shared MySQL worker."""
    with pytest.raises(ValueError, match="private query outbox stream requires a user_id"):
        run_query_worker.TransactionalQueryOutboxAdapter(
            _Factory(),  # type: ignore[arg-type]
            stream_name="agenticrag:e2e:real-query-a1b2:query",
        )
