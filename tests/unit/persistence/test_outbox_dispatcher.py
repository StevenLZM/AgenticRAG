"""Unit contracts for transactional outbox dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from agentic_rag.persistence.outbox import OutboxDispatcher


@dataclass
class FakeOutboxRow:
    id: str
    aggregate_id: str
    stream_name: str
    created_at: datetime
    status: str = "pending"
    attempt_count: int = 0


class FakeOutbox:
    def __init__(self) -> None:
        self.rows: list[FakeOutboxRow] = []
        self.retries: list[str] = []

    def pending(self, aggregate_id: str, stream_name: str) -> FakeOutboxRow:
        row = FakeOutboxRow(
            id=f"outbox-{aggregate_id}",
            aggregate_id=aggregate_id,
            stream_name=stream_name,
            created_at=datetime.now(UTC),
        )
        self.rows.append(row)
        return row

    async def claim_pending(self, limit: int) -> list[FakeOutboxRow]:
        return [row for row in self.rows if row.status == "pending"][:limit]

    async def mark_dispatched(self, outbox_id: str) -> None:
        next(row for row in self.rows if row.id == outbox_id).status = "dispatched"

    async def schedule_retry(self, outbox_id: str) -> None:
        self.retries.append(outbox_id)


class FakeBroker:
    def __init__(self) -> None:
        self.fail_once = False
        self.published: list[tuple[str, str, datetime]] = []

    async def publish(
        self,
        stream: str,
        aggregate_id: str,
        enqueued_at: datetime,
        dedupe_key: str | None = None,
    ) -> str:
        if self.fail_once:
            self.fail_once = False
            from redis.exceptions import RedisError

            raise RedisError("unavailable")
        self.published.append((stream, aggregate_id, enqueued_at))
        return "1-0"


@pytest.mark.asyncio
async def test_dispatcher_marks_outbox_only_after_publish() -> None:
    """A publish failure leaves the outbox pending for a later retry."""
    outbox = FakeOutbox()
    broker = FakeBroker()
    row = outbox.pending("run-1", "agenticrag:jobs:query")
    broker.fail_once = True

    dispatcher = OutboxDispatcher(outbox, broker)

    assert await dispatcher.dispatch_once() == 0
    assert row.status == "pending"
    assert outbox.retries == [row.id]

    assert await dispatcher.dispatch_once() == 1
    assert row.status == "dispatched"
    assert broker.published == [(row.stream_name, row.aggregate_id, row.created_at)]
