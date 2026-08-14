"""Query-specific transactional outbox dispatch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from agentic_rag.persistence.outbox import OutboxDispatcher
from agentic_rag.persistence.repositories import OutboxRecord


@dataclass
class FakeBroker:
    published: list[tuple[str, str, str | None]]

    async def publish(
        self,
        stream: str,
        aggregate_id: str,
        enqueued_at: datetime,
        dedupe_key: str | None = None,
    ) -> str:
        del enqueued_at
        self.published.append((stream, aggregate_id, dedupe_key))
        return "1-0"


class FakeOutbox:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.rows = [
            OutboxRecord(
                id="query-outbox-1",
                aggregate_type="query_run",
                aggregate_id="run-1",
                stream_name="agenticrag:jobs:query",
                status="pending",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
            ),
            OutboxRecord(
                id="ingestion-outbox-1",
                aggregate_type="ingestion_job",
                aggregate_id="job-1",
                stream_name="agenticrag:jobs:ingestion",
                status="pending",
                attempt_count=0,
                next_attempt_at=now,
                created_at=now,
            ),
        ]
        self.dispatched: list[str] = []
        self.retries: list[str] = []

    async def claim_pending(
        self, limit: int, *, aggregate_type: str | None = None
    ) -> list[OutboxRecord]:
        return [
            row
            for row in self.rows
            if row.status == "pending"
            and (aggregate_type is None or row.aggregate_type == aggregate_type)
        ][:limit]

    async def mark_dispatched(self, outbox_id: str) -> None:
        self.dispatched.append(outbox_id)

    async def schedule_retry(self, outbox_id: str) -> None:
        self.retries.append(outbox_id)


@pytest.mark.asyncio
async def test_query_dispatch_claims_only_query_rows() -> None:
    outbox = FakeOutbox()
    broker = FakeBroker([])

    dispatcher = OutboxDispatcher(outbox, broker, aggregate_type="query_run")

    assert await dispatcher.dispatch_once() == 1
    assert outbox.dispatched == ["query-outbox-1"]
    assert broker.published == [
        ("agenticrag:jobs:query", "run-1", "query-outbox-1:0")
    ]

