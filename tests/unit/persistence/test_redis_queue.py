"""Unit contracts for Redis Streams response normalization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from redis.asyncio import Redis

from agentic_rag.persistence.redis_queue import RedisStreamsBroker, StreamMessage


class FakeRedis:
    def __init__(self, fields: dict[str | bytes, str | bytes]) -> None:
        self._fields = fields
        self.eval_calls: list[tuple[str, int, tuple[Any, ...]]] = []

    async def xgroup_create(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def xreadgroup(self, *args: Any, **kwargs: Any) -> list[Any]:
        return [("jobs", [("1-0", self._fields)])]

    async def xautoclaim(self, *args: Any, **kwargs: Any) -> list[Any]:
        return ["0-0", [("1-0", self._fields)], []]

    async def eval(self, script: str, numkeys: int, *args: Any) -> bytes:
        self.eval_calls.append((script, numkeys, args))
        return b"7-0"

    async def xadd(self, *args: Any, **kwargs: Any) -> bytes:
        return b"8-0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"aggregate_id": "run-1", "enqueued_at": "2026-08-05T00:00:00+00:00"},
        {
            b"aggregate_id": b"run-1",
            b"enqueued_at": b"2026-08-05T00:00:00+00:00",
        },
    ],
)
async def test_broker_normalizes_decoded_and_byte_stream_fields(
    fields: dict[str | bytes, str | bytes],
) -> None:
    """redis-py's default byte responses are equivalent to decoded responses."""
    broker = RedisStreamsBroker(cast(Redis, FakeRedis(fields)))
    expected = ("run-1", datetime(2026, 8, 5, tzinfo=UTC))

    delivered = await broker.consume("jobs", "workers", "worker-a", block_ms=0)
    reclaimed = await broker.reclaim("jobs", "workers", "worker-b", min_idle_ms=0)

    assert [(message.aggregate_id, message.enqueued_at) for message in delivered] == [
        expected
    ]
    assert [(message.aggregate_id, message.enqueued_at) for message in reclaimed] == [
        expected
    ]


@pytest.mark.asyncio
async def test_publish_uses_atomic_aggregate_deduplication() -> None:
    client = FakeRedis({})
    broker = RedisStreamsBroker(cast(Redis, client))

    first = await broker.publish(
        "jobs", "job-1", datetime(2026, 8, 5, tzinfo=UTC), dedupe_key="outbox-1:0"
    )
    second = await broker.publish(
        "jobs", "job-1", datetime(2026, 8, 5, tzinfo=UTC), dedupe_key="outbox-1:0"
    )

    assert first == second == "7-0"
    assert all(call[1] == 2 for call in client.eval_calls)
    assert all(
        call[2][:3] == ("jobs:dedupe", "jobs", "outbox-1:0")
        for call in client.eval_calls
    )


@pytest.mark.asyncio
async def test_dead_letter_is_idempotent_for_one_source_message() -> None:
    client = FakeRedis({})
    broker = RedisStreamsBroker(cast(Redis, client))
    message = StreamMessage(
        id="1-0",
        aggregate_id="job-1",
        enqueued_at=datetime(2026, 8, 5, tzinfo=UTC),
    )

    await broker.dead_letter("jobs:dead", message, "failed")
    await broker.dead_letter("jobs:dead", message, "failed")

    assert len(client.eval_calls) == 2
    assert all(
        call[2][:3] == ("jobs:dead:dedupe", "jobs:dead", "1-0")
        for call in client.eval_calls
    )
