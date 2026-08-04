"""Unit contracts for Redis Streams response normalization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from redis.asyncio import Redis

from agentic_rag.persistence.redis_queue import RedisStreamsBroker


class FakeRedis:
    def __init__(self, fields: dict[str | bytes, str | bytes]) -> None:
        self._fields = fields

    async def xgroup_create(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def xreadgroup(self, *args: Any, **kwargs: Any) -> list[Any]:
        return [("jobs", [("1-0", self._fields)])]

    async def xautoclaim(self, *args: Any, **kwargs: Any) -> list[Any]:
        return ["0-0", [("1-0", self._fields)], []]


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
