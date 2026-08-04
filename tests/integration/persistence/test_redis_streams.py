"""Redis Streams integration contracts for the local disposable Redis instance."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from agentic_rag.persistence.redis_queue import RedisStreamsBroker


def _local_redis_dsn() -> str:
    dsn = os.environ.get("AGENTIC_RAG_TEST_REDIS_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_RAG_TEST_REDIS_DSN to run local Redis integration")
    parsed = urlparse(dsn)
    if parsed.scheme not in {"redis", "rediss"} or parsed.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        pytest.skip("AGENTIC_RAG_TEST_REDIS_DSN must target an explicit local Redis")
    return dsn


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_aggregate_delivery_is_acknowledged_after_processing_claim() -> (
    None
):
    """A redelivery retains its aggregate ID until a durable worker claim can fence it."""
    client = Redis.from_url(_local_redis_dsn(), decode_responses=True)
    suffix = uuid4().hex
    stream = f"agenticrag:test:jobs:{suffix}"
    dead_stream = f"agenticrag:test:dead:{suffix}"
    broker = RedisStreamsBroker(client)
    enqueued_at = datetime.now(UTC)
    connected = False

    try:
        try:
            await client.ping()
            connected = True
        except RedisError as error:
            pytest.skip(f"local Redis is unavailable: {error}")

        await broker.publish(stream, "run-1", enqueued_at)
        await broker.publish(stream, "run-1", enqueued_at)
        delivered = await broker.consume(stream, "workers", "worker-a", block_ms=100)
        assert [message.aggregate_id for message in delivered] == ["run-1", "run-1"]

        durable_claims: set[str] = set()
        for message in delivered:
            durable_claims.add(message.aggregate_id)
            await broker.ack(stream, "workers", message.id)
        assert durable_claims == {"run-1"}

        await broker.publish(stream, "run-2", enqueued_at)
        pending = await broker.consume(stream, "workers", "worker-a", block_ms=100)
        reclaimed = await broker.reclaim(stream, "workers", "worker-b", min_idle_ms=0)
        assert [message.aggregate_id for message in pending] == ["run-2"]
        assert [message.aggregate_id for message in reclaimed] == ["run-2"]

        await broker.dead_letter(dead_stream, reclaimed[0], "processing failed")
        await broker.ack(stream, "workers", reclaimed[0].id)
        dead_letters = await client.xrange(dead_stream)
        assert dead_letters[0][1]["aggregate_id"] == "run-2"
    finally:
        if connected:
            await client.delete(stream, dead_stream)
        await client.aclose()
