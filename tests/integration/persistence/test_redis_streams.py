"""Redis Streams integration contracts for the local disposable Redis instance."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from agentic_rag.persistence.redis_queue import RedisStreamsBroker, StreamMessage
from agentic_rag.persistence.outbox import OutboxDispatcher
from agentic_rag.persistence.repositories import OutboxRecord


class _FailFirstMarkOutbox:
    def __init__(self) -> None:
        self.fail_mark = True
        self.marked: list[str] = []

    async def list_pending(self, limit: int) -> list[OutboxRecord]:
        raise AssertionError("redispatch does not list rows")

    async def claim_pending(self, limit: int) -> list[OutboxRecord]:
        raise AssertionError("redispatch does not claim rows")

    async def mark_dispatched(self, outbox_id: str) -> None:
        if self.fail_mark:
            self.fail_mark = False
            raise RuntimeError("injected SQL mark failure")
        self.marked.append(outbox_id)

    async def schedule_retry(self, outbox_id: str) -> None:
        raise AssertionError("redispatch does not schedule rows")


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
async def test_duplicate_aggregate_delivery_retains_the_aggregate_id() -> None:
    """A worker can use the stable aggregate ID to fence duplicate deliveries."""
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

        for message in delivered:
            await broker.ack(stream, "workers", message.id)

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_lua_dedupes_retry_and_allows_new_reclaim_generation() -> None:
    """One delivery generation has one entry even if SQL marking fails."""
    client = Redis.from_url(_local_redis_dsn(), decode_responses=True)
    suffix = uuid4().hex
    stream = f"agenticrag:test:dedupe:{suffix}"
    broker = RedisStreamsBroker(client)
    outbox = _FailFirstMarkOutbox()
    dispatcher = OutboxDispatcher(outbox, broker)
    created_at = datetime.now(UTC)
    first_generation = OutboxRecord(
        id=f"outbox-{suffix}",
        aggregate_type="ingestion_job",
        aggregate_id=f"job-{suffix}",
        stream_name=stream,
        status="pending",
        attempt_count=0,
        next_attempt_at=created_at,
        created_at=created_at,
    )

    try:
        # A configured but unavailable Redis is an integration failure, not a skip.
        await client.ping()
        with pytest.raises(RuntimeError, match="SQL mark failure"):
            await dispatcher.redispatch(first_generation)
        await dispatcher.redispatch(first_generation)

        entries = await client.xrange(stream)
        assert len(entries) == 1
        assert entries[0][1]["aggregate_id"] == first_generation.aggregate_id
        assert outbox.marked == [first_generation.id]

        reclaimed_generation = replace(first_generation, attempt_count=1)
        await dispatcher.redispatch(reclaimed_generation)
        entries = await client.xrange(stream)
        assert len(entries) == 2
        assert [fields["aggregate_id"] for _, fields in entries] == [
            first_generation.aggregate_id,
            first_generation.aggregate_id,
        ]
    finally:
        await client.delete(stream, f"{stream}:dedupe")
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_dead_letter_publish_is_idempotent_per_attempt_generation() -> None:
    """A crash between XADD and the SQL marker cannot duplicate the DLQ record."""
    client = Redis.from_url(_local_redis_dsn(), decode_responses=True)
    suffix = uuid4().hex
    dead_stream = f"agenticrag:test:dead-once:{suffix}"
    broker = RedisStreamsBroker(client)
    message = StreamMessage(
        id="1-0",
        aggregate_id=f"job-{suffix}",
        enqueued_at=datetime.now(UTC),
    )
    dedupe_key = f"ingestion-dead:{message.aggregate_id}:3"

    try:
        # Once explicitly configured, an unavailable service is a failed contract.
        await client.ping()
        await broker.dead_letter(
            dead_stream, message, "parse_failed", dedupe_key=dedupe_key
        )
        await broker.dead_letter(
            dead_stream, message, "parse_failed", dedupe_key=dedupe_key
        )

        entries = await client.xrange(dead_stream)
        assert len(entries) == 1
        assert entries[0][1]["aggregate_id"] == message.aggregate_id
        assert entries[0][1]["reason"] == "parse_failed"
    finally:
        await client.delete(dead_stream, f"{dead_stream}:dedupe")
        await client.aclose()
