"""Isolation contract for the opt-in real Query acceptance broker."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agentic_rag.persistence.redis_queue import StreamMessage
from agentic_rag.runtime.query_worker import QUERY_DEAD_STREAM, QUERY_GROUP, QUERY_STREAM
from agentic_rag.testing.isolated_query_broker import IsolatedQueryBroker
from agentic_rag.testing.real_provider_config import (
    explicit_provider_configuration_issue,
    provider_configuration_issue,
)


class _RecordingBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def publish(
        self,
        stream: str,
        aggregate_id: str,
        enqueued_at: datetime,
        dedupe_key: str | None = None,
    ) -> str:
        self.calls.append(("publish", stream, aggregate_id, dedupe_key))
        return "1-0"

    async def consume(
        self, stream: str, group: str, consumer: str, block_ms: int
    ) -> list[StreamMessage]:
        self.calls.append(("consume", stream, group, consumer, block_ms))
        return []

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        self.calls.append(("ack", stream, group, message_id))

    async def reclaim(
        self, stream: str, group: str, consumer: str, min_idle_ms: int
    ) -> list[StreamMessage]:
        self.calls.append(("reclaim", stream, group, consumer, min_idle_ms))
        return []

    async def dead_letter(
        self,
        dead_stream: str,
        message: StreamMessage,
        reason: str,
        dedupe_key: str | None = None,
    ) -> None:
        self.calls.append(("dead_letter", dead_stream, message.id, reason, dedupe_key))


@pytest.mark.asyncio
async def test_isolated_query_broker_maps_all_query_stream_operations() -> None:
    inner = _RecordingBroker()
    broker = IsolatedQueryBroker(inner, namespace="real-query-a1b2")
    timestamp = datetime(2026, 8, 15, tzinfo=UTC)
    message = StreamMessage(id="1-0", aggregate_id="run-1", enqueued_at=timestamp)

    assert await broker.publish(QUERY_STREAM, "run-1", timestamp, "outbox-1:0") == "1-0"
    assert await broker.consume(QUERY_STREAM, QUERY_GROUP, "worker-1", 10) == []
    assert await broker.reclaim(QUERY_STREAM, QUERY_GROUP, "worker-1", 20) == []
    await broker.ack(QUERY_STREAM, QUERY_GROUP, "1-0")
    await broker.dead_letter(QUERY_DEAD_STREAM, message, "failed", "delivery-1")

    stream = "agenticrag:e2e:real-query-a1b2:query"
    group = "agenticrag-e2e-real-query-a1b2"
    assert broker.query_stream == stream
    assert await broker.publish(stream, "run-1", timestamp, "outbox-2:0") == "1-0"
    assert inner.calls == [
        ("publish", stream, "run-1", "outbox-1:0"),
        ("consume", stream, group, "worker-1", 10),
        ("reclaim", stream, group, "worker-1", 20),
        ("ack", stream, group, "1-0"),
        ("dead_letter", f"{stream}:dead", "1-0", "failed", "delivery-1"),
        ("publish", stream, "run-1", "outbox-2:0"),
    ]
    assert broker.cleanup_keys == (
        stream,
        f"{stream}:dedupe",
        f"{stream}:dead",
        f"{stream}:dead:dedupe",
    )


@pytest.mark.asyncio
async def test_isolated_query_broker_rejects_an_unmapped_production_stream() -> None:
    broker = IsolatedQueryBroker(_RecordingBroker(), namespace="real-query-a1b2")

    with pytest.raises(ValueError, match="isolated Query broker"):
        await broker.publish("agenticrag:jobs:other", "run-1", datetime.now(UTC))


def test_real_provider_configuration_distinguishes_missing_from_invalid_values() -> None:
    valid = SimpleNamespace(
        deepseek_base_url="https://deepseek.example/v1",
        deepseek_api_key="deepseek-key",
        qwen_embedding_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-key",
        mem0_enabled=True,
    )

    assert provider_configuration_issue(valid) is None
    assert provider_configuration_issue(
        SimpleNamespace(**{**vars(valid), "qwen_api_key": None})
    ) == ("missing", ("qwen_api_key",))
    assert provider_configuration_issue(
        SimpleNamespace(**{**vars(valid), "mem0_enabled": False})
    ) == ("invalid", ("mem0_enabled",))
    assert provider_configuration_issue(
        SimpleNamespace(**{**vars(valid), "deepseek_api_key": "replace-with-key"})
    ) == ("invalid", ("deepseek_api_key",))


def test_explicit_invalid_provider_environment_overrides_a_missing_settings_skip() -> None:
    """A disabled/placeholder provider is a failure even if another field is absent."""
    assert explicit_provider_configuration_issue(
        {
            "AGENTIC_RAG_MEM0_ENABLED": "false",
            "AGENTIC_RAG_DEEPSEEK_API_KEY": "replace-with-key",
        }
    ) == ("invalid", ("deepseek_api_key", "mem0_enabled"))
