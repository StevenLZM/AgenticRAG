"""Private Redis Stream names for opt-in live Query acceptance runs.

The production QueryWorker intentionally uses stable stream and consumer-group
names.  A live acceptance case must not attach to those shared names, because
doing so could consume another process's delivery.  This adapter preserves the
production worker and outbox interfaces while translating only the fixed Query
stream protocol to a run-private namespace on the real Redis client.
"""

from __future__ import annotations

import re
from datetime import datetime

from agentic_rag.persistence.redis_queue import StreamBroker, StreamMessage
from agentic_rag.runtime.query_worker import QUERY_DEAD_STREAM, QUERY_GROUP, QUERY_STREAM


_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class IsolatedQueryBroker:
    """Map QueryWorker's fixed stream/group contract to one isolated prefix."""

    def __init__(self, broker: StreamBroker, *, namespace: str) -> None:
        if not _NAMESPACE.fullmatch(namespace):
            raise ValueError("isolated Query broker namespace must be lowercase URL-safe text")
        self._broker = broker
        prefix = f"agenticrag:e2e:{namespace}"
        self._query_stream = f"{prefix}:query"
        self._dead_stream = f"{self._query_stream}:dead"
        self._group = f"agenticrag-e2e-{namespace}"

    @property
    def query_stream(self) -> str:
        """The durable private stream written by the scoped Run repository."""
        return self._query_stream

    @property
    def cleanup_keys(self) -> tuple[str, str, str, str]:
        """Every Redis key owned by this adapter, including dead-letter dedupe."""
        return (
            self._query_stream,
            f"{self._query_stream}:dedupe",
            self._dead_stream,
            f"{self._dead_stream}:dedupe",
        )

    async def publish(
        self,
        stream: str,
        aggregate_id: str,
        enqueued_at: datetime,
        dedupe_key: str | None = None,
    ) -> str:
        return await self._broker.publish(
            self._stream(stream), aggregate_id, enqueued_at, dedupe_key
        )

    async def consume(
        self, stream: str, group: str, consumer: str, block_ms: int
    ) -> list[StreamMessage]:
        return await self._broker.consume(
            self._stream(stream), self._group_for(group), consumer, block_ms
        )

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        await self._broker.ack(self._stream(stream), self._group_for(group), message_id)

    async def reclaim(
        self, stream: str, group: str, consumer: str, min_idle_ms: int
    ) -> list[StreamMessage]:
        return await self._broker.reclaim(
            self._stream(stream), self._group_for(group), consumer, min_idle_ms
        )

    async def dead_letter(
        self,
        dead_stream: str,
        message: StreamMessage,
        reason: str,
        dedupe_key: str | None = None,
    ) -> None:
        await self._broker.dead_letter(
            self._stream(dead_stream), message, reason, dedupe_key
        )

    def _stream(self, stream: str) -> str:
        if stream in {QUERY_STREAM, self._query_stream}:
            return self._query_stream
        if stream in {QUERY_DEAD_STREAM, self._dead_stream}:
            return self._dead_stream
        raise ValueError("isolated Query broker rejects an unmapped production stream")

    def _group_for(self, group: str) -> str:
        if group != QUERY_GROUP:
            raise ValueError("isolated Query broker rejects an unmapped consumer group")
        return self._group
