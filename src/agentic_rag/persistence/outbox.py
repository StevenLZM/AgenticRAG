"""Transactional outbox dispatcher."""

from __future__ import annotations

from redis.exceptions import RedisError

from agentic_rag.persistence.redis_queue import StreamBroker
from agentic_rag.persistence.repositories import OutboxRecord, OutboxRepository


class OutboxDispatcher:
    """Publishes rows claimed by the caller-owned transaction, then records dispatch."""

    def __init__(self, outbox: OutboxRepository, broker: StreamBroker) -> None:
        self._outbox = outbox
        self._broker = broker

    async def dispatch_once(self, limit: int = 100) -> int:
        rows = await self._outbox.claim_pending(limit=limit)
        dispatched = 0
        for row in rows:
            try:
                await self.redispatch(row)
            except RedisError:
                await self._outbox.schedule_retry(row.id)
            else:
                dispatched += 1
        return dispatched

    async def redispatch(self, row: OutboxRecord) -> None:
        """Publish one claimed durable row and mark it dispatched."""
        await self._broker.publish(
            row.stream_name,
            row.aggregate_id,
            row.created_at,
            dedupe_key=f"{row.id}:{row.attempt_count}",
        )
        await self._outbox.mark_dispatched(row.id)
