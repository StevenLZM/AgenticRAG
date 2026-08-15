"""Transactional outbox dispatcher."""

from __future__ import annotations

from redis.exceptions import RedisError

from agentic_rag.observability.logging import emit_degradation
from agentic_rag.persistence.redis_queue import StreamBroker
from agentic_rag.persistence.repositories import OutboxRecord, OutboxRepository


class OutboxDispatcher:
    """Publishes rows claimed by the caller-owned transaction, then records dispatch."""

    def __init__(
        self,
        outbox: OutboxRepository,
        broker: StreamBroker,
        *,
        aggregate_type: str | None = None,
    ) -> None:
        self._outbox = outbox
        self._broker = broker
        self._aggregate_type = aggregate_type

    async def dispatch_once(self, limit: int = 100) -> int:
        if self._aggregate_type is None:
            # Keep deployment-owned lightweight outbox ports source-compatible.
            rows = await self._outbox.claim_pending(limit=limit)
        else:
            rows = await self._outbox.claim_pending(
                limit=limit, aggregate_type=self._aggregate_type
            )
        dispatched = 0
        for row in rows:
            try:
                await self.redispatch(row)
            except RedisError:
                await self._outbox.schedule_retry(row.id)
                await emit_degradation(
                    component="outbox",
                    reason="outbox_retry",
                    run_id=row.aggregate_id,
                    snapshot_id="",
                    attempt=row.attempt_count + 1,
                    retryable=True,
                    outcome="degraded",
                    event_type="OUTBOX_RETRY",
                )
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
