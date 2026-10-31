"""Independent stage workers; a notification grants no execution authority."""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from redis.exceptions import RedisError

from agentic_rag.orchestration.store import CoordinationStore, StaleClaim, WorkClaim
from agentic_rag.orchestration.store_sql import WORK_STREAMS
from agentic_rag.persistence.redis_queue import StreamBroker

ROLE_STAGES = {
    "supervisor": ("route", "supervisor"), "retrieval": ("fast_retrieval", "retrieval"),
    "analysis": ("analysis",), "evidence": ("prepare", "grade", "analysis_grade"),
    "answer": ("answer", "quality"),
}
ROLE_STAGES["all"] = tuple(WORK_STREAMS)
GROUP = "supervisor-stage-workers-v1"
log = logging.getLogger(__name__)


class AgentWorker:
    def __init__(self, store: CoordinationStore, execute: Callable[[WorkClaim], Awaitable[dict[str, Any]]], *,
                 worker_id: str, role: str, broker: StreamBroker | None = None, lease_seconds: int = 30) -> None:
        self.store, self.execute, self.worker_id = store, execute, worker_id
        self.stages, self.broker, self.lease_seconds = ROLE_STAGES[role], broker, lease_seconds

    async def run_one(self, work_id: str | None = None) -> bool:
        if work_id is None:
            for stage in self.stages:
                items = await self.store.list_ready(stage, limit=8)
                for ready in items:
                    if await self.run_one(ready["id"]):
                        return True
            return False
        item = await self.store.get_work(work_id)
        if item is None or item["stage"] not in self.stages:
            return False
        claim = await self.store.claim(work_id, self.worker_id, self.lease_seconds)
        if claim is None:
            return False
        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                await self.store.heartbeat(claim, self.lease_seconds)
        async def execute_work() -> dict[str, Any]:
            return await self.execute(claim)
        work = asyncio.create_task(execute_work())
        renewal = asyncio.create_task(heartbeat())
        try:
            remaining = max(0.01, (claim.deadline - datetime.now(UTC)).total_seconds())
            async with asyncio.timeout(remaining):
                done, _ = await asyncio.wait((work, renewal), return_when=asyncio.FIRST_COMPLETED)
                if renewal in done:
                    await renewal
                    raise StaleClaim("lease renewal stopped")
                result = await work
                # Answer and quality executors commit their own atomic streaming
                # terminal transition, which can close the Run before this point.
                if claim.stage not in ("answer", "quality"):
                    await self.store.submit_result(claim, result)
        except StaleClaim:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as error:
            try:
                if claim.stage in ("answer", "quality"):
                    from agentic_rag.orchestration.streaming import fail_output
                    await fail_output(self.store, claim, type(error).__name__)
                else:
                    await self.store.fail(claim, type(error).__name__, retryable=isinstance(error, (TimeoutError, ConnectionError, OSError)))
            except StaleClaim:
                pass
            log.warning("Stage failed: stage=%s work=%s error=%s", claim.stage, claim.work_id, type(error).__name__)
        finally:
            work.cancel()
            renewal.cancel()
            await asyncio.gather(work, renewal, return_exceptions=True)
        return True

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        streams = tuple(dict.fromkeys(WORK_STREAMS[s] for s in self.stages))
        while not stop_event.is_set():
            did_work = False
            if self.broker is not None:
                try:
                    for stream in streams:
                        messages = [*await self.broker.reclaim(stream, GROUP, self.worker_id, self.lease_seconds * 2000),
                                    *await self.broker.consume(stream, GROUP, self.worker_id, 1)]
                        for message in messages:
                            did_work = await self.run_one(message.aggregate_id) or did_work
                            # SQL owns recovery for BUSY, duplicate and lost leases.
                            await self.broker.ack(stream, GROUP, message.id)
                except (RedisError, OSError, ConnectionError, TimeoutError):
                    log.warning("Stage notification transport unavailable; SQL reconciliation remains active")
            did_work = await self.run_one() or did_work
            if not did_work:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                except TimeoutError:
                    pass
