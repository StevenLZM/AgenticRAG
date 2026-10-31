"""SQL reconciliation plus replayable Redis dispatch/listening and public Run projection."""
import asyncio
import logging
from datetime import timedelta
from typing import Any

from redis.exceptions import RedisError
from sqlalchemy import and_, exists, inspect, or_, select, update

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.coordinator import Coordinator
from agentic_rag.orchestration.store import CoordinationStore
from agentic_rag.orchestration.store_sql import INITIAL_STREAM, RESULT_STREAM, WAKE_STREAM, now, utc
from agentic_rag.persistence.redis_queue import StreamBroker
from agentic_rag.persistence.repositories import agent_runs, task_outbox
from agentic_rag.runtime.models import RuntimeConfigSnapshot

log = logging.getLogger(__name__)
LISTENER_GROUP = "supervisor-reception-v1"


async def check_schema(engine: Any) -> None:
    async with engine.connect() as connection:
        tables = await connection.run_sync(lambda conn: set(inspect(conn).get_table_names()))
    missing = set(schema.metadata.tables) - tables
    if missing:
        raise RuntimeError("supervisor schema migration 0009 is required")


class CoordinationService:
    def __init__(self, store: CoordinationStore, broker: StreamBroker, *, worker_id: str) -> None:
        self.store, self.broker, self.worker_id = store, broker, worker_id
        self.coordinator = Coordinator(store)

    async def dispatch(self) -> None:
        for row in await self.store.pending_outbox():
            await self.broker.publish(row["stream"], row["aggregate_id"], utc(row["created_at"]), row["dedupe_key"])
            await self.store.dispatched(row["id"])

    async def receive(self) -> None:
        for stream in (INITIAL_STREAM, RESULT_STREAM, WAKE_STREAM):
            messages = [*await self.broker.reclaim(stream, LISTENER_GROUP, self.worker_id, 30_000),
                        *await self.broker.consume(stream, LISTENER_GROUP, self.worker_id, 1)]
            for message in messages:
                if stream == RESULT_STREAM:
                    await self.store.receive_result(message.aggregate_id)
                elif stream == WAKE_STREAM:
                    try:
                        await self.coordinator.advance(message.aggregate_id)
                    except KeyError:
                        pass
                # INITIAL is reconciled from the scoped SQL Run, never from a
                # broker question, tenant, snapshot or task description.
                await self.broker.ack(stream, LISTENER_GROUP, message.id)

    async def reconcile_public_runs(self) -> None:
        async with self.store.session_factory() as db:
            # Filter before LIMIT: historical legacy/cancelled rows must not
            # permanently displace new supervisor Runs from reconciliation.
            active_coordination = exists(select(schema.runs.c.id).where(
                schema.runs.c.id == agent_runs.c.id, schema.runs.c.status.in_(("active", "finalizing")),
            ))
            rows = (await db.execute(select(agent_runs).where(
                agent_runs.c.runtime_config_snapshot["orchestration_version"].as_string() == "supervisor-v1",
                or_(agent_runs.c.status.in_(("queued", "running", "cancel_requested")),
                    and_(agent_runs.c.status == "cancelled", active_coordination)),
            ).order_by(agent_runs.c.created_at).limit(500))).mappings().all()
            notices = (await db.execute(select(task_outbox).where(task_outbox.c.stream_name == INITIAL_STREAM, task_outbox.c.status == "pending").limit(100))).mappings().all()
        for row in rows:
            raw = row["runtime_config_snapshot"]
            if raw.get("orchestration_version") != "supervisor-v1":
                continue
            snapshot = RuntimeConfigSnapshot.model_validate(raw)
            if row["status"] in ("cancel_requested", "cancelled"):
                try:
                    await self.store.cancel(row["id"], row["user_id"])
                except KeyError:
                    pass
            else:
                admitted = await self.store.initialize(run_id=row["id"], user_id=row["user_id"], question=row["question"], snapshot=snapshot,
                                                       absolute_deadline=utc(row["created_at"]) + timedelta(seconds=snapshot.query_run_timeout_seconds))
                if not admitted:
                    continue
            try:
                current = await self.store.read_run(row["id"])
            except Exception:
                if row["status"] == "cancelled":
                    continue
                raise
            state = current["status"]
            public_status = "running" if state in ("active", "finalizing") else state
            public_answer, sources = None, None
            if current["data"].get("answer") is not None:
                from agentic_rag.orchestration.streaming import public_answer as project, source_snapshot
                public_answer = project(current).model_dump(mode="json")
                sources = source_snapshot(current)
            async with self.store.session_factory.begin() as db:
                values = {"status": public_status, "active_slot": 1 if public_status == "running" else None,
                          "route": current["data"].get("route"), "answer": public_answer, "answer_sources": sources,
                          "error_code": current["data"].get("termination_reason") if public_status == "failed" else None}
                if public_status != "running":
                    values["finished_at"] = await now(db)
                # A public cancellation wins over a stale active projection.
                conditions = ("queued", "running") if public_status == "running" else ("queued", "running", "cancel_requested")
                await db.execute(update(agent_runs).where(agent_runs.c.id == row["id"], agent_runs.c.status.in_(conditions)).values(**values))
        for notice in notices:
            await self.broker.publish(INITIAL_STREAM, notice["aggregate_id"], utc(notice["created_at"]), f"initial:{notice['id']}")
            async with self.store.session_factory.begin() as db:
                await db.execute(update(task_outbox).where(task_outbox.c.id == notice["id"]).values(status="dispatched"))

    async def tick(self, *, public_runs: bool = True) -> None:
        if public_runs:
            try:
                await self.reconcile_public_runs()
            except (RedisError, OSError, ConnectionError, TimeoutError):
                log.warning("Initial notifications unavailable; continuing SQL reconciliation")
        await self.store.recover()
        # SQL is sufficient to recover a Redis outage or listener crash.
        async with self.store.session_factory() as db:
            unreceived = list((await db.execute(select(schema.results.c.id).where(~schema.results.c.id.in_(select(schema.inbox.c.result_id))).limit(100))).scalars())
            runs = list((await db.execute(select(schema.runs.c.id).where(schema.runs.c.status == "active").order_by(schema.runs.c.created_at).limit(100))).scalars())
        for result_id in unreceived:
            await self.store.receive_result(result_id)
        for run_id in runs:
            await self.coordinator.advance(run_id)
        try:
            await self.dispatch()
            await self.receive()
        except (RedisError, OSError, ConnectionError, TimeoutError):
            log.warning("Redis notifications unavailable; SQL reconciliation completed")

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.tick()
            except Exception as error:
                # SQL exception reprs can contain bound private JSON payloads.
                log.error("Coordination tick failed; durable state retained; error=%s", type(error).__name__)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1)
            except TimeoutError:
                pass
