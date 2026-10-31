"""Database reconciliation makes lost broker notifications recoverable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.contracts import content_hash
from agentic_rag.orchestration.dispatch import collect_locked
from agentic_rag.orchestration.store_sql import RESULT_STREAM, WAKE_STREAM, WORK_STREAMS, lock_run, notify, now, save_run, utc
from agentic_rag.runtime.ids import new_id

if TYPE_CHECKING:
    from agentic_rag.orchestration.store import CoordinationStore


async def terminal_failure(db: AsyncSession, run: dict[str, Any], item: dict[str, Any], code: str) -> None:
    await db.execute(update(schema.work_items).where(schema.work_items.c.id == item["id"]).values(
        status="failed", error_code=code, lease_until=None,
    ))
    key = item["payload"].get("task_key")
    if key in run["data"]["tasks"]:
        run["data"]["tasks"][key]["state"] = "failed"
    existing = (await db.execute(select(schema.results.c.id).where(schema.results.c.work_id == item["id"]))).scalar_one_or_none()
    if not existing:
        result_id = new_id()
        payload = {"error": code, "work_payload": item["payload"]}
        await db.execute(insert(schema.results).values(
            id=result_id, run_id=run["id"], work_id=item["id"], stage=item["stage"],
            payload=payload, content_hash=content_hash(payload), created_at=await now(db),
        ))
        await notify(db, run["id"], RESULT_STREAM, result_id, f"result:{result_id}")
        await notify(db, run["id"], WAKE_STREAM, run["id"], f"failure:{result_id}")


async def recover(store: CoordinationStore) -> int:
    async with store.session_factory() as db:
        ids = list((await db.execute(select(schema.runs.c.id).where(
            schema.runs.c.status.in_(("active", "finalizing")),
        ).order_by(schema.runs.c.created_at).limit(100))).scalars())
    recovered = 0
    for run_id in ids:
        async with store.session_factory.begin() as db:
            run = await lock_run(db, run_id)
            current = await now(db)
            if utc(run["deadline"]) <= current:
                run["status"] = "failed"
                run["data"]["termination_reason"] = "deadline"
                from agentic_rag.orchestration.streaming import terminate_output
                await terminate_output(db, run, "deadline", quality_only=bool(run["data"].get("answer") and run["data"]["answer"]["status"] == "finalizing"))
                await db.execute(update(schema.work_items).where(
                    schema.work_items.c.run_id == run_id,
                    schema.work_items.c.status.in_(("queued", "running", "retry_wait")),
                ).values(status="cancelled", lease_until=None))
                await save_run(db, run)
                recovered += 1
                continue
            items = (await db.execute(select(schema.work_items).where(
                schema.work_items.c.run_id == run_id,
                schema.work_items.c.status.in_(("queued", "running", "retry_wait")),
            ))).mappings().all()
            for raw in items:
                item = dict(raw)
                expired = item["status"] == "running" and utc(item["lease_until"]) <= current
                if expired:
                    if item["stage"] in ("answer", "quality"):
                        from agentic_rag.orchestration.streaming import terminate_output
                        await terminate_output(db, run, "worker_lost", quality_only=item["stage"] == "quality")
                        await db.execute(update(schema.work_items).where(schema.work_items.c.id == item["id"]).values(status="failed", lease_until=None))
                        recovered += 1
                        continue
                    if item["attempt_count"] >= 3:
                        await terminal_failure(db, run, item, "lease_retry_exhausted")
                        recovered += 1
                        continue
                    item["dispatch_generation"] += 1
                    await db.execute(update(schema.work_items).where(schema.work_items.c.id == item["id"]).values(
                        status="queued", owner=None, lease_until=None, not_before=current,
                        dispatch_generation=item["dispatch_generation"],
                    ))
                    item["status"] = "queued"
                    recovered += 1
                if item["status"] in ("queued", "retry_wait") and utc(item["not_before"]) <= current:
                    # SQL remains the recovery responsibility after a BUSY ACK.
                    # Re-publish with a new notification generation when the
                    # previous one has already been delivered.
                    key = f"work:{item['id']}:{item['dispatch_generation']}"
                    delivered = (await db.execute(select(schema.outbox.c.status).where(schema.outbox.c.dedupe_key == key))).scalar_one_or_none()
                    if delivered == "dispatched":
                        item["dispatch_generation"] += 1
                        await db.execute(update(schema.work_items).where(schema.work_items.c.id == item["id"]).values(
                            dispatch_generation=item["dispatch_generation"],
                        ))
                        key = f"work:{item['id']}:{item['dispatch_generation']}"
                    await notify(db, run_id, WORK_STREAMS[item["stage"]], item["id"], key)
            for group_key in tuple(run["data"]["groups"]):
                await collect_locked(db, run, group_key)
            undelivered = (await db.execute(select(schema.results.c.id).where(
                schema.results.c.run_id == run_id, ~schema.results.c.id.in_(select(schema.inbox.c.result_id)),
            ))).scalars().all()
            for result_id in undelivered:
                # The prior notice can have been ACKed/lost before durable Inbox
                # reception. Reconciliation creates a fresh bounded-time notice.
                await notify(db, run_id, RESULT_STREAM, result_id,
                             f"result-repair:{result_id}:{int(current.timestamp()) // 15}")
            if (run["data"]["inbox_next_seq"] > run["data"]["applied_inbox_seq"]
                    or run["data"]["pending_results"] or run["data"]["pending_semantic_results"]):
                await notify(db, run_id, WAKE_STREAM, run_id,
                             f"recovery-wake:{run_id}:{run['version']}:{int(current.timestamp())}")
            await save_run(db, run, changed=False)
    return recovered
