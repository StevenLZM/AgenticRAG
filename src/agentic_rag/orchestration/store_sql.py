"""Small transaction helpers shared by dispatch, recovery and publication."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.runtime.ids import new_id

WORK_STREAMS = {
    "route": "agenticrag:supervisor:control",
    "supervisor": "agenticrag:supervisor:control",
    "fast_retrieval": "agenticrag:tasks:retrieval",
    "retrieval": "agenticrag:tasks:retrieval",
    "analysis": "agenticrag:tasks:analysis",
    "analysis_grade": "agenticrag:evidence-assessments",
    "prepare": "agenticrag:evidence-assessments",
    "grade": "agenticrag:evidence-assessments",
    "answer": "agenticrag:answers",
    "quality": "agenticrag:answers",
}
RESULT_STREAM = "agenticrag:task-results"
WAKE_STREAM = "agenticrag:supervisor-wakeups"
INITIAL_STREAM = "agenticrag:jobs:supervisor"


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


async def now(db: AsyncSession) -> datetime:
    value = (await db.execute(select(func.utc_timestamp(6) if db.bind is not None and db.bind.dialect.name == "mysql" else func.current_timestamp()))).scalar_one()
    return utc(value)


async def lock_run(db: AsyncSession, run_id: str) -> dict[str, Any]:
    # Acquiring a write lock also serializes SQLite-based offline tests. MySQL
    # uses the same per-Run row lock; no model or broker call occurs under it.
    await db.execute(update(schema.runs).where(schema.runs.c.id == run_id).values(version=schema.runs.c.version))
    row = (await db.execute(select(schema.runs).where(schema.runs.c.id == run_id).with_for_update())).mappings().one_or_none()
    if row is None:
        raise KeyError(run_id)
    return dict(row)


async def save_run(db: AsyncSession, run: dict[str, Any], *, changed: bool = True) -> None:
    if changed:
        run["version"] += 1
    await db.execute(update(schema.runs).where(schema.runs.c.id == run["id"]).values(
        data=run["data"], status=run["status"], version=run["version"],
    ))


async def notify(db: AsyncSession, run_id: str, stream: str, aggregate_id: str, key: str) -> str:
    existing = (await db.execute(select(schema.outbox.c.id).where(schema.outbox.c.dedupe_key == key))).scalar_one_or_none()
    if existing:
        return existing
    event_id, current = new_id(), await now(db)
    await db.execute(insert(schema.outbox).values(
        id=event_id, run_id=run_id, stream=stream, aggregate_id=aggregate_id,
        dedupe_key=key, status="pending", next_attempt_at=current, created_at=current,
    ))
    return event_id


async def enqueue(db: AsyncSession, run_id: str, stage: str, logical_key: str, payload: dict[str, Any]) -> str:
    existing = (await db.execute(select(schema.work_items.c.id).where(
        schema.work_items.c.run_id == run_id, schema.work_items.c.logical_key == logical_key,
    ))).scalar_one_or_none()
    if existing:
        return existing
    work_id, current = new_id(), await now(db)
    await db.execute(insert(schema.work_items).values(
        id=work_id, run_id=run_id, stage=stage, logical_key=logical_key, payload=payload,
        status="queued", lease_token=0, attempt_count=0, dispatch_generation=1,
        not_before=current, created_at=current,
    ))
    await notify(db, run_id, WORK_STREAMS[stage], work_id, f"work:{work_id}:1")
    return work_id
