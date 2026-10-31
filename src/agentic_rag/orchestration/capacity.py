"""Cross-process capacity leases, acquired in a stable order without SQL waits."""
from datetime import timedelta
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.store_sql import now, utc


def requirements(stage: str, user_id: str, limits: dict[str, int]) -> dict[str, int]:
    if stage == "prepare" or not limits:
        return {}
    requirements = {"global:model": limits["model"], f"tenant:{user_id}:model": limits["tenant_model"]}
    if stage in ("retrieval", "fast_retrieval"):
        requirements["global:rerank"] = limits["rerank"]
    import hashlib
    return {key if len(key) <= 255 else "tenant:" + hashlib.sha256(user_id.encode()).hexdigest(): value for key, value in requirements.items()}


async def acquire(db: AsyncSession, item: dict[str, Any], user_id: str, limits: dict[str, int], lease_seconds: int) -> bool:
    current, selected = await now(db), []
    for resource, limit in sorted(requirements(item["stage"], user_id, limits).items()):
        for slot in range(limit):
            exists = (await db.execute(select(schema.resources.c.slot).where(schema.resources.c.resource == resource, schema.resources.c.slot == slot))).first()
            if exists is None:
                try:
                    async with db.begin_nested():
                        await db.execute(insert(schema.resources).values(resource=resource, slot=slot, token=0))
                except IntegrityError:
                    pass
        rows = (await db.execute(select(schema.resources).where(schema.resources.c.resource == resource).order_by(schema.resources.c.slot).with_for_update())).mappings().all()
        available = next((r for r in rows if r["slot"] < limit and (r["lease_until"] is None or utc(r["lease_until"]) <= current)), None)
        if available is None:
            return False
        selected.append(available)
    for acquired in selected:
        await db.execute(update(schema.resources).where(schema.resources.c.resource == acquired["resource"], schema.resources.c.slot == acquired["slot"]).values(owner=item["id"], token=item["lease_token"] + 1, lease_until=current + timedelta(seconds=lease_seconds)))
    return True


async def renew(db: AsyncSession, work_id: str, token: int, lease_seconds: int, *, expected_resources: set[str]) -> None:
    from agentic_rag.orchestration.store import StaleClaim
    current = await now(db)
    held = (await db.execute(select(schema.resources).where(schema.resources.c.owner == work_id,
        schema.resources.c.token == token, schema.resources.c.lease_until > current).with_for_update())).mappings().all()
    if {row["resource"] for row in held} != expected_resources or len(held) != len(expected_resources):
        raise StaleClaim("capacity lease ownership lost")
    await db.execute(update(schema.resources).where(schema.resources.c.owner == work_id, schema.resources.c.token == token).values(lease_until=await now(db) + timedelta(seconds=lease_seconds)))


async def release(db: AsyncSession, work_id: str, token: int) -> None:
    await db.execute(update(schema.resources).where(schema.resources.c.owner == work_id, schema.resources.c.token == token).values(owner=None, lease_until=None))


async def lock_admission(db: AsyncSession) -> None:
    """Serialize new-Run admission across coordinator processes."""
    try:
        async with db.begin_nested():
            await db.execute(insert(schema.resources).values(resource="supervisor:admission", slot=0, token=0))
    except IntegrityError:
        pass
    await db.execute(select(schema.resources).where(schema.resources.c.resource == "supervisor:admission", schema.resources.c.slot == 0).with_for_update())
