"""Global context model capacity; crash leases remain charged until expiry."""

import asyncio
from datetime import UTC
from uuid import uuid4
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    String,
    Table,
    insert,
    select,
    update,
    func,
)
from sqlalchemy.exc import IntegrityError
from agentic_rag.persistence.repositories import metadata

context_model_capacity = Table(
    "context_model_capacity",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("capacity", Integer, nullable=False),
    Column("version", Integer, nullable=False),
    Column("leases", JSON, nullable=False),
)


class ContextCapacityBusy(ConnectionError):
    pass


class SqlContextModelLimiter:
    def __init__(self, factory, *, limit, timeout_seconds):
        if not 1 <= limit <= 32 or not 0 < timeout_seconds <= 300:
            raise ValueError("finite global model capacity and timeout required")
        self.factory, self.limit, self.timeout = factory, limit, timeout_seconds

    async def acquire(self):
        async with self.factory.begin() as db:
            try:
                async with db.begin_nested():
                    await db.execute(
                        insert(context_model_capacity).values(
                            id="context-model",
                            capacity=self.limit,
                            version=0,
                            leases={},
                        )
                    )
            except IntegrityError:
                pass
            row = (
                (
                    await db.execute(
                        select(context_model_capacity)
                        .where(context_model_capacity.c.id == "context-model")
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if row["capacity"] != self.limit:
                raise ValueError("global context concurrency differs between workers")
            current = (
                (await db.scalar(select(func.current_timestamp())))
                .replace(tzinfo=UTC)
                .timestamp()
            )
            leases = {
                key: expiry for key, expiry in row["leases"].items() if expiry > current
            }
            if len(leases) >= self.limit:
                raise ContextCapacityBusy("global context model slots exhausted")
            token = uuid4().hex
            leases[token] = current + self.timeout + 60
            result = await db.execute(
                update(context_model_capacity)
                .where(
                    context_model_capacity.c.id == "context-model",
                    context_model_capacity.c.version == row["version"],
                )
                .values(leases=leases, version=row["version"] + 1)
            )
            if result.rowcount != 1:
                raise ContextCapacityBusy("global context capacity raced")
            return token

    async def release(self, token):
        async with self.factory.begin() as db:
            row = (
                (
                    await db.execute(
                        select(context_model_capacity)
                        .where(context_model_capacity.c.id == "context-model")
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            leases = dict(row["leases"])
            leases.pop(token, None)
            await db.execute(
                update(context_model_capacity)
                .where(
                    context_model_capacity.c.id == "context-model",
                    context_model_capacity.c.version == row["version"],
                )
                .values(leases=leases, version=row["version"] + 1)
            )


class LimitedContextModel:
    def __init__(self, model, limiter):
        self.model, self.limiter = model, limiter

    async def _call(self, method, value):
        token = await self.limiter.acquire()
        # On cancellation/error the provider may still be executing. Retain
        # capacity until bounded timeout + grace, just like unknown spend.
        async with asyncio.timeout(self.limiter.timeout):
            result = await getattr(self.model, method)(value)
        await self.limiter.release(token)
        return result

    async def extract(self, value):
        return await self._call("extract", value)

    async def verify(self, value):
        return await self._call("verify", value)
