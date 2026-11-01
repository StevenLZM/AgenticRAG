import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from agentic_rag.persistence.repositories import metadata
from agentic_rag.persistence.context_limiter import (
    SqlContextModelLimiter,
    ContextCapacityBusy,
)


@pytest.mark.asyncio
async def test_independent_workers_share_one_durable_model_slot(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/limit.db")
    async with engine.begin() as db:
        await db.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine)
    first = SqlContextModelLimiter(factory, limit=1, timeout_seconds=1)
    second = SqlContextModelLimiter(factory, limit=1, timeout_seconds=1)
    token = await first.acquire()
    with pytest.raises(ContextCapacityBusy):
        await second.acquire()
    await first.release(token)
    token2 = await second.acquire()
    assert token != token2
    await second.release(token2)
    await engine.dispose()


@pytest.mark.asyncio
async def test_failed_dispatch_holds_slot_until_durable_expiry(tmp_path):
    from sqlalchemy import update
    from agentic_rag.persistence.context_limiter import (
        context_model_capacity,
        LimitedContextModel,
    )

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/failed.db")
    async with engine.begin() as db:
        await db.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine)

    class Provider:
        async def extract(self, value):
            raise TimeoutError("unknown remote completion")

    model = LimitedContextModel(
        Provider(), SqlContextModelLimiter(factory, limit=1, timeout_seconds=1)
    )
    with pytest.raises(TimeoutError):
        await model.extract(None)
    recovered = SqlContextModelLimiter(factory, limit=1, timeout_seconds=1)
    with pytest.raises(ContextCapacityBusy):
        await recovered.acquire()
    async with factory.begin() as db:
        await db.execute(update(context_model_capacity).values(leases={"crashed": 0}))
    assert await recovered.acquire()
    await engine.dispose()
