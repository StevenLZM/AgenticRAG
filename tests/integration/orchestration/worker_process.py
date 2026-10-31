"""Disposable-process harness; no provider calls, shared state only through SQL/Redis."""
import asyncio
import os
import sys

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.orchestration.store import CoordinationStore
from agentic_rag.orchestration.worker import AgentWorker
from agentic_rag.orchestration.service import CoordinationService
from agentic_rag.persistence.redis_queue import RedisStreamsBroker
from tests.unit.orchestration.test_evidence import candidate, ranked


async def main():
    dsn, mode, value = sys.argv[1:]
    engine = create_async_engine(dsn)
    store = CoordinationStore(async_sessionmaker(engine, expire_on_commit=False))
    try:
        if mode == "coordinate":
            redis = Redis.from_url(value, decode_responses=True)
            try:
                await CoordinationService(store, RedisStreamsBroker(redis), worker_id=str(os.getpid())).tick(public_runs=False)
            finally:
                await redis.aclose()
        else:
            async def execute(claim):
                await store.consume_budget(claim, "queries")
                if mode == "crash":
                    os._exit(17)
                return candidate(claim.payload["task_key"], "revenue", ranked()).model_dump(mode="json")
            await AgentWorker(store, execute, worker_id=str(os.getpid()), role="retrieval").run_one(value)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
