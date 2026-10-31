"""Run only against an explicitly provided disposable MySQL schema."""
import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.store import CoordinationStore
from tests.unit.orchestration.test_evidence import SNAPSHOT
from tests.unit.orchestration.test_store import plan

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("same_work", [True, False])
async def test_mysql_repeatable_read_has_one_claim_winner_and_case_sensitive_keys(same_work):
    dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_RAG_TEST_MYSQL_DSN to an explicitly disposable schema")
    engine = create_async_engine(dsn, isolation_level="REPEATABLE READ")
    run_id = str(uuid4())
    try:
        async with engine.begin() as conn:
            await conn.run_sync(schema.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        first, second = CoordinationStore(factory), CoordinationStore(factory)
        await first.initialize(run_id=run_id, user_id="test", question="q", snapshot=SNAPSHOT.model_copy(update={"max_parallel_subagents_per_run": 1}))
        work = await first.submit_plan(run_id, "decision", 0, plan())
        winners = await asyncio.gather(first.claim(work[0], "one"), second.claim(work[0 if same_work else 1], "two"))
        assert sum(value is not None for value in winners) == 1
        async with factory() as db:
            collations = (await db.execute(text("SHOW FULL COLUMNS FROM agent_work_items"))).mappings().all()
        assert next(c for c in collations if c["Field"] == "logical_key")["Collation"].endswith("_bin")
    finally:
        async with engine.begin() as conn:
            for table in reversed(list(schema.metadata.tables.values())):
                if "run_id" in table.c:
                    await conn.execute(delete(table).where(table.c.run_id == run_id))
            await conn.execute(delete(schema.runs).where(schema.runs.c.id == run_id))
        await engine.dispose()
