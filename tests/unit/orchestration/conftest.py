from importlib import import_module

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.unit.orchestration.test_evidence import SNAPSHOT


@pytest.fixture
async def store(tmp_path):
    schema = import_module("agentic_rag.orchestration.schema")
    module = import_module("agentic_rag.orchestration.store")
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'coordination.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(schema.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    value = module.CoordinationStore(factory, enforce_source_access=False)
    await value.initialize(
        run_id="run-1", user_id="user-1", question="Compare revenues",
        snapshot=SNAPSHOT, timeout_seconds=300,
    )
    yield value
    await engine.dispose()
