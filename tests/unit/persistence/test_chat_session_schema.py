"""Schema readiness must detect an unmigrated database before serving queries."""
import importlib.util

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from agentic_rag.persistence.repositories import metadata


async def test_schema_check_rejects_legacy_database_and_accepts_upgraded_schema():
    assert importlib.util.find_spec("agentic_rag.persistence.schema_readiness") is not None
    from agentic_rag.persistence.schema_readiness import ChatSchemaUnavailable, check_chat_schema
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        with pytest.raises(ChatSchemaUnavailable, match="0008_chat_sessions"):
            await check_chat_schema(engine)
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        await check_chat_schema(engine)
        async with engine.begin() as conn:
            await conn.execute(text("DROP INDEX ix_chat_sessions_user_activity"))
        with pytest.raises(ChatSchemaUnavailable):
            await check_chat_schema(engine)
    finally:
        await engine.dispose()
