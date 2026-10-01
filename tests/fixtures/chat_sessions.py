"""Portable behavior checks plus opt-in, real MySQL transaction checks."""
from __future__ import annotations

import asyncio
import importlib.util
import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.mysql import create_mysql_engine
from agentic_rag.persistence.repositories import metadata
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.runtime.run_manager import RunManager, TransactionalRunRepository


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="0.1.0", graph_version="query-v2", prompt_version="prompt-v2",
    main_model_id="main", light_model_id="light", embedding_model="embedding",
    embedding_dimensions=1024, reranker_version="reranker-v1",
    retrieval_config_version="retrieval-v1", index_generation="index-v1",
    memory_config_version="memory-v1",
)


@pytest.fixture(params=["sqlite", "mysql"])
async def chat_database(request, tmp_path):
    if request.param == "mysql":
        dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN")
        if not dsn:
            pytest.skip("set AGENTIC_RAG_TEST_MYSQL_DSN to an isolated test database")
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", dsn.replace("%", "%%"))
        await asyncio.to_thread(command.upgrade, config, "head")
        engine = create_mysql_engine(dsn)
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'chat.sqlite'}")
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False), engine
    finally:
        await engine.dispose()


@pytest.fixture
def chat_users():
    suffix = uuid4().hex
    return UserScope(user_id=f"chat-a-{suffix}"), UserScope(user_id=f"chat-b-{suffix}")


def chat_service(factory):
    assert importlib.util.find_spec("agentic_rag.runtime.chat_sessions") is not None, "chat service missing"
    from agentic_rag.runtime.chat_sessions import ChatSessionService
    manager = RunManager(session_factory=factory, runs=TransactionalRunRepository(factory))
    return ChatSessionService(factory, manager)
