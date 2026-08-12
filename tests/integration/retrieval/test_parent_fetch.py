"""Opt-in MySQL integration coverage for fail-closed parent hydration.

Set ``AGENTIC_RAG_TEST_MYSQL_DSN`` to a disposable MySQL database.  The
fixture never guesses a developer endpoint.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import insert, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.domain.models import DocumentStatus, DocumentVersionStatus, UserScope
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence.repositories import (
    SqlAlchemyParentRepository,
    document_versions,
    documents,
    parent_chunks,
)
from agentic_rag.retrieval.parents import ParentFetcher, ParentScopeViolation


pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class ParentFetchFixture:
    fetcher: ParentFetcher
    owner_parent_id: str
    other_user_parent_id: str
    inactive_parent_id: str


@pytest.fixture(scope="module")
def mysql_dsn() -> str:
    dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_RAG_TEST_MYSQL_DSN to a disposable MySQL database")
    parsed = urlparse(dsn)
    if parsed.scheme != "mysql+asyncmy":
        pytest.skip("AGENTIC_RAG_TEST_MYSQL_DSN must use mysql+asyncmy")

    engine = create_mysql_engine(dsn, pool_pre_ping=True)

    async def probe() -> None:
        try:
            async with engine.connect():
                pass
        finally:
            await engine.dispose()

    try:
        asyncio.run(probe())
    except (OSError, RuntimeError, SQLAlchemyError) as error:
        pytest.skip(f"disposable MySQL database is unavailable: {type(error).__name__}")
    return dsn


@pytest.fixture(scope="module", autouse=True)
def migrated_schema(mysql_dsn: str) -> Iterator[None]:
    migration = Config("alembic.ini")
    migration.set_main_option("sqlalchemy.url", mysql_dsn)
    command.upgrade(migration, "head")
    yield


@pytest.fixture
async def parent_fetcher(
    mysql_dsn: str, migrated_schema: None
) -> AsyncIterator[ParentFetchFixture]:
    engine = create_mysql_engine(mysql_dsn, pool_pre_ping=True)
    factory = create_session_factory(engine)
    suffix = uuid4().hex
    owner_parent_id = f"owner-parent-{suffix}"
    other_user_parent_id = f"other-user-parent-{suffix}"
    inactive_parent_id = f"inactive-parent-{suffix}"
    try:
        async with factory.begin() as session:
            owner_document_id = str(uuid4())
            owner_version_id = str(uuid4())
            await _insert_document(
                session,
                document_id=owner_document_id,
                version_id=owner_version_id,
                user_id="u1",
            )
            await _insert_parent(
                session,
                parent_id=owner_parent_id,
                document_id=owner_document_id,
                version_id=owner_version_id,
                user_id="u1",
                status="active",
            )
            await _insert_parent(
                session,
                parent_id=inactive_parent_id,
                document_id=owner_document_id,
                version_id=owner_version_id,
                user_id="u1",
                status="inactive",
                ordinal=1,
            )

            other_document_id = str(uuid4())
            other_version_id = str(uuid4())
            await _insert_document(
                session,
                document_id=other_document_id,
                version_id=other_version_id,
                user_id="u2",
            )
            await _insert_parent(
                session,
                parent_id=other_user_parent_id,
                document_id=other_document_id,
                version_id=other_version_id,
                user_id="u2",
                status="active",
            )

        async with factory() as session:
            yield ParentFetchFixture(
                fetcher=ParentFetcher(SqlAlchemyParentRepository(session)),
                owner_parent_id=owner_parent_id,
                other_user_parent_id=other_user_parent_id,
                inactive_parent_id=inactive_parent_id,
            )
    finally:
        await engine.dispose()


async def test_parent_fetch_returns_active_scoped_parent_in_requested_order(
    parent_fetcher: ParentFetchFixture,
) -> None:
    scope = UserScope(user_id="u1")

    evidence = await parent_fetcher.fetch(
        [parent_fetcher.owner_parent_id], scope
    )

    assert [item.parent_id for item in evidence] == [parent_fetcher.owner_parent_id]
    assert evidence[0].content == "content for u1"


async def test_parent_fetch_fails_closed_on_scope_mismatch(
    parent_fetcher: ParentFetchFixture,
) -> None:
    with pytest.raises(ParentScopeViolation):
        await parent_fetcher.fetch(
            [parent_fetcher.other_user_parent_id], UserScope(user_id="u1")
        )


async def test_parent_fetch_fails_closed_for_inactive_parent(
    parent_fetcher: ParentFetchFixture,
) -> None:
    with pytest.raises(ParentScopeViolation):
        await parent_fetcher.fetch(
            [parent_fetcher.inactive_parent_id], UserScope(user_id="u1")
        )


async def _insert_document(
    session: AsyncSession, *, document_id: str, version_id: str, user_id: str
) -> None:
    await session.execute(
        insert(documents).values(
            id=document_id,
            user_id=user_id,
            source_type="text",
            filename=f"{user_id}.txt",
            mime_type="text/plain",
            content_hash="a" * 64,
            status=DocumentStatus.ACTIVE.value,
            source_trust="untrusted",
        )
    )
    await session.execute(
        insert(document_versions).values(
            id=version_id,
            document_id=document_id,
            version_no=1,
            parser_version="docling-v1",
            pipeline_version="ingestion-v1",
            parent_count=2,
            child_count=2,
            embedding_version="text-embedding-v3",
            index_generation="parent-fetch-test",
            status=DocumentVersionStatus.ACTIVE.value,
        )
    )
    await session.execute(
        update(documents)
        .where(documents.c.id == document_id)
        .values(active_version_id=version_id)
    )


async def _insert_parent(
    session: AsyncSession,
    *,
    parent_id: str,
    document_id: str,
    version_id: str,
    user_id: str,
    status: str,
    ordinal: int = 0,
) -> None:
    await session.execute(
        insert(parent_chunks).values(
            id=parent_id,
            user_id=user_id,
            document_id=document_id,
            document_version_id=version_id,
            ordinal=ordinal,
            heading_path=["Parent fetch"],
            content_type="paragraph",
            content=f"content for {user_id}",
            page_from=1,
            page_to=1,
            ast_locator="{\"locator\":\"parent-fetch\"}",
            content_hash="b" * 64,
            status=status,
        )
    )
