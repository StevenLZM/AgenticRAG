"""Real SQL behavior for immutable Manifest attachment and locator schema."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import event, insert, update
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.schema import CreateTable

from agentic_rag.ingestion.indexer import StagingContext
from agentic_rag.persistence.repositories import (
    document_versions,
    documents,
    metadata,
    parent_chunks,
)
from agentic_rag.persistence.staging import (
    ParentStagingConflict,
    SqlAlchemyParentStagingStore,
    StagingVersionError,
)


def _context() -> StagingContext:
    return StagingContext(
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        version_no=1,
        pipeline_version="ingestion-v1",
        embedding_version="text-embedding-v3",
        index_generation="index-v1",
    )


@pytest.fixture
async def manifest_store() -> AsyncIterator[
    tuple[
        SqlAlchemyParentStagingStore,
        async_sessionmaker[AsyncSession],
        AsyncEngine,
    ]
]:
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    async with factory.begin() as session:
        await session.execute(
            insert(documents).values(
                id="document-1",
                user_id="user-1",
                source_type="text",
                filename="test.txt",
                mime_type="text/plain",
                content_hash="a" * 64,
                status="processing",
                source_trust="untrusted",
                created_at=now,
                updated_at=now,
            )
        )
        await session.execute(
            insert(document_versions).values(
                id="version-1",
                document_id="document-1",
                version_no=1,
                parser_version="docling-v1",
                pipeline_version="ingestion-v1",
                parent_count=0,
                child_count=0,
                embedding_version="text-embedding-v3",
                index_generation="index-v1",
                status="building",
                created_at=now,
            )
        )
        await session.execute(
            insert(parent_chunks).values(
                id="parent-1",
                user_id="user-1",
                document_id="document-1",
                document_version_id="version-1",
                ordinal=0,
                heading_path=[],
                content_type="paragraph",
                content="hello",
                page_from=1,
                page_to=1,
                ast_locator='{"spans":[]}',
                content_hash="b" * 64,
                status="inactive",
            )
        )
    try:
        yield SqlAlchemyParentStagingStore(factory), factory, engine
    finally:
        await engine.dispose()


async def _attach(
    store: SqlAlchemyParentStagingStore,
    *,
    canonical_hash: str = "c" * 64,
    manifest_hash: str = "d" * 64,
    child_count: int = 1,
) -> None:
    await store.attach_manifest(
        _context(),
        canonical_ast_uri=f"artifact://canonical/{canonical_hash}.json",
        canonical_ast_sha256=canonical_hash,
        manifest_uri="artifact://manifest/fixed.json",
        manifest_hash=manifest_hash,
        parent_count=1,
        child_count=child_count,
    )


def test_parent_locator_schema_has_no_512_byte_ceiling() -> None:
    ddl = str(CreateTable(parent_chunks).compile(dialect=mysql.dialect()))
    assert "ast_locator LONGTEXT NOT NULL" in ddl


@pytest.mark.asyncio
async def test_identical_manifest_retry_is_a_database_noop(
    manifest_store: tuple[
        SqlAlchemyParentStagingStore,
        async_sessionmaker[AsyncSession],
        AsyncEngine,
    ],
) -> None:
    store, _, engine = manifest_store
    await _attach(store)
    updates: list[str] = []

    def record_update(
        _connection: Any,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("UPDATE DOCUMENT_VERSIONS"):
            updates.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record_update)
    try:
        await _attach(store)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_update)

    assert updates == []


@pytest.mark.asyncio
async def test_conflicting_manifest_retry_is_rejected(
    manifest_store: tuple[
        SqlAlchemyParentStagingStore,
        async_sessionmaker[AsyncSession],
        AsyncEngine,
    ],
) -> None:
    store, _, _ = manifest_store
    await _attach(store)

    with pytest.raises(ParentStagingConflict, match="Manifest"):
        await _attach(store, canonical_hash="e" * 64)


@pytest.mark.asyncio
async def test_conflicting_manifest_counts_are_rejected(
    manifest_store: tuple[
        SqlAlchemyParentStagingStore,
        async_sessionmaker[AsyncSession],
        AsyncEngine,
    ],
) -> None:
    store, _, _ = manifest_store
    await _attach(store)

    with pytest.raises(ParentStagingConflict, match="Manifest"):
        await _attach(store, child_count=2)


@pytest.mark.asyncio
async def test_conflicting_manifest_artifact_hash_is_rejected(
    manifest_store: tuple[
        SqlAlchemyParentStagingStore,
        async_sessionmaker[AsyncSession],
        AsyncEngine,
    ],
) -> None:
    store, _, _ = manifest_store
    await _attach(store)

    with pytest.raises(ParentStagingConflict, match="Manifest"):
        await _attach(store, manifest_hash="e" * 64)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version_no", 2),
        ("pipeline_version", "ingestion-v2"),
        ("embedding_version", "text-embedding-v4"),
    ],
)
@pytest.mark.asyncio
async def test_context_version_metadata_must_match_the_durable_version(
    manifest_store: tuple[
        SqlAlchemyParentStagingStore,
        async_sessionmaker[AsyncSession],
        AsyncEngine,
    ],
    field: str,
    value: object,
) -> None:
    store, _, _ = manifest_store
    context = _context().model_copy(update={field: value})

    with pytest.raises(StagingVersionError):
        await store.count(context)


@pytest.mark.asyncio
async def test_generation_rejects_a_durable_unsupported_embedding_version(
    manifest_store: tuple[
        SqlAlchemyParentStagingStore,
        async_sessionmaker[AsyncSession],
        AsyncEngine,
    ],
) -> None:
    store, factory, _ = manifest_store
    async with factory.begin() as session:
        await session.execute(
            update(document_versions)
            .where(document_versions.c.id == "version-1")
            .values(embedding_version="text-embedding-v4")
        )
    context = _context().model_copy(
        update={"embedding_version": "text-embedding-v4"}
    )

    with pytest.raises(StagingVersionError):
        await store.count(context)
