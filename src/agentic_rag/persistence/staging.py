"""MySQL adapter for idempotent, inactive Parent staging."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import DocumentVersionStatus
from agentic_rag.ingestion.chunker import ParentChunk
from agentic_rag.ingestion.indexer import StagingContext
from agentic_rag.persistence.repositories import (
    document_versions,
    documents,
    parent_chunks,
)


class StagingVersionError(RuntimeError):
    """Raised when the durable version does not match the trusted context."""


class ParentStagingConflict(RuntimeError):
    """Raised when deterministic IDs resolve to different stored Parent data."""


class SqlAlchemyParentStagingStore:
    """Stage inactive Parents using one transaction per idempotent operation."""

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        self._session_factory = session_factory

    async def stage(
        self, context: StagingContext, parents: Sequence[ParentChunk]
    ) -> int:
        expected = tuple(parents)
        async with self._session_factory.begin() as session:
            await self._require_version(session, context)
            for parent in expected:
                values = _parent_values(context, parent)
                statement = mysql_insert(parent_chunks).values(**values)
                await session.execute(
                    statement.on_duplicate_key_update(id=parent_chunks.c.id)
                )
            await session.execute(
                update(document_versions)
                .where(document_versions.c.id == context.document_version_id)
                .values(status=DocumentVersionStatus.BUILDING.value)
            )
            stored_rows = (
                (
                    await session.execute(
                        select(parent_chunks).where(
                            parent_chunks.c.document_version_id
                            == context.document_version_id
                        )
                    )
                )
                .mappings()
                .all()
            )
            rows = [dict(row) for row in stored_rows]
            self._require_exact_rows(context, expected, rows)
            return len(rows)

    async def count(self, context: StagingContext) -> int:
        async with self._session_factory() as session:
            await self._require_version(session, context)
            return int(
                (
                    await session.execute(
                        select(func.count())
                        .select_from(parent_chunks)
                        .where(
                            parent_chunks.c.user_id == context.user_id,
                            parent_chunks.c.document_id == context.document_id,
                            parent_chunks.c.document_version_id
                            == context.document_version_id,
                            parent_chunks.c.status == "inactive",
                        )
                    )
                ).scalar_one()
            )

    async def attach_manifest(
        self,
        context: StagingContext,
        *,
        canonical_ast_uri: str,
        canonical_ast_sha256: str,
        manifest_uri: str,
        manifest_hash: str,
        parent_count: int,
        child_count: int,
    ) -> None:
        async with self._session_factory.begin() as session:
            await self._require_version(session, context)
            actual_parent_count = int(
                (
                    await session.execute(
                        select(func.count())
                        .select_from(parent_chunks)
                        .where(
                            parent_chunks.c.user_id == context.user_id,
                            parent_chunks.c.document_id == context.document_id,
                            parent_chunks.c.document_version_id
                            == context.document_version_id,
                            parent_chunks.c.status == "inactive",
                        )
                    )
                ).scalar_one()
            )
            if actual_parent_count != parent_count:
                raise ParentStagingConflict(
                    "Parent count changed before Manifest attachment"
                )
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(document_versions)
                    .where(document_versions.c.id == context.document_version_id)
                    .values(
                        canonical_ast_path=canonical_ast_uri,
                        canonical_ast_hash=canonical_ast_sha256,
                        manifest_path=manifest_uri,
                        manifest_hash=manifest_hash,
                        parent_count=parent_count,
                        child_count=child_count,
                        status=DocumentVersionStatus.BUILDING.value,
                    )
                ),
            )
            if result.rowcount != 1:
                raise StagingVersionError(context.document_version_id)

    async def _require_version(
        self, session: AsyncSession, context: StagingContext
    ) -> None:
        row = (
            await session.execute(
                select(
                    document_versions.c.id,
                    document_versions.c.embedding_version,
                    document_versions.c.index_generation,
                    document_versions.c.status,
                )
                .select_from(
                    document_versions.join(
                        documents,
                        document_versions.c.document_id == documents.c.id,
                    )
                )
                .where(
                    document_versions.c.id == context.document_version_id,
                    document_versions.c.document_id == context.document_id,
                    documents.c.user_id == context.user_id,
                    documents.c.status != "deleted",
                )
            )
        ).mappings().one_or_none()
        if row is None:
            raise StagingVersionError(
                "document version is absent or outside the trusted user scope"
            )
        if (
            row["embedding_version"] != "text-embedding-v3"
            or row["index_generation"] != context.index_generation
            or row["status"]
            not in {
                DocumentVersionStatus.UPLOADED.value,
                DocumentVersionStatus.BUILDING.value,
            }
        ):
            raise StagingVersionError(
                "document version configuration/status does not permit staging"
            )

    @staticmethod
    def _require_exact_rows(
        context: StagingContext,
        parents: tuple[ParentChunk, ...],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        expected = {
            parent.id: _parent_values(context, parent) for parent in parents
        }
        actual = {cast(str, row["id"]): dict(row) for row in rows}
        if set(actual) != set(expected):
            raise ParentStagingConflict(
                "stored Parent IDs/count do not match this deterministic version"
            )
        compared_fields = tuple(next(iter(expected.values()))) if expected else ()
        for parent_id, values in expected.items():
            if any(actual[parent_id][field] != values[field] for field in compared_fields):
                raise ParentStagingConflict(
                    f"stored Parent {parent_id!r} conflicts with deterministic payload"
                )


def _parent_values(
    context: StagingContext, parent: ParentChunk
) -> dict[str, Any]:
    return {
        "id": parent.id,
        "user_id": context.user_id,
        "document_id": context.document_id,
        "document_version_id": context.document_version_id,
        "ordinal": parent.ordinal,
        "heading_path": list(parent.heading_path),
        "content_type": parent.content_type,
        "content": parent.content,
        "page_from": parent.page_from,
        "page_to": parent.page_to,
        "ast_locator": json.dumps(
            parent.ast_locator.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "content_hash": parent.content_hash,
        "status": "inactive",
    }
