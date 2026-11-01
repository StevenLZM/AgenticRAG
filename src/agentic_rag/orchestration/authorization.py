"""Reauthorize immutable source identities at freeze and disclosure boundaries."""
from typing import Any

from sqlalchemy import select, tuple_

from agentic_rag.persistence.repositories import documents, document_versions


class SourceAccessRevoked(PermissionError):
    pass


async def authorize_sources(db: Any, run: dict[str, Any], evidence: list[dict[str, Any]], *, context_sources=None) -> None:
    contextual = [e for e in evidence if e.get("source_kind") == "canonical_excerpt" or e.get("context_refs")]
    if contextual:
        if context_sources is None:
            raise SourceAccessRevoked("context source authority is required")
        from agentic_rag.domain.models import UserScope
        from agentic_rag.runtime.models import RuntimeConfigSnapshot
        from agentic_rag.query.evidence_builder import EvidenceItem, EvidenceManifestEntry, evidence_projection
        scope = UserScope(user_id=run["user_id"])
        snapshot = RuntimeConfigSnapshot.model_validate(run["data"]["snapshot"])
        try:
            for raw in contextual:
                item = EvidenceItem.model_validate(raw)
                entry = EvidenceManifestEntry(**evidence_projection(item))
                if item.source_kind == "canonical_excerpt":
                    await context_sources.authorize(entry, scope, snapshot)
                else:
                    await context_sources.authorize_parent(entry, scope, snapshot)
        except (ValueError, OSError, KeyError) as error:
            raise SourceAccessRevoked("context source access was revoked or invalid") from error
    legacy_v2 = [e for e in evidence if e.get("schema_version") == 2 and e.get("source_kind", "parent_chunk") == "parent_chunk" and not e.get("context_refs")]
    if legacy_v2:
        from agentic_rag.query.audit import authorize_repository_parent_excerpt
        from agentic_rag.query.evidence_builder import EvidenceItem, EvidenceManifestEntry, evidence_projection
        from agentic_rag.persistence.repositories import SqlAlchemyParentRepository
        from agentic_rag.domain.models import UserScope
        from agentic_rag.runtime.models import RuntimeConfigSnapshot
        scope = UserScope(user_id=run["user_id"])
        snapshot = RuntimeConfigSnapshot.model_validate(run["data"]["snapshot"])
        if db is None:
            raise SourceAccessRevoked("schema-v2 Parent requires scoped source repository")
        rows = await SqlAlchemyParentRepository(db).get_many([e["parent_id"] for e in legacy_v2], scope)
        by_id = {row.id: row for row in rows}
        try:
            for raw in legacy_v2:
                item = EvidenceItem.model_validate(raw)
                entry = EvidenceManifestEntry(**evidence_projection(item))
                authorize_repository_parent_excerpt(entry, by_id.get(entry.parent_id), scope, snapshot)
        except (ValueError, OSError, KeyError) as error:
            raise SourceAccessRevoked("schema-v2 Parent original content is unavailable or invalid") from error
    if not run["data"].get("enforce_source_access", True) or not evidence:
        return
    expected = {(e["document_id"], e["document_version_id"]) for e in evidence}
    rows = (await db.execute(select(documents.c.id, document_versions.c.id.label("version_id")).select_from(
        documents.join(document_versions, document_versions.c.document_id == documents.c.id),
    ).where(documents.c.user_id == run["user_id"], documents.c.status != "deleted", documents.c.deletion_status.is_(None),
            tuple_(documents.c.id, document_versions.c.id).in_(expected),
            document_versions.c.index_generation == run["data"]["snapshot"]["index_generation"],
    ).with_for_update(read=True))).all()
    if {(row.id, row.version_id) for row in rows} != expected:
        raise SourceAccessRevoked("frozen source access was revoked")
