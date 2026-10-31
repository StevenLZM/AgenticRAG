"""Reauthorize immutable source identities at freeze and disclosure boundaries."""
from typing import Any

from sqlalchemy import select, tuple_

from agentic_rag.persistence.repositories import documents, document_versions


class SourceAccessRevoked(PermissionError):
    pass


async def authorize_sources(db: Any, run: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
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
