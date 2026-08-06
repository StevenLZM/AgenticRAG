"""JSON-only durable state and lease identity for document ingestion."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class ArtifactPointer(BaseModel):
    """Serializable reference to an immutable Artifact payload."""

    model_config = ConfigDict(frozen=True)

    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class IngestionJobClaim(BaseModel):
    """Fencing identity issued by the durable Job repository."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    user_id: str
    document_id: str
    document_version_id: str
    owner: str
    claim_generation: int = Field(ge=0)


class IngestionRuntime(BaseModel):
    """Trusted MySQL projection required by the fixed ingestion graph."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    user_id: str
    document_id: str
    document_version_id: str
    version_no: int = Field(gt=0)
    filename: str
    mime_type: str
    source_type: Literal["pdf", "scanned_pdf", "excel", "text"]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_version: str
    pipeline_version: str
    embedding_version: str
    index_generation: str
    source_trust: str = "untrusted"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def ingestion_key(self) -> str:
        payload = "\x1f".join(
            (
                self.user_id,
                self.document_id,
                self.content_hash,
                self.pipeline_version,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IngestionState(TypedDict, total=False):
    """Checkpoint payload. Values are JSON primitives or JSON model dumps only."""

    job_id: str
    user_id: str
    document_id: str
    document_version_id: str
    owner: str
    claim_generation: int
    runtime: dict[str, Any]
    ingestion_key: str
    source_ref: dict[str, Any]
    fragment_refs: list[dict[str, Any]]
    canonical_ref: dict[str, Any]
    chunks_ref: dict[str, Any]
    manifest_hash: str
    terminal_status: Literal["completed", "quarantined", "failed"] | None
    quarantine_reasons: list[str]
    completed_nodes: list[str]
