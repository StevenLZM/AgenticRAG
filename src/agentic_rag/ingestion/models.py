"""Upload orchestration without parser or graph implementation details."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import DocumentVersionStatus, JobStatus, UserScope
from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore
from agentic_rag.persistence.repositories import (
    DocumentRepository,
    IngestionJob,
    IngestionJobRepository,
)
from agentic_rag.runtime.ids import new_id
from agentic_rag.safety.uploads import (
    UploadDecision,
    UploadSafetyScanner,
    UploadSafetyStatus,
    normalize_upload_filename,
)


@dataclass(frozen=True, slots=True)
class UploadVersions:
    parser: str
    pipeline: str
    embedding: str
    index_generation: str


class UploadRejectedError(ValueError):
    """Raised before persistence when an upload fails the safety allowlist."""

    def __init__(self, decision: UploadDecision) -> None:
        super().__init__("upload rejected by safety policy")
        self.decision = decision


class UploadTooLargeError(ValueError):
    """Raised before scanning or persistence when a byte limit is exceeded."""

    def __init__(self) -> None:
        super().__init__("upload exceeds configured size limit")


class DocumentService:
    """Coordinate artifact durability with one caller-owned MySQL transaction."""

    def __init__(
        self,
        *,
        scanner: UploadSafetyScanner,
        artifacts: ArtifactStore,
        session_factory: async_sessionmaker[AsyncSession],
        documents: DocumentRepository,
        jobs: IngestionJobRepository,
        versions: UploadVersions,
        max_upload_bytes: int,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("maximum upload size must be positive")
        self._scanner = scanner
        self._artifacts = artifacts
        self._session_factory = session_factory
        self._documents = documents
        self._jobs = jobs
        self._versions = versions
        self._max_upload_bytes = max_upload_bytes

    async def create_upload(
        self,
        scope: UserScope,
        filename: str,
        declared_mime: str,
        content: bytes,
    ) -> IngestionJob:
        if len(content) > self._max_upload_bytes:
            raise UploadTooLargeError()
        decision = await asyncio.to_thread(
            self._scanner.scan, filename, declared_mime, content
        )
        if decision.status is UploadSafetyStatus.REJECTED:
            raise UploadRejectedError(decision)

        normalized_filename = normalize_upload_filename(filename)
        document_id = new_id()
        document_version_id = new_id()
        artifact_path = (
            f"documents/{scope.user_id}/{document_id}/{document_version_id}/"
            f"source/{normalized_filename}"
        )
        artifact_ref = await asyncio.to_thread(
            self._artifacts.put_bytes, artifact_path, content
        )
        try:
            async with self._session_factory.begin() as transaction:
                _document, version = await self._documents.create(
                    scope,
                    source_type=_source_type(decision.detected_mime),
                    filename=normalized_filename,
                    mime_type=decision.detected_mime,
                    content_hash=decision.content_hash,
                    parser_version=self._versions.parser,
                    pipeline_version=self._versions.pipeline,
                    embedding_version=self._versions.embedding,
                    index_generation=self._versions.index_generation,
                    document_id=document_id,
                    document_version_id=document_version_id,
                    version_status=(
                        DocumentVersionStatus.QUARANTINED
                        if decision.status is UploadSafetyStatus.QUARANTINED
                        else DocumentVersionStatus.UPLOADED
                    ),
                    transaction=transaction,
                )
                return await self._jobs.create(
                    scope,
                    document_id,
                    version.id,
                    status=(
                        JobStatus.QUARANTINED
                        if decision.status is UploadSafetyStatus.QUARANTINED
                        else JobStatus.QUEUED
                    ),
                    transaction=transaction,
                )
        except BaseException:
            await self._delete_unreferenced(artifact_ref)
            raise

    async def get_job(self, scope: UserScope, job_id: str) -> IngestionJob | None:
        async with self._session_factory.begin() as transaction:
            return await self._jobs.get(job_id, scope, transaction=transaction)

    async def delete_document(self, scope: UserScope, document_id: str) -> bool:
        async with self._session_factory.begin() as transaction:
            return await self._documents.soft_delete(
                document_id, scope, transaction=transaction
            )

    async def _delete_unreferenced(self, artifact_ref: ArtifactRef) -> None:
        try:
            await asyncio.to_thread(self._artifacts.delete, artifact_ref)
        except Exception:
            # Preserve the database exception. The unique path remains safe for a
            # later artifact reconciler to remove if local cleanup itself failed.
            pass


def _source_type(mime_type: str) -> str:
    if mime_type == "application/pdf":
        return "pdf"
    if mime_type == "text/plain":
        return "text"
    return "excel"
