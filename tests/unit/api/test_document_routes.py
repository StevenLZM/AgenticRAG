"""Infrastructure-free HTTP contract tests for document routes."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from agentic_rag.api.documents import documents_router
from agentic_rag.api.errors import register_error_handlers
from agentic_rag.domain.models import JobStatus, UserScope
from agentic_rag.ingestion.models import UploadRejectedError
from agentic_rag.persistence.repositories import IngestionJob
from agentic_rag.safety.uploads import UploadDecision, UploadSafetyStatus


class FakeDocumentService:
    def __init__(self) -> None:
        self.scope: UserScope | None = None
        self.reject_upload = False
        self.job = IngestionJob(
            id="job-1",
            user_id="api-user",
            document_id="document-1",
            document_version_id="version-1",
            status=JobStatus.QUEUED,
        )

    async def create_upload(
        self,
        scope: UserScope,
        filename: str,
        declared_mime: str,
        content: bytes,
    ) -> IngestionJob:
        self.scope = scope
        if self.reject_upload:
            raise UploadRejectedError(
                UploadDecision(
                    status=UploadSafetyStatus.REJECTED,
                    detected_mime="application/zip",
                    content_hash=hashlib.sha256(content).hexdigest(),
                    reasons=("mime_signature_mismatch",),
                )
            )
        return self.job

    async def get_job(self, scope: UserScope, job_id: str) -> IngestionJob | None:
        self.scope = scope
        return self.job if job_id == self.job.id else None

    async def delete_document(self, scope: UserScope, document_id: str) -> bool:
        self.scope = scope
        return document_id == self.job.document_id


def _app(service: FakeDocumentService) -> FastAPI:
    app = FastAPI()
    app.state.container = SimpleNamespace(
        document_service=service,
        settings=SimpleNamespace(default_user_id="api-user"),
    )
    app.include_router(documents_router)
    register_error_handlers(app)
    return app


async def test_post_returns_202_job_contract_in_default_user_scope() -> None:
    service = FakeDocumentService()
    transport = httpx.ASGITransport(app=_app(service), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/documents",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": "job-1",
        "document_id": "document-1",
        "document_version_id": "version-1",
        "status": "queued",
    }
    assert service.scope == UserScope(user_id="api-user")


async def test_rejected_upload_uses_sanitized_415_contract() -> None:
    service = FakeDocumentService()
    service.reject_upload = True
    transport = httpx.ASGITransport(app=_app(service), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/documents",
            files={"file": ("report.pdf", b"PK\x03\x04", "application/pdf")},
        )

    assert response.status_code == 415
    assert response.json()["error_code"] == "UPLOAD_REJECTED"
    assert "mime_signature_mismatch" not in response.text


async def test_get_and_delete_hide_missing_or_out_of_scope_resources() -> None:
    service = FakeDocumentService()
    transport = httpx.ASGITransport(app=_app(service), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        found = await client.get("/v1/ingestion-jobs/job-1")
        missing = await client.get("/v1/ingestion-jobs/other-job")
        removed = await client.delete("/v1/documents/document-1")
        absent = await client.delete("/v1/documents/other-document")

    assert found.status_code == 200
    assert missing.status_code == 404
    assert removed.status_code == 204
    assert absent.status_code == 404
