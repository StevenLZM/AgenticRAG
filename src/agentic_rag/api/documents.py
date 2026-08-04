"""User-scoped document upload and ingestion-job HTTP contracts."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Request, Response, UploadFile, status
from pydantic import BaseModel

from agentic_rag.api.errors import ApiException
from agentic_rag.domain.models import JobStatus, UserScope
from agentic_rag.ingestion.models import DocumentService, UploadRejectedError
from agentic_rag.persistence.repositories import IngestionJob


class IngestionJobResponse(BaseModel):
    job_id: str
    document_id: str
    document_version_id: str
    status: JobStatus

    @classmethod
    def from_job(cls, job: IngestionJob) -> "IngestionJobResponse":
        return cls(
            job_id=job.id,
            document_id=job.document_id,
            document_version_id=job.document_version_id,
            status=job.status,
        )


documents_router = APIRouter(prefix="/v1", tags=["documents"])


def _request_service(request: Request) -> tuple[DocumentService, UserScope]:
    container = request.app.state.container
    return (
        container.document_service,
        UserScope(user_id=container.settings.default_user_id),
    )


@documents_router.post(
    "/documents",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestionJobResponse,
)
async def create_document(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF, UTF-8 text, or Excel file")],
) -> IngestionJobResponse:
    service, scope = _request_service(request)
    content = await file.read()
    try:
        job = await service.create_upload(
            scope,
            filename=file.filename or "",
            declared_mime=file.content_type or "application/octet-stream",
            content=content,
        )
    except UploadRejectedError as error:
        raise ApiException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            error_code="UPLOAD_REJECTED",
            message="The uploaded file failed the safety policy.",
            retryable=False,
        ) from error
    return IngestionJobResponse.from_job(job)


@documents_router.get(
    "/ingestion-jobs/{job_id}", response_model=IngestionJobResponse
)
async def get_ingestion_job(request: Request, job_id: str) -> IngestionJobResponse:
    service, scope = _request_service(request)
    job = await service.get_job(scope, job_id)
    if job is None:
        raise ApiException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="INGESTION_JOB_NOT_FOUND",
            message="The ingestion job was not found.",
        )
    return IngestionJobResponse.from_job(job)


@documents_router.delete(
    "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(request: Request, document_id: str) -> Response:
    service, scope = _request_service(request)
    if not await service.delete_document(scope, document_id):
        raise ApiException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="DOCUMENT_NOT_FOUND",
            message="The document was not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
