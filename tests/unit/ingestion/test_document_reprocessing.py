"""Reprocessing uses the same upload safety gate, version/job/outbox and worker path."""
import hashlib

import pytest
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from agentic_rag.domain.models import UserScope
from agentic_rag.ingestion.models import DocumentService, UploadVersions
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.repositories import (
    SqlAlchemyDocumentRepository, SqlAlchemyIngestionJobRepository,
    documents, document_versions, ingestion_jobs, metadata, task_outbox,
)
from agentic_rag.safety.uploads import DefaultUploadSafetyScanner


@pytest.fixture
async def service(tmp_path):
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    def make(pipeline, generation):
        return DocumentService(scanner=DefaultUploadSafetyScanner(), artifacts=artifacts,
            session_factory=factory, documents=SqlAlchemyDocumentRepository(), jobs=SqlAlchemyIngestionJobRepository(),
            versions=UploadVersions("docling-v1", pipeline, "text-embedding-v3", generation), max_upload_bytes=1024)
    old = await make("ingestion-v1", "index-v2").create_upload(UserScope(user_id="u"), "test.txt", "text/plain", b"original facts")
    async with factory.begin() as conn:
        await conn.execute(update(documents).where(documents.c.id == old.document_id).values(status="active", active_version_id=old.document_version_id))
        await conn.execute(update(document_versions).where(document_versions.c.id == old.document_version_id).values(status="active"))
        await conn.execute(update(ingestion_jobs).where(ingestion_jobs.c.id == old.id).values(status="completed"))
    try:
        yield make("ingestion-v2", "index-v3"), old, factory, artifacts
    finally:
        await engine.dispose()


async def test_reprocess_preserves_active_original_until_new_version_is_published(service):
    svc, old, factory, artifacts = service
    job = await svc.create_upload(UserScope(user_id="u"), "test.txt", "text/plain", b"original facts", reprocess_document_id=old.document_id)
    assert job.document_id == old.document_id and job.document_version_id != old.document_version_id
    async with factory() as conn:
        doc = (await conn.execute(select(documents))).mappings().one()
        version = (await conn.execute(select(document_versions).where(document_versions.c.id == job.document_version_id))).mappings().one()
        outbox = (await conn.execute(select(task_outbox).where(task_outbox.c.aggregate_id == job.id))).mappings().one()
    assert doc["status"] == "active" and doc["active_version_id"] == old.document_version_id
    assert version["version_no"] == 2 and version["status"] == "uploaded"
    assert (version["pipeline_version"], version["index_generation"]) == ("ingestion-v2", "index-v3")
    assert outbox["status"] == "pending"
    ref = artifacts.describe(f"artifact://documents/u/{old.document_id}/{job.document_version_id}/source/test.txt")
    assert ref.sha256 == hashlib.sha256(b"original facts").hexdigest()


@pytest.mark.parametrize("user,payload", [("foreign", b"original facts"), ("u", b"changed facts")])
async def test_reprocess_cannot_replace_foreign_or_changed_original(service, user, payload):
    svc, old, factory, _ = service
    with pytest.raises(ValueError):
        await svc.create_upload(UserScope(user_id=user), "test.txt", "text/plain", payload, reprocess_document_id=old.document_id)
    async with factory() as conn:
        assert await conn.scalar(select(func.count()).select_from(document_versions)) == 1


async def test_repeated_pending_reprocess_does_not_enqueue_duplicate(service):
    svc, old, factory, _ = service
    await svc.create_upload(UserScope(user_id="u"), "test.txt", "text/plain", b"original facts", reprocess_document_id=old.document_id)
    with pytest.raises(ValueError):
        await svc.create_upload(UserScope(user_id="u"), "test.txt", "text/plain", b"original facts", reprocess_document_id=old.document_id)
    async with factory() as conn:
        assert await conn.scalar(select(func.count()).select_from(document_versions)) == 2
        assert await conn.scalar(select(func.count()).select_from(ingestion_jobs)) == 2
