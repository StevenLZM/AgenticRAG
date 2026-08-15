"""Opt-in production-composition ingestion E2E against disposable local services."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from alembic import command
from alembic.config import Config
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import delete, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast

from agentic_rag.bootstrap import AppContainer, build_container
from agentic_rag.config import Settings
from agentic_rag.domain.models import UserScope
from agentic_rag.ingestion.assembler import GlobalAssembler
from agentic_rag.ingestion.chunker import ChunkingPipeline
from agentic_rag.ingestion.graph import DefaultIngestionPipeline, IngestionGraph
from agentic_rag.ingestion.indexer import IndexWriter
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.ingestion.parser import DocumentParser
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.ingestion.worker import (
    INGESTION_GROUP,
    INGESTION_STREAM,
    IngestionWorker,
    SqlAlchemyIngestionJobStore,
)
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
from agentic_rag.persistence.outbox import OutboxDispatcher
from agentic_rag.persistence.repositories import (
    OutboxRecord,
    SqlAlchemyOutboxRepository,
    documents,
    ingestion_jobs,
    parent_chunks,
)
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
from agentic_rag.safety.content import ContentSafetyScanner
from agentic_rag.safety.uploads import DefaultUploadSafetyScanner

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def _tokenizer() -> HuggingFaceTokenizer:
    core = Tokenizer(WordLevel(vocab={"[UNK]": 0, "Hello": 1}, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    return HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=384)


def _minimal_pdf(text: str) -> bytes:
    stream = b"BT /F1 12 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode())
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(payload)


def _scanned_text_pdf(text: str) -> bytes:
    image = Image.new("RGB", (2000, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 150), text, fill="black", font=ImageFont.load_default(size=96))
    payload = BytesIO()
    image.save(payload, format="PDF", resolution=200)
    return payload.getvalue()


def _minimal_xlsx() -> bytes:
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Hello</t></is></c></row></sheetData></worksheet>',
        )
    return payload.getvalue()


class _FixedEmbedding:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index % 11) / 11 for index in range(1024)] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1024


class _TransactionalOutbox:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def list_pending(self, limit: int) -> list[OutboxRecord]:
        async with self._factory() as session:
            return await SqlAlchemyOutboxRepository(session).list_pending(limit)

    async def claim_pending(self, limit: int) -> list[OutboxRecord]:
        async with self._factory.begin() as session:
            return await SqlAlchemyOutboxRepository(session).claim_pending(limit)

    async def mark_dispatched(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).mark_dispatched(outbox_id)

    async def schedule_retry(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).schedule_retry(outbox_id)


class _FailOnceAfterStaging:
    def __init__(self, delegate: IndexWriter) -> None:
        self._delegate = delegate
        self._failed = False

    async def stage(self, *args: Any, **kwargs: Any) -> VersionManifest:
        result = await self._delegate.stage(*args, **kwargs)
        if not self._failed:
            self._failed = True
            raise RuntimeError("injected crash after real cross-store staging")
        return result


def _explicit_local_services(tmp_path: Path) -> Settings:
    mysql_dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN")
    redis_dsn = os.getenv("AGENTIC_RAG_TEST_REDIS_DSN")
    elasticsearch_url = os.getenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL")
    if not mysql_dsn or not redis_dsn or not elasticsearch_url:
        pytest.skip(
            "set AGENTIC_RAG_TEST_MYSQL_DSN, AGENTIC_RAG_TEST_REDIS_DSN and "
            "AGENTIC_RAG_TEST_ELASTICSEARCH_URL to disposable local services"
        )
    if make_url(mysql_dsn).host not in {"127.0.0.1", "::1", "localhost"}:
        pytest.skip("real ingestion E2E MySQL must be an explicit local endpoint")
    for value, schemes in (
        (redis_dsn, {"redis", "rediss"}),
        (elasticsearch_url, {"http", "https"}),
    ):
        parsed = urlparse(value)
        if parsed.scheme not in schemes or parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            pytest.skip("real ingestion E2E services must be explicit local endpoints")
    return Settings(
        mysql_dsn=mysql_dsn,
        redis_url=redis_dsn,
        elasticsearch_url=elasticsearch_url,
        deepseek_base_url="http://127.0.0.1:1/v1",
        qwen_embedding_base_url="http://127.0.0.1:1/v1",
        index_generation=f"e2e-{uuid4().hex}",
        artifact_root=tmp_path / "artifacts",
        query_checkpoint_path=tmp_path / "query.sqlite",
        ingestion_checkpoint_path=tmp_path / "ingestion.sqlite",
    )


@pytest.fixture
async def real_container(tmp_path: Path) -> AsyncIterator[AppContainer]:
    settings = _explicit_local_services(tmp_path)
    migration = Config("alembic.ini")
    migration.set_main_option("sqlalchemy.url", settings.mysql_dsn)
    await asyncio.to_thread(command.upgrade, migration, "head")
    container = build_container(settings)
    # Configured-but-unavailable infrastructure is a test failure, not a skip.
    await container.redis.ping()
    await container.elasticsearch.info()
    async with container.mysql_engine.connect() as connection:
        await connection.execute(select(1))
    try:
        yield container
    finally:
        await container.elasticsearch.indices.delete(
            index=f"agenticrag-children-{settings.index_generation}",
            ignore_unavailable=True,
        )
        await container.redis.delete(
            INGESTION_STREAM,
            f"{INGESTION_STREAM}:dedupe",
            f"{INGESTION_STREAM}:dead",
            f"{INGESTION_STREAM}:dead:dedupe",
        )
        await container.close()


@pytest.mark.asyncio
async def test_real_four_format_worker_resume_reaches_active_without_duplicates(
    real_container: AppContainer,
) -> None:
    container = real_container
    factory = container.repositories.session_factory
    user_id = f"e2e-ingestion-{uuid4().hex}"
    scope = UserScope(user_id=user_id)
    uploads = (
        ("notes.txt", "text/plain", b"Hello production ingestion"),
        (
            "table.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            _minimal_xlsx(),
        ),
        ("report.pdf", "application/pdf", _minimal_pdf("Hello production")),
        ("scan.pdf", "application/pdf", _scanned_text_pdf("HELLO DOCUMENT")),
    )
    jobs = [
        await container.document_service.create_upload(scope, name, mime, payload)
        for name, mime, payload in uploads
    ]
    job_store = SqlAlchemyIngestionJobStore(factory)
    parent_store = SqlAlchemyParentStagingStore(factory)
    child_store = ElasticsearchChildIndexStore(container.elasticsearch)
    artifacts = cast(LocalArtifactStore, container.artifacts)
    publisher = VersionPublisher(
        repository=SqlAlchemyPublicationRepository(factory),
        parent_store=parent_store,
        child_store=child_store,
        artifacts=artifacts,
    )
    writer = _FailOnceAfterStaging(
        IndexWriter(
            embedding=_FixedEmbedding(),
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        )
    )
    pipeline = DefaultIngestionPipeline(
        jobs=job_store,
        artifacts=artifacts,
        upload_scanner=DefaultUploadSafetyScanner(
            max_upload_bytes=container.settings.max_upload_bytes
        ),
        parser=DocumentParser(
            source_loader=artifacts.read_bytes,
            artifacts=artifacts,
        ),
        assembler=GlobalAssembler(),
        content_scanner=ContentSafetyScanner(),
        chunker=ChunkingPipeline(_tokenizer()),
        index_writer=writer,  # type: ignore[arg-type]
        publisher=publisher,
    )
    dispatcher = OutboxDispatcher(_TransactionalOutbox(factory), container.broker)
    try:
        assert await dispatcher.dispatch_once() == 4
        messages = await container.broker.consume(
            INGESTION_STREAM, INGESTION_GROUP, "e2e-worker", 100
        )
        assert {message.aggregate_id for message in messages} == {
            job.id for job in jobs
        }

        async with container.checkpoints.open_ingestion() as checkpointer:
            graph = IngestionGraph(pipeline=pipeline, checkpointer=checkpointer)
            worker = IngestionWorker(
                jobs=job_store,
                broker=container.broker,
                graph=graph,
                reconciler=None,
                worker_id="e2e-worker",
                heartbeat_interval_seconds=0.1,
            )
            first, *rest = messages
            await worker.process_message(first)
            assert await job_store.get_status(first.aggregate_id) is not None
            await worker.process_message(first)
            for message in rest:
                await worker.process_message(message)

        async with factory() as session:
            document_rows = (
                (
                    await session.execute(
                        select(documents.c.id, documents.c.status).where(
                            documents.c.user_id == user_id
                        )
                    )
                )
                .mappings()
                .all()
            )
            job_statuses = (
                (
                    await session.execute(
                        select(ingestion_jobs.c.status).where(
                            ingestion_jobs.c.user_id == user_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            total_parents = await session.scalar(
                select(func.count())
                .select_from(parent_chunks)
                .where(parent_chunks.c.user_id == user_id)
            )
            distinct_parents = await session.scalar(
                select(func.count(func.distinct(parent_chunks.c.id))).where(
                    parent_chunks.c.user_id == user_id
                )
            )
        assert len(document_rows) == 4
        assert {row["status"] for row in document_rows} == {"active"}
        assert set(job_statuses) == {"completed"}
        assert total_parents == distinct_parents
        response = await container.elasticsearch.count(
            index=child_store.index_name(container.settings.index_generation),
            query={"term": {"user_id": user_id}},
        )
        assert response["count"] > 0
    finally:
        async with factory.begin() as session:
            await session.execute(
                delete(documents).where(documents.c.user_id == user_id)
            )
