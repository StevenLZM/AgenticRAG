#!/usr/bin/env python3
"""Run the single local ingestion worker and its in-process reconciler."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from docling_core.transforms.chunker.tokenizer.huggingface import (  # noqa: E402
    HuggingFaceTokenizer,
)
from openai import AsyncOpenAI  # type: ignore[import-not-found] # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker  # noqa: E402

from agentic_rag.bootstrap import build_container  # noqa: E402
from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.ingestion.assembler import GlobalAssembler  # noqa: E402
from agentic_rag.ingestion.chunker import CHILD_MAX_TOKENS, ChunkingPipeline  # noqa: E402
from agentic_rag.ingestion.graph import (  # noqa: E402
    DefaultIngestionPipeline,
    IngestionGraph,
)
from agentic_rag.ingestion.indexer import IndexWriter  # noqa: E402
from agentic_rag.ingestion.parser import DocumentParser  # noqa: E402
from agentic_rag.ingestion.publisher import VersionPublisher  # noqa: E402
from agentic_rag.ingestion.reconciler import IngestionReconciler  # noqa: E402
from agentic_rag.ingestion.worker import (  # noqa: E402
    IngestionWorker,
    SqlAlchemyIngestionJobStore,
)
from agentic_rag.persistence.elasticsearch import (  # noqa: E402
    ElasticsearchChildIndexStore,
)
from agentic_rag.persistence.lifecycle import (  # noqa: E402
    SqlAlchemyPublicationRepository,
    SqlAlchemyReconciliationRepository,
)
from agentic_rag.persistence.artifacts import LocalArtifactStore  # noqa: E402
from agentic_rag.persistence.outbox import OutboxDispatcher  # noqa: E402
from agentic_rag.persistence.repositories import (  # noqa: E402
    OutboxRecord,
    SqlAlchemyOutboxRepository,
)
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore  # noqa: E402
from agentic_rag.safety.content import ContentSafetyScanner  # noqa: E402
from agentic_rag.safety.uploads import DefaultUploadSafetyScanner  # noqa: E402


class QwenEmbeddingAdapter:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._client = AsyncOpenAI(
            base_url=settings.qwen_embedding_base_url,
            api_key=(
                settings.qwen_api_key.get_secret_value()
                if settings.qwen_api_key is not None
                else "local"
            ),
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self._dimensions,
        )
        return [list(item.embedding) for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def close(self) -> None:
        await self._client.close()


class TransactionalOutboxAdapter:
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


async def run(settings: Settings) -> None:
    container = build_container(settings)
    artifacts = cast(LocalArtifactStore, container.artifacts)
    embedding = QwenEmbeddingAdapter(settings)
    factory = container.repositories.session_factory
    jobs = SqlAlchemyIngestionJobStore(factory)
    parent_store = SqlAlchemyParentStagingStore(factory)
    child_store = ElasticsearchChildIndexStore(container.elasticsearch)
    publisher = VersionPublisher(
        repository=SqlAlchemyPublicationRepository(factory),
        parent_store=parent_store,
        child_store=child_store,
        artifacts=artifacts,
    )
    reconciler = IngestionReconciler(
        repository=SqlAlchemyReconciliationRepository(factory),
        publisher=publisher,
        dispatcher=OutboxDispatcher(
            TransactionalOutboxAdapter(factory), container.broker
        ),
        parent_store=parent_store,
        child_store=child_store,
        artifacts=artifacts,
    )
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", max_tokens=CHILD_MAX_TOKENS
    )
    parser = DocumentParser(
        source_loader=artifacts.read_bytes,
        artifacts=artifacts,
        max_file_size=settings.max_upload_bytes,
    )
    pipeline = DefaultIngestionPipeline(
        jobs=jobs,
        artifacts=artifacts,
        upload_scanner=DefaultUploadSafetyScanner(
            max_upload_bytes=settings.max_upload_bytes
        ),
        parser=parser,
        assembler=GlobalAssembler(artifacts=artifacts),
        content_scanner=ContentSafetyScanner(),
        chunker=ChunkingPipeline(tokenizer),
        index_writer=IndexWriter(
            embedding=embedding,
            parent_store=parent_store,
            child_store=child_store,
            artifacts=artifacts,
        ),
        publisher=publisher,
    )
    try:
        async with container.checkpoints.open_ingestion() as checkpointer:
            graph = IngestionGraph(pipeline=pipeline, checkpointer=checkpointer)
            worker = IngestionWorker(
                jobs=jobs,
                broker=container.broker,
                graph=graph,
                reconciler=reconciler,
                worker_id=f"{socket.gethostname()}:{os.getpid()}",
            )
            await worker.run_forever()
    finally:
        await embedding.close()
        await container.close()


def main() -> int:
    try:
        asyncio.run(run(Settings()))  # type: ignore[call-arg]
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
