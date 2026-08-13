"""Explicit construction and lifecycle ownership for application dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from elasticsearch import AsyncElasticsearch
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentic_rag.api.health import ReadinessChecks, build_readiness_checks
from agentic_rag.config import Settings
from agentic_rag.ingestion.models import DocumentService, UploadVersions
from agentic_rag.persistence.artifacts import ArtifactStore, LocalArtifactStore
from agentic_rag.persistence.checkpoint import CheckpointBackend
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence.redis_queue import RedisStreamsBroker, StreamBroker
from agentic_rag.persistence.repositories import (
    AgentEvent,
    SqlAlchemyDocumentRepository,
    SqlAlchemyEventRepository,
    SqlAlchemyIngestionJobRepository,
)
from agentic_rag.runtime.run_manager import RunManager, TransactionalRunRepository
from agentic_rag.memory.models import MemoryContext, MemoryRecord
from agentic_rag.memory.service import MemoryService
from agentic_rag.domain.models import UserScope
from agentic_rag.safety.uploads import DefaultUploadSafetyScanner


@dataclass(frozen=True, slots=True)
class Repositories:
    """Thin owner of the session factory used by transaction-scoped repositories."""

    session_factory: async_sessionmaker[AsyncSession]


@dataclass(slots=True)
class AppContainer:
    """Process-owned boundaries; this object must never be placed in graph state."""

    settings: Settings
    repositories: Repositories
    broker: StreamBroker
    artifacts: ArtifactStore
    checkpoints: CheckpointBackend
    elasticsearch: AsyncElasticsearch
    readiness_checks: ReadinessChecks
    document_service: DocumentService
    mysql_engine: AsyncEngine
    redis: Redis
    reranker_initialized: bool
    run_manager: RunManager | None = None
    event_repository: object | None = None
    memory_service: MemoryService | None = None

    async def close(self) -> None:
        """Attempt cleanup of every process-owned async client."""
        await asyncio.gather(
            self.elasticsearch.close(),
            self.redis.aclose(),
            self.mysql_engine.dispose(),
            return_exceptions=True,
        )


class _TransactionalEventRepository:
    """Open a short session per API/graph event operation."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def append(self, event: AgentEvent) -> int:
        async with self._factory.begin() as session:
            return await SqlAlchemyEventRepository(session).append(event)

    async def list_after(
        self, run_id: str, scope: UserScope, after_id: int, limit: int
    ) -> list[AgentEvent]:
        async with self._factory() as session:
            return await SqlAlchemyEventRepository(session).list_after(
                run_id, scope, after_id, limit
            )


def build_container(settings: Settings) -> AppContainer:
    """Construct adapters without opening network connections or calling models."""
    mysql_engine = create_mysql_engine(settings.mysql_dsn, pool_pre_ping=True)
    redis = Redis.from_url(settings.redis_url)
    elasticsearch = AsyncElasticsearch(settings.elasticsearch_url)
    artifacts = LocalArtifactStore(settings.artifact_root)
    checkpoints = CheckpointBackend(settings)
    reranker_initialized = bool(settings.reranker_model.strip())
    readiness_checks = build_readiness_checks(
        settings=settings,
        mysql=mysql_engine,
        redis=redis,
        elasticsearch=elasticsearch,
        artifacts=artifacts,
        checkpoints=checkpoints,
        reranker_initialized=reranker_initialized,
    )
    repositories = Repositories(create_session_factory(mysql_engine))
    event_repository = _TransactionalEventRepository(repositories.session_factory)
    # Mem0 construction belongs to deployment composition.  Until a client is
    # injected, this safe no-op implementation exposes no cross-user data and
    # keeps query creation/status APIs available for local health checks.
    class _UnavailableMemory:
        async def load_context(self, scope: UserScope, query: str, limit: int = 10) -> MemoryContext:
            del scope, query, limit
            return MemoryContext(degraded=True)

        async def extract_and_store(self, scope: UserScope, run_id: str, messages: object) -> None:
            del scope, run_id, messages

        async def list(self, scope: UserScope) -> list[MemoryRecord]:
            del scope
            raise OSError("memory provider is not configured")

        async def delete(self, scope: UserScope, memory_id: str) -> None:
            del scope, memory_id
            raise OSError("memory provider is not configured")

        async def reconcile_deletions(self) -> None:
            return None

    run_manager = RunManager(
        session_factory=repositories.session_factory,
        runs=TransactionalRunRepository(repositories.session_factory),
    )
    document_service = DocumentService(
        scanner=DefaultUploadSafetyScanner(
            max_upload_bytes=settings.max_upload_bytes
        ),
        artifacts=artifacts,
        session_factory=repositories.session_factory,
        documents=SqlAlchemyDocumentRepository(),
        jobs=SqlAlchemyIngestionJobRepository(),
        max_upload_bytes=settings.max_upload_bytes,
        versions=UploadVersions(
            parser=settings.parser_version,
            pipeline=settings.ingestion_pipeline_version,
            embedding=settings.embedding_model,
            index_generation=settings.index_generation,
        ),
    )
    return AppContainer(
        settings=settings,
        repositories=repositories,
        broker=RedisStreamsBroker(redis),
        artifacts=artifacts,
        checkpoints=checkpoints,
        elasticsearch=elasticsearch,
        readiness_checks=readiness_checks,
        document_service=document_service,
        mysql_engine=mysql_engine,
        redis=redis,
        reranker_initialized=reranker_initialized,
        run_manager=run_manager,
        event_repository=event_repository,
        memory_service=_UnavailableMemory(),  # type: ignore[arg-type]
    )
