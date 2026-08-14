"""Explicit construction and lifecycle ownership for application dependencies."""

from __future__ import annotations

import asyncio
import logging
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
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.runtime.query_composition import build_query_snapshot
from agentic_rag.domain.models import UserScope
from agentic_rag.memory.factory import (
    MemoryCompositionError,
    UnavailableMemoryService,
    build_memory_service_sync,
)
from agentic_rag.memory.service import MemoryService
from agentic_rag.safety.uploads import DefaultUploadSafetyScanner

logger = logging.getLogger(__name__)


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
    runtime_snapshot: RuntimeConfigSnapshot
    run_manager: RunManager | None = None
    event_repository: object | None = None
    memory_service: MemoryService | None = None

    async def close(self) -> None:
        """Attempt cleanup of every process-owned async client."""
        memory_resources = tuple(
            getattr(self.memory_service, "_owned_resources", ())
            if self.memory_service is not None
            else ()
        )

        async def close_resource(resource: object) -> None:
            close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
            if not callable(close):
                return
            value = close()
            if asyncio.iscoroutine(value):
                await value

        await asyncio.gather(
            self.elasticsearch.close(),
            self.redis.aclose(),
            self.mysql_engine.dispose(),
            *(close_resource(resource) for resource in reversed(memory_resources)),
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
    runtime_snapshot = build_query_snapshot(settings)
    repositories = Repositories(create_session_factory(mysql_engine))
    event_repository = _TransactionalEventRepository(repositories.session_factory)
    memory_container = type("MemoryContainer", (), {"repositories": repositories})()
    try:
        memory_service = build_memory_service_sync(
            memory_container, settings, runtime_snapshot
        )
    except MemoryCompositionError as error:
        # Core query APIs remain available with memory degraded, while the
        # readiness check and structured log make the operator action explicit.
        logger.error(
            "memory_provider_degraded",
            extra={
                "component": "mem0",
                "reason": type(error).__name__,
                "outcome": "degraded",
                "retryable": False,
            },
        )
        memory_service = UnavailableMemoryService(type(error).__name__)
    memory_available = bool(getattr(memory_service, "available", False))
    readiness_checks = build_readiness_checks(
        settings=settings,
        mysql=mysql_engine,
        redis=redis,
        elasticsearch=elasticsearch,
        artifacts=artifacts,
        checkpoints=checkpoints,
        reranker_initialized=reranker_initialized,
        memory_available=memory_available or not settings.mem0_enabled,
    )

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
        runtime_snapshot=runtime_snapshot,
        run_manager=run_manager,
        event_repository=event_repository,
        memory_service=memory_service,
    )
