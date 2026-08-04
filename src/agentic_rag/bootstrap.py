"""Explicit construction and lifecycle ownership for application dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from elasticsearch import AsyncElasticsearch
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentic_rag.api.health import ReadinessChecks, build_readiness_checks
from agentic_rag.config import Settings
from agentic_rag.persistence.artifacts import ArtifactStore, LocalArtifactStore
from agentic_rag.persistence.checkpoint import CheckpointBackend
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.persistence.redis_queue import RedisStreamsBroker, StreamBroker


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
    mysql_engine: AsyncEngine
    redis: Redis
    reranker_initialized: bool

    async def close(self) -> None:
        """Attempt cleanup of every process-owned async client."""
        await asyncio.gather(
            self.elasticsearch.close(),
            self.redis.aclose(),
            self.mysql_engine.dispose(),
            return_exceptions=True,
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
    return AppContainer(
        settings=settings,
        repositories=Repositories(create_session_factory(mysql_engine)),
        broker=RedisStreamsBroker(redis),
        artifacts=artifacts,
        checkpoints=checkpoints,
        elasticsearch=elasticsearch,
        readiness_checks=readiness_checks,
        mysql_engine=mysql_engine,
        redis=redis,
        reranker_initialized=reranker_initialized,
    )
