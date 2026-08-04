"""Liveness and dependency-based readiness contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Literal

from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_rag.config import Settings
from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore
from agentic_rag.persistence.checkpoint import CheckpointBackend
from agentic_rag.runtime.ids import new_id


DependencyStatus = Literal["available", "unavailable"]
HealthCheck = Callable[[], Awaitable[None]]


class HealthResponse(BaseModel):
    status: Literal["live", "ready", "unready"]
    dependencies: dict[str, DependencyStatus] | None = None


class ReadinessChecks:
    """Run independently injected dependency checks with a bounded wait."""

    def __init__(
        self, checks: Mapping[str, HealthCheck], *, timeout_seconds: float = 5.0
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("health check timeout must be positive")
        self._checks = dict(checks)
        self._timeout_seconds = timeout_seconds

    async def run(self) -> dict[str, DependencyStatus]:
        names = list(self._checks)
        states = await asyncio.gather(
            *(self._run_one(self._checks[name]) for name in names)
        )
        return dict(zip(names, states, strict=True))

    async def _run_one(self, check: HealthCheck) -> DependencyStatus:
        try:
            await asyncio.wait_for(check(), timeout=self._timeout_seconds)
        except Exception:
            return "unavailable"
        return "available"


async def check_configuration(settings: Settings) -> None:
    """Validate required local configuration without calling model providers."""
    configured_values = (
        settings.mysql_dsn,
        settings.redis_url,
        settings.elasticsearch_url,
        settings.deepseek_base_url,
        settings.qwen_embedding_base_url,
    )
    credentials = (settings.deepseek_api_key, settings.qwen_api_key)
    if not all(value.strip() for value in configured_values) or not all(
        secret is not None and secret.get_secret_value().strip()
        for secret in credentials
    ):
        raise RuntimeError("required model credentials are not configured")


async def check_mysql(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT 1"))
        if result.scalar_one() != 1:
            raise RuntimeError("MySQL health query returned an unexpected result")


async def check_redis(client: Redis) -> None:
    if not await client.ping():
        raise RuntimeError("Redis PING returned false")


async def check_elasticsearch(client: AsyncElasticsearch) -> None:
    await client.cluster.health()


def _artifact_round_trip(store: ArtifactStore) -> None:
    probe_id = new_id()
    payload = {"probe_id": probe_id}
    ref: ArtifactRef | None = None
    try:
        ref = store.put_json(f"_health/{probe_id}.json", payload)
        if store.read_json(ref) != payload:
            raise RuntimeError("artifact health round trip did not preserve content")
    finally:
        if ref is not None:
            store.delete(ref)


async def check_artifacts(store: ArtifactStore) -> None:
    await asyncio.to_thread(_artifact_round_trip, store)


async def check_checkpoints(checkpoints: CheckpointBackend) -> None:
    async with checkpoints.open_query():
        pass
    async with checkpoints.open_ingestion():
        pass


async def check_reranker(initialized: bool) -> None:
    if not initialized:
        raise RuntimeError("reranker is not initialized")


def build_readiness_checks(
    *,
    settings: Settings,
    mysql: AsyncEngine,
    redis: Redis,
    elasticsearch: AsyncElasticsearch,
    artifacts: ArtifactStore,
    checkpoints: CheckpointBackend,
    reranker_initialized: bool,
) -> ReadinessChecks:
    """Bind real adapters to the injectable health-check runner."""
    return ReadinessChecks(
        {
            "configuration": partial(check_configuration, settings),
            "mysql": partial(check_mysql, mysql),
            "redis": partial(check_redis, redis),
            "elasticsearch": partial(check_elasticsearch, elasticsearch),
            "artifacts": partial(check_artifacts, artifacts),
            "checkpoints": partial(check_checkpoints, checkpoints),
            "reranker": partial(check_reranker, reranker_initialized),
        }
    )


health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get(
    "/live", response_model=HealthResponse, response_model_exclude_none=True
)
async def live() -> HealthResponse:
    """Report that the process can service its event loop."""
    return HealthResponse(status="live", dependencies=None)


@health_router.get("/ready", response_model=HealthResponse)
async def ready(request: Request, response: Response) -> HealthResponse:
    """Report whether all required configured dependencies are available."""
    dependencies = await request.app.state.container.readiness_checks.run()
    is_ready = all(value == "available" for value in dependencies.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ready" if is_ready else "unready",
        dependencies=dependencies,
    )
