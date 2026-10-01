"""Sanitized runtime configuration summary for the same-origin console."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ValidationError

from agentic_rag.api.errors import ApiException
from agentic_rag.runtime.models import RuntimeConfigSnapshot


DependencyStatus = Literal["available", "unavailable"]


class RuntimeSummaryResponse(BaseModel):
    """Public runtime metadata that is safe to render in the operator console."""

    runtime_config_snapshot_id: str
    app_version: str
    graph_version: str
    main_model_id: str
    light_model_id: str
    deepseek_protocol: Literal["auto", "chat", "responses"]
    embedding_model: str
    index_generation: str
    memory_enabled: bool
    memory_available: bool
    dependencies: dict[str, DependencyStatus]
    evaluation_requests_enabled: bool = False


runtime_summary_router = APIRouter(prefix="/v1", tags=["runtime"])


def _runtime_snapshot(request: Request) -> RuntimeConfigSnapshot:
    """Read the immutable deployment snapshot or fail closed without details."""
    candidate = getattr(request.app.state.container, "runtime_snapshot", None)
    if isinstance(candidate, RuntimeConfigSnapshot):
        return candidate
    if isinstance(candidate, Mapping):
        try:
            return RuntimeConfigSnapshot.model_validate(candidate)
        except (TypeError, ValueError, ValidationError):
            pass
    raise ApiException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        error_code="RUNTIME_SNAPSHOT_UNAVAILABLE",
        message="The runtime configuration summary is temporarily unavailable.",
        retryable=True,
    )


@runtime_summary_router.get("/runtime/summary", response_model=RuntimeSummaryResponse)
async def runtime_summary(request: Request) -> RuntimeSummaryResponse:
    """Return allowlisted snapshot metadata and current dependency availability."""
    snapshot = _runtime_snapshot(request)
    dependencies = await request.app.state.container.readiness_checks.run()
    settings = request.app.state.container.settings
    memory_enabled = bool(getattr(settings, "mem0_enabled", False))
    memory_available = memory_enabled and dependencies.get("memory") == "available"
    return RuntimeSummaryResponse(
        runtime_config_snapshot_id=snapshot.snapshot_id,
        app_version=snapshot.app_version,
        graph_version=snapshot.graph_version,
        main_model_id=snapshot.main_model_id,
        light_model_id=snapshot.light_model_id,
        deepseek_protocol=snapshot.deepseek_protocol,
        embedding_model=snapshot.embedding_model,
        index_generation=snapshot.index_generation,
        memory_enabled=memory_enabled,
        memory_available=memory_available,
        dependencies=dependencies,
        evaluation_requests_enabled=bool(getattr(settings, "allow_evaluation_requests", False)),
    )
