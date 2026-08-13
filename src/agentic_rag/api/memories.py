"""User-scoped long-term memory endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from agentic_rag.api.errors import ApiException
from agentic_rag.domain.models import UserScope
from agentic_rag.memory.models import MemoryRecord


class MemoriesResponse(BaseModel):
    memories: list[MemoryRecord]


memories_router = APIRouter(prefix="/v1", tags=["memories"])


def _scope(request: Request) -> UserScope:
    settings = request.app.state.container.settings
    return UserScope(user_id=str(getattr(settings, "default_user_id", "default_user")))


def _memory_service(request: Request):
    service = getattr(request.app.state.container, "memory_service", None)
    if service is None:
        raise ApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DEPENDENCY_UNAVAILABLE",
            message="The memory service is temporarily unavailable.",
            retryable=True,
            degraded_components=("memory",),
        )
    return service


@memories_router.get("/memories", response_model=MemoriesResponse)
async def list_memories(request: Request) -> MemoriesResponse:
    try:
        records = await _memory_service(request).list(_scope(request))
    except (OSError, TimeoutError, ConnectionError) as error:
        raise ApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MEMORY_UNAVAILABLE",
            message="The memory service is temporarily unavailable.",
            retryable=True,
            degraded_components=("memory",),
        ) from error
    return MemoriesResponse(memories=records)


@memories_router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(request: Request, memory_id: str) -> Response:
    if not memory_id.strip():
        raise ApiException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="INVALID_MEMORY_ID",
            message="The memory id is invalid.",
        )
    try:
        await _memory_service(request).delete(_scope(request), memory_id.strip())
    except (OSError, TimeoutError, ConnectionError) as error:
        # A delete cannot claim tombstone-backed completion while the provider
        # is unavailable.  The client may safely retry the same idempotent call.
        raise ApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MEMORY_UNAVAILABLE",
            message="The memory service is temporarily unavailable.",
            retryable=True,
            degraded_components=("memory",),
        ) from error
    # Deletion is deliberately idempotent and does not reveal whether an ID
    # belonged to another user.  MemoryService has already written the scoped
    # tombstone before attempting provider deletion.
    return Response(status_code=status.HTTP_204_NO_CONTENT)
