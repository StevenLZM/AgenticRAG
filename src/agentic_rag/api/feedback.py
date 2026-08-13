"""Scoped user-feedback endpoint backed by the existing Agent Event log."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from agentic_rag.api.errors import ApiException
from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.repositories import AgentEvent
from agentic_rag.runtime.ids import content_id


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    rating: Literal["up", "down"]
    comment: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2_000)] | None = None


class FeedbackResponse(BaseModel):
    run_id: str
    rating: Literal["up", "down"]
    status: str = "accepted"


feedback_router = APIRouter(prefix="/v1", tags=["feedback"])


def _scope(request: Request) -> UserScope:
    settings = request.app.state.container.settings
    return UserScope(user_id=str(getattr(settings, "default_user_id", "default_user")))


def _dependency(request: Request, name: str):
    value = getattr(request.app.state.container, name, None)
    if value is None:
        raise ApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DEPENDENCY_UNAVAILABLE",
            message="The feedback service is temporarily unavailable.",
            retryable=True,
            degraded_components=(name,),
        )
    return value


@feedback_router.post("/feedback", status_code=status.HTTP_202_ACCEPTED, response_model=FeedbackResponse)
async def create_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    scope = _scope(request)
    run = await _dependency(request, "run_manager").get(payload.run_id, scope)
    if run is None:
        raise ApiException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="QUERY_RUN_NOT_FOUND",
            message="The query run was not found.",
        )
    comment = payload.comment.strip() if payload.comment else ""
    summary = f"rating={payload.rating}"
    if comment:
        summary += f" comment={comment}"
    event = AgentEvent(
        event_key=content_id("feedback", run.id, scope.user_id, payload.rating, comment)[:64],
        trace_id=run.id,
        run_id=run.id,
        user_id=scope.user_id,
        event_type="USER_FEEDBACK",
        summary=summary[:1_000],
        runtime_config_snapshot_id=run.runtime_config_snapshot_id,
    )
    await _dependency(request, "event_repository").append(event)
    return FeedbackResponse(run_id=run.id, rating=payload.rating)
