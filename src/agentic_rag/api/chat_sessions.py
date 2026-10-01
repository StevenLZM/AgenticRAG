"""Scoped persistent chat endpoints; all public data is explicitly projected."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, ConfigDict, StringConstraints

from agentic_rag.api.errors import ApiException
from agentic_rag.api.query_context import (
    request_scope,
    require_dependency,
    runtime_snapshot,
)
from agentic_rag.domain.chat_sessions import (
    ChatSessionSummary,
    SessionNotFound,
    utc_datetime,
)
from agentic_rag.domain.models import RunStatus
from agentic_rag.persistence.repositories import QueryRun
from agentic_rag.query.answer_sources import cited_ids
from agentic_rag.query.phases import PHASES, QueryPhase
from agentic_rag.query.public_answer import PublicAnswer, project_public_answer
from agentic_rag.runtime.chat_sources import SourceView

chat_sessions_router = APIRouter(prefix="/v1/chat-sessions", tags=["chat"])
Limit = Annotated[int, Query(ge=1, le=100)]
Cursor = Annotated[str | None, Query(max_length=2048)]
Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32000)
]
Title = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    creation_request_id: UUID


class RenameSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title


class SubmitTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Question
    client_request_id: UUID


def public_time(value: datetime | None) -> str | None:
    return (
        utc_datetime(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
        if value
        else None
    )


class SessionResponse(BaseModel):
    session_id: str
    title: str
    title_source: Literal["default", "first_question", "manual"]
    created_at: str
    updated_at: str
    last_activity_at: str
    active_run_id: str | None
    active_run_status: RunStatus | None
    phase: QueryPhase | None

    @classmethod
    def project(cls, summary: ChatSessionSummary, phases: dict):
        s = summary.session
        return cls(
            session_id=s.id,
            title=s.title,
            title_source=s.title_source,
            created_at=public_time(s.created_at),
            updated_at=public_time(s.updated_at),
            last_activity_at=public_time(s.last_activity_at),
            active_run_id=summary.active_run_id,
            active_run_status=summary.active_run_status,
            phase=phases.get(summary.active_run_id, "processing")
            if summary.active_run_status == RunStatus.RUNNING
            else None,
        )


class TurnResponse(BaseModel):
    run_id: str
    client_request_id: str | None
    created_at: str
    finished_at: str | None
    question: str
    status: RunStatus
    answer: PublicAnswer | None
    phase: QueryPhase | None
    source_status: Literal["available", "unavailable", "none"]
    terminal_code: str | None

    @classmethod
    def project(cls, run: QueryRun, phases: dict):
        answer = (
            project_public_answer(
                run.answer, runtime_config_snapshot_id=run.runtime_config_snapshot_id
            )
            if run.status == RunStatus.COMPLETED
            else None
        )
        code = None
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            code = answer.status if answer and answer.status else run.status.value
        return cls(
            run_id=run.id,
            client_request_id=run.client_request_id,
            created_at=public_time(run.created_at),
            finished_at=public_time(run.finished_at),
            question=run.question,
            status=run.status,
            answer=answer,
            phase=phases.get(run.id, "processing")
            if run.status == RunStatus.RUNNING
            else None,
            source_status=("available" if run.answer_sources else "unavailable")
            if cited_ids(answer)
            else "none",
            terminal_code=code,
        )


class SessionsPage(BaseModel):
    items: tuple[SessionResponse, ...]
    next_cursor: str | None


class TurnsPage(BaseModel):
    items: tuple[TurnResponse, ...]
    next_cursor: str | None


def service(request):
    return require_dependency(request, "chat_session_service")


async def phases_for(request: Request, run_ids: list[str]) -> dict:
    reader = getattr(request.app.state.container, "query_phase_reader", None)
    if reader is None or not run_ids:
        return {}
    try:
        return {
            key: value
            for key, value in (
                await reader.latest(request_scope(request), run_ids)
            ).items()
            if value in PHASES
        }
    except Exception:
        return {}


def no_store(response):
    response.headers["Cache-Control"] = "no-store"


@chat_sessions_router.post("", response_model=SessionResponse, status_code=201)
async def create_session(request: Request, payload: CreateSession, response: Response):
    scope = request_scope(request)
    session, created = await service(request).create(
        scope, str(payload.creation_request_id)
    )
    response.status_code = 201 if created else 200
    response.headers["Location"] = f"/v1/chat-sessions/{session.id}"
    no_store(response)
    summary = await service(request).get(scope, session.id)
    return SessionResponse.project(
        summary,
        await phases_for(
            request, [summary.active_run_id] if summary.active_run_id else []
        ),
    )


@chat_sessions_router.get("", response_model=SessionsPage)
async def list_sessions(
    request: Request, response: Response, cursor: Cursor = None, limit: Limit = 30
):
    try:
        page = await service(request).list(
            request_scope(request), cursor=cursor, limit=limit
        )
    except ValueError as error:
        raise ApiException(
            status_code=422,
            error_code="VALIDATION_ERROR",
            message="Invalid pagination cursor.",
        ) from error
    phases = await phases_for(
        request, [item.active_run_id for item in page.items if item.active_run_id]
    )
    no_store(response)
    return SessionsPage(
        items=tuple(SessionResponse.project(item, phases) for item in page.items),
        next_cursor=page.next_cursor,
    )


@chat_sessions_router.get("/{session_id}", response_model=SessionResponse)
async def get_session(request: Request, session_id: UUID, response: Response):
    item = await service(request).get(request_scope(request), str(session_id))
    no_store(response)
    return SessionResponse.project(
        item,
        await phases_for(request, [item.active_run_id] if item.active_run_id else []),
    )


@chat_sessions_router.patch("/{session_id}", response_model=SessionResponse)
async def rename_session(
    request: Request, session_id: UUID, payload: RenameSession, response: Response
):
    await service(request).rename(
        request_scope(request), str(session_id), payload.title
    )
    return await get_session(request, session_id, response)


@chat_sessions_router.delete("/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: UUID):
    await service(request).delete(request_scope(request), str(session_id))
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


@chat_sessions_router.get("/{session_id}/turns", response_model=TurnsPage)
async def list_turns(
    request: Request,
    session_id: UUID,
    response: Response,
    cursor: Cursor = None,
    limit: Limit = 30,
):
    try:
        page = await service(request).turns(
            request_scope(request), str(session_id), cursor=cursor, limit=limit
        )
    except ValueError as error:
        raise ApiException(
            status_code=422,
            error_code="VALIDATION_ERROR",
            message="Invalid pagination cursor.",
        ) from error
    phases = await phases_for(
        request, [run.id for run in page.items if run.status == RunStatus.RUNNING]
    )
    no_store(response)
    return TurnsPage(
        items=tuple(TurnResponse.project(run, phases) for run in page.items),
        next_cursor=page.next_cursor,
    )


@chat_sessions_router.post(
    "/{session_id}/turns", response_model=TurnResponse, status_code=202
)
async def submit_turn(
    request: Request, session_id: UUID, payload: SubmitTurn, response: Response
):
    result = await service(request).submit(
        request_scope(request),
        str(session_id),
        payload.query,
        runtime_snapshot(request),
        client_request_id=str(payload.client_request_id),
    )
    response.status_code = 202 if result.created else 200
    response.headers["Location"] = f"/v1/query-runs/{result.run.id}"
    no_store(response)
    return TurnResponse.project(result.run, await phases_for(request, [result.run.id]))


@chat_sessions_router.get(
    "/{session_id}/submissions/{client_request_id}", response_model=TurnResponse
)
async def find_submission(
    request: Request, session_id: UUID, client_request_id: UUID, response: Response
):
    run = await service(request).find_submission(
        request_scope(request), str(session_id), str(client_request_id)
    )
    if run is None:
        raise SessionNotFound(str(session_id))
    no_store(response)
    return TurnResponse.project(run, await phases_for(request, [run.id]))


@chat_sessions_router.get(
    "/{session_id}/turns/{run_id}/sources", response_model=SourceView
)
async def get_sources(
    request: Request, session_id: UUID, run_id: UUID, response: Response
):
    view = await require_dependency(request, "chat_source_service").get(
        request_scope(request), str(session_id), str(run_id)
    )
    no_store(response)
    return view
