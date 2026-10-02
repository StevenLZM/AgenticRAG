"""Scoped durable query-run and reconnectable event HTTP endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agentic_rag.api.errors import ApiException
from agentic_rag.api.query_context import request_scope as _scope, runtime_snapshot as _snapshot, require_dependency as _dependency
from agentic_rag.domain.chat_sessions import utc_datetime
from agentic_rag.domain.models import RunStatus
from agentic_rag.observability.logging import sanitize_attributes, sanitize_summary
from agentic_rag.persistence.repositories import ActiveRunConflict, AgentEvent, QueryRun
from agentic_rag.query.phases import PHASES, QueryPhase
from agentic_rag.query.public_answer import PublicAnswer, project_public_answer
from agentic_rag.runtime.ids import new_id
from agentic_rag.runtime.models import EvaluationMetadata


QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32_000)]
ThreadText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class QueryRequest(BaseModel):
    """A bounded client query; user scope is injected from server settings."""

    model_config = ConfigDict(extra="forbid")

    query: QueryText
    thread_id: ThreadText | None = None
    wait_seconds: int = Field(default=30, ge=0, le=30)
    evaluation: EvaluationMetadata | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "QueryRequest":
        self.query = self.query.strip()
        if self.thread_id is not None:
            self.thread_id = self.thread_id.strip()
        return self


class QueryRunResponse(BaseModel):
    """Stable public Run representation; private leases are never serialized."""

    run_id: str
    status: RunStatus
    thread_id: str
    runtime_config_snapshot_id: str
    question: str = ""
    created_at: str | None = None
    result_ref: str | None = None
    error_code: str | None = None
    answer: PublicAnswer | None = None
    phase: QueryPhase | None = None

    @classmethod
    def from_run(cls, run: QueryRun, *, phase: QueryPhase | None = None) -> "QueryRunResponse":
        # Some deployment-owned repositories attach an already materialized
        # audited answer.  The SQL Run port deliberately keeps this optional;
        # never expose leases, snapshots or worker internals here.
        answer = project_public_answer(
            run.answer if run.status == RunStatus.COMPLETED else None,
            runtime_config_snapshot_id=run.runtime_config_snapshot_id,
        )
        return cls(
            run_id=run.id,
            status=run.status,
            thread_id=run.thread_id,
            runtime_config_snapshot_id=run.runtime_config_snapshot_id,
            question=run.question,
            created_at=(
                utc_datetime(run.created_at).isoformat(timespec="microseconds").replace("+00:00", "Z")
                if run.created_at else None
            ),
            result_ref=run.result_ref,
            error_code=run.error_code,
            answer=answer,
            phase=(phase or "processing") if run.status == RunStatus.RUNNING else None,
        )


class QueryRunsResponse(BaseModel):
    run_id: str
    status: RunStatus
    thread_id: str


query_runs_router = APIRouter(prefix="/v1", tags=["query"])
TERMINAL_STATUSES = {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}
_PUBLIC_EVENT_TYPES = {
    "QUERY_PHASE_CHANGED",
    "RUN_STARTED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "MEMORY_LOADED",
    "QUERY_ROUTED",
    "FAST_RAG_COMPLETED",
    "RESEARCH_LOOP_COMPLETED",
    "RETRIEVAL_COMPLETED",
    "EVIDENCE_GRADED",
    "FAITHFULNESS_AUDITED",
    "CITATION_VALIDATED",
    "ANSWER_GENERATED",
    "ANSWER_FINALIZED",
    "TODO_UPDATED",
    "TOOL_STARTED",
    "TOOL_COMPLETED",
    "COMPONENT_DEGRADED",
    "COMPONENT_REFUSED",
    "RETRIEVAL_DEGRADED",
    "CIRCUIT_OPEN",
    "OUTBOX_RETRY",
    "WORKER_DLQ",
    "QUERY_REFUSED",
    "QUERY_CANCELLED",
    "QUERY_TIMEOUT",
    "LEASE_LOST",
    "MODEL_RETRY",
    "MODEL_RETRY_EXHAUSTED",
    "MODEL_REPAIR_EXHAUSTED",
    "AUDIT_REFUSED",
    "RUN_CANCEL_REQUESTED",
    "RUN_CANCELLED",
    "USER_FEEDBACK",
}
_PUBLIC_DEGRADATION_EVENT_TYPES = {
    "MEMORY_LOADED", "QUERY_ROUTED", "FAST_RAG_COMPLETED", "RESEARCH_LOOP_COMPLETED",
    "EVIDENCE_GRADED", "ANSWER_GENERATED", "ANSWER_FINALIZED",
    "COMPONENT_DEGRADED",
    "COMPONENT_REFUSED",
    "RETRIEVAL_DEGRADED",
    "CIRCUIT_OPEN",
    "OUTBOX_RETRY",
    "WORKER_DLQ",
    "QUERY_TIMEOUT",
    "LEASE_LOST",
    "MODEL_RETRY",
    "MODEL_RETRY_EXHAUSTED",
    "MODEL_REPAIR_EXHAUSTED",
    "AUDIT_REFUSED",
}
_PUBLIC_DEGRADATION_ATTRIBUTE_FIELDS = frozenset(
    {
        "initial_route", "route", "executed_path", "gap_type", "response_mode", "termination_reason",
        "attempt",
        "component",
        "reason",
        "outcome",
        "retryable",
        "operation",
        "requested_model",
        "protocol",
        "client_timeout_seconds",
        "error_class",
        "http_status",
        "provider_request_id",
    }
)


def _thread_id(request: QueryRequest) -> str:
    return request.thread_id or new_id()


def _active_conflict_location(error: ActiveRunConflict) -> str | None:
    existing = getattr(error, "existing_run_id", None)
    return f"/v1/query-runs/{existing}" if isinstance(existing, str) and existing else None


async def _create_run(request: Request, payload: QueryRequest) -> QueryRun:
    manager = _dependency(request, "run_manager")
    snapshot = _snapshot(request)
    if payload.evaluation is not None:
        settings = request.app.state.container.settings
        if not getattr(settings, "allow_evaluation_requests", False):
            raise ApiException(status_code=403, error_code="EVALUATION_DISABLED",
                               message="Evaluation requests are disabled.", retryable=False)
        # An experiment cannot join an ordinary conversation, nor select a user.
        if payload.thread_id is None or not payload.thread_id.startswith("eval-v2-"):
            raise ApiException(status_code=422, error_code="EVALUATION_SESSION_REQUIRED",
                               message="An isolated evaluation thread is required.", retryable=False)
        snapshot = snapshot.model_copy(update={"evaluation": payload.evaluation})
    try:
        return await manager.create(
            _scope(request), _thread_id(payload), payload.query, snapshot
        )
    except ActiveRunConflict:
        # RunManager resolves a race-safe server-owned existing ID where the
        # SQL adapter supports it.  Lightweight ports may attach the same
        # field themselves; no client-provided ID is ever trusted.
        raise


@query_runs_router.post(
    "/query-runs", status_code=status.HTTP_202_ACCEPTED, response_model=QueryRunsResponse
)
async def create_query_run(request: Request, payload: QueryRequest, response: Response) -> QueryRunsResponse:
    run = await _create_run(request, payload)
    response.headers["Location"] = f"/v1/query-runs/{run.id}"
    return QueryRunsResponse(run_id=run.id, status=run.status, thread_id=run.thread_id)


async def _owned_run(request: Request, run_id: str) -> QueryRun:
    manager = _dependency(request, "run_manager")
    run = await manager.get(run_id, _scope(request))
    if run is None:
        raise ApiException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="QUERY_RUN_NOT_FOUND",
            message="The query run was not found.",
        )
    return run


@query_runs_router.get("/query-runs/{run_id}", response_model=QueryRunResponse)
async def get_query_run(request: Request, run_id: str, response: Response) -> QueryRunResponse:
    response.headers["Cache-Control"] = "no-store"
    run = await _owned_run(request, run_id)
    phases = {}
    reader = getattr(request.app.state.container, "query_phase_reader", None)
    if reader is not None and run.status == RunStatus.RUNNING:
        try:
            phases = await reader.latest(_scope(request), [run_id])
        except Exception:
            pass
    return QueryRunResponse.from_run(run, phase=phases.get(run_id))


@query_runs_router.post("/query-runs/{run_id}/cancel", response_model=QueryRunResponse)
async def cancel_query_run(request: Request, run_id: str) -> QueryRunResponse:
    await _owned_run(request, run_id)
    manager = _dependency(request, "run_manager")
    try:
        await manager.request_cancel(_scope(request), run_id)
    except ActiveRunConflict:
        pass
    # Completion may win the cancellation race. Read its durable payload as well
    # as its status, and recheck access if the session was deleted meanwhile.
    return QueryRunResponse.from_run(await _owned_run(request, run_id))


@query_runs_router.get("/query-runs/{run_id}/events")
async def query_run_events(
    request: Request,
    run_id: str,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    await _owned_run(request, run_id)
    start = _parse_cursor(last_event_id)

    async def stream() -> AsyncIterator[str]:
        cursor = start
        while True:
            try:
                current = await _owned_run(request, run_id)
            except ApiException as error:
                if error.status_code == 404:
                    return
                raise
            events = await _dependency(request, "event_repository").list_after(
                run_id, _scope(request), cursor, 100
            )
            if events:
                for event in events:
                    # A session can be deleted while a terminal backlog streams.
                    try:
                        await _owned_run(request, run_id)
                    except ApiException as error:
                        if error.status_code == 404:
                            return
                        raise
                    if event.id is None or event.id <= cursor:
                        continue
                    cursor = event.id
                    container = request.app.state.container
                    yield _sse_event(event, artifacts=getattr(container, "artifacts", None))
                continue
            if current.status in TERMINAL_STATUSES:
                return
            # Comments are intentionally not JSON events and contain no state.
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _parse_cursor(value: str | None) -> int:
    try:
        cursor = int(value or "0")
    except (TypeError, ValueError):
        return 0
    return max(cursor, 0)


def _sse_event(event: AgentEvent, *, artifacts: object | None = None) -> str:
    is_public = event.event_type in _PUBLIC_EVENT_TYPES
    event_type = event.event_type if is_public else "PROGRESS"
    payload: dict[str, object] = {
        "run_id": event.run_id,
        "event_type": event_type,
        # Unknown event types may contain internal/tool payload summaries. Do
        # not forward their source text merely because the event row is scoped.
        # A public event *type* only permits its stable notification shape.
        # Event summaries can still originate from model, provider or tool
        # boundaries, so project them through the telemetry sanitizer instead
        # of treating the durable row as a client-safe payload.
        "summary": sanitize_summary(event.summary) if is_public else "progress update",
        "created_at": (event.created_at or datetime.now(UTC)).isoformat(),
    }
    if event_type == "QUERY_PHASE_CHANGED" and event.summary in PHASES:
        payload["phase"] = event.summary
    attributes = _safe_degradation_attributes(event, artifacts)
    if event_type == "QUERY_ROUTED" and event.summary in {"chat", "fast_rag", "research"}:
        payload["route"] = event.summary
    if attributes.get("route") in {"chat", "fast_rag", "research"}:
        payload["route"] = attributes["route"]
    if attributes:
        payload["attributes"] = attributes
    return f"id: {event.id}\nevent: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _safe_degradation_attributes(
    event: AgentEvent, artifacts: object | None
) -> dict[str, object]:
    """Read only the event artifact's existing allowlisted degradation metadata."""
    if (
        event.event_type not in _PUBLIC_DEGRADATION_EVENT_TYPES
        or not isinstance(event.payload_ref, str)
        or not event.payload_ref
        or artifacts is None
    ):
        return {}
    describe = getattr(artifacts, "describe", None)
    read_json = getattr(artifacts, "read_json", None)
    if not callable(describe) or not callable(read_json):
        return {}
    try:
        payload = read_json(describe(event.payload_ref))
    except Exception:
        return {}
    raw_attributes = payload.get("attributes") if isinstance(payload, Mapping) else None
    sanitized = sanitize_attributes(raw_attributes if isinstance(raw_attributes, Mapping) else None)
    return {
        key: sanitized[key]
        for key in _PUBLIC_DEGRADATION_ATTRIBUTE_FIELDS
        if key in sanitized
    }


@query_runs_router.post("/query")
async def sync_query(request: Request, payload: QueryRequest, response: Response) -> QueryRunResponse:
    run = await _create_run(request, payload)
    manager = _dependency(request, "run_manager")
    deadline = asyncio.get_running_loop().time() + payload.wait_seconds
    current = run
    while payload.wait_seconds > 0 and asyncio.get_running_loop().time() < deadline:
        if current.status in TERMINAL_STATUSES:
            break
        await asyncio.sleep(min(0.1, max(0.0, deadline - asyncio.get_running_loop().time())))
        refreshed = await manager.get(run.id, _scope(request))
        if refreshed is None:
            break
        current = refreshed
    response.headers["Location"] = f"/v1/query-runs/{run.id}"
    # Worker completion is only recorded after the QueryGraph's mandatory
    # audits pass; no second Run is created when the wait expires.
    if current.status is RunStatus.COMPLETED:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_202_ACCEPTED
    return QueryRunResponse.from_run(current)
