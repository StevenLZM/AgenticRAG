"""Deterministic HTTP contracts for the Phase 4 query runtime APIs.

The real MySQL/Redis/Mem0 adapters are covered by their own integration suites;
these tests keep the API boundary executable with injected ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from agentic_rag.api.errors import register_error_handlers
from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.memory.models import MemoryRecord, MemoryType
from agentic_rag.persistence.repositories import ActiveRunConflict, AgentEvent, QueryRun
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SCOPE = UserScope(user_id="api-user")
SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main",
    light_model_id="light",
    embedding_model="embedding",
    embedding_dimensions=1024,
    reranker_version="reranker",
    retrieval_config_version="retrieval",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)


def _run(
    run_id: str = "run-1",
    *,
    status: RunStatus = RunStatus.QUEUED,
    thread_id: str = "thread-1",
    question: str = "What notice applies?",
) -> QueryRun:
    return QueryRun(
        id=run_id,
        user_id=SCOPE.user_id,
        thread_id=thread_id,
        checkpoint_thread_id=f"query:{SCOPE.user_id}:{thread_id}",
        status=status,
        active_slot=1 if status in {RunStatus.QUEUED, RunStatus.RUNNING} else None,
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        runtime_config_snapshot=SNAPSHOT.model_dump(mode="json"),
        question=question,
    )


@dataclass
class FakeRunManager:
    runs: dict[str, QueryRun] = field(default_factory=lambda: {"run-1": _run()})
    creates: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    conflict: bool = False

    async def create(
        self,
        scope: UserScope,
        thread_id: str,
        question: str,
        snapshot: RuntimeConfigSnapshot,
    ) -> QueryRun:
        if self.conflict:
            error = ActiveRunConflict("active run")
            error.existing_run_id = "run-1"  # type: ignore[attr-defined]
            raise error
        run = _run(
            run_id=f"run-{len(self.runs) + 1}",
            thread_id=thread_id,
            question=question,
        )
        self.runs[run.id] = run
        self.creates.append((thread_id, question))
        return run

    async def get(self, run_id: str, scope: UserScope) -> QueryRun | None:
        run = self.runs.get(run_id)
        return run if run is not None and run.user_id == scope.user_id else None

    async def request_cancel(self, scope: UserScope, run_id: str) -> RunStatus:
        run = await self.get(run_id, scope)
        if run is None:
            return RunStatus.FAILED
        updated = replace(run, status=RunStatus.CANCEL_REQUESTED)
        self.runs[run_id] = updated
        self.cancelled.append(run_id)
        return updated.status


@dataclass
class FakeEvents:
    events: list[AgentEvent] = field(default_factory=list)

    async def append(self, event: AgentEvent) -> int:
        event_id = len(self.events) + 1
        self.events.append(replace(event, id=event_id))
        return event_id

    async def list_after(
        self, run_id: str, scope: UserScope, after_id: int, limit: int
    ) -> list[AgentEvent]:
        return [
            event
            for event in self.events
            if event.run_id == run_id
            and event.user_id == scope.user_id
            and (event.id or 0) > after_id
        ][:limit]


@dataclass
class FakeMemory:
    records: list[MemoryRecord] = field(
        default_factory=lambda: [
            MemoryRecord(
                id="memory-1",
                user_id=SCOPE.user_id,
                text="prefers concise answers",
                memory_type=MemoryType.SEMANTIC,
            )
        ]
    )
    deleted: list[str] = field(default_factory=list)

    async def list(self, scope: UserScope) -> list[MemoryRecord]:
        return [record for record in self.records if record.user_id == scope.user_id]

    async def delete(self, scope: UserScope, memory_id: str) -> None:
        self.deleted.append(f"{scope.user_id}:{memory_id}")


def _app(
    runs: FakeRunManager | None = None,
    events: FakeEvents | None = None,
    memory: FakeMemory | None = None,
) -> tuple[FastAPI, FakeRunManager, FakeEvents, FakeMemory]:
    run_manager = runs or FakeRunManager()
    event_repository = events or FakeEvents()
    memory_service = memory or FakeMemory()
    app = FastAPI()
    app.state.container = SimpleNamespace(
        settings=SimpleNamespace(default_user_id=SCOPE.user_id),
        run_manager=run_manager,
        event_repository=event_repository,
        memory_service=memory_service,
    )
    from agentic_rag.api.feedback import feedback_router
    from agentic_rag.api.memories import memories_router
    from agentic_rag.api.query_runs import query_runs_router

    app.include_router(query_runs_router)
    app.include_router(memories_router)
    app.include_router(feedback_router)
    register_error_handlers(app)
    return app, run_manager, event_repository, memory_service


@pytest.mark.integration
async def test_create_run_returns_202_and_active_conflict_location() -> None:
    runs = FakeRunManager()
    app, _, _, _ = _app(runs)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/v1/query-runs", json={"query": "  notice? ", "thread_id": "thread-new"}
        )
        runs.conflict = True
        conflict = await client.post(
            "/v1/query-runs", json={"query": "second", "thread_id": "thread-1"}
        )

    assert created.status_code == 202
    assert created.json()["run_id"] == "run-2"
    assert runs.creates == [("thread-new", "notice?")]
    assert conflict.status_code == 409
    assert conflict.headers["location"] == "/v1/query-runs/run-1"
    assert conflict.json()["error_code"] == "ACTIVE_RUN_EXISTS"


@pytest.mark.integration
async def test_run_get_cancel_and_feedback_are_scoped() -> None:
    app, runs, events, _ = _app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        found = await client.get("/v1/query-runs/run-1")
        cancelled = await client.post("/v1/query-runs/run-1/cancel")
        feedback = await client.post(
            "/v1/feedback",
            json={"run_id": "run-1", "rating": "up", "comment": "useful"},
        )
        missing = await client.get("/v1/query-runs/not-owned")

    assert found.status_code == 200
    assert found.json()["status"] == "queued"
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancel_requested"
    assert feedback.status_code == 202
    assert events.events[-1].event_type == "USER_FEEDBACK"
    assert "useful" in events.events[-1].summary
    assert missing.status_code == 404


@pytest.mark.integration
async def test_sse_reconnect_uses_event_cursor_and_redacts_payload() -> None:
    events = FakeEvents(
        events=[
            AgentEvent(
                id=1,
                event_key="event-1",
                trace_id="trace-1",
                run_id="run-1",
                user_id=SCOPE.user_id,
                event_type="RETRIEVAL_COMPLETED",
                summary="retrieval complete",
                payload_ref="artifact://secret/raw-tool-payload",
                runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
                created_at=datetime.now(UTC),
            ),
            AgentEvent(
                id=2,
                event_key="event-2",
                trace_id="trace-1",
                run_id="run-1",
                user_id=SCOPE.user_id,
                event_type="ANSWER_FINALIZED",
                summary="completed",
                runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
                created_at=datetime.now(UTC),
            ),
        ]
    )
    runs = FakeRunManager(runs={"run-1": _run(status=RunStatus.COMPLETED)})
    app, _, _, _ = _app(runs, events)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("GET", "/v1/query-runs/run-1/events") as response:
            body = "\n".join([line async for line in response.aiter_lines()])
        async with client.stream(
            "GET",
            "/v1/query-runs/run-1/events",
            headers={"Last-Event-ID": "1"},
        ) as resumed_response:
            resumed = "\n".join([line async for line in resumed_response.aiter_lines()])

    assert response.status_code == 200
    assert "id: 1" in body and "id: 2" in body
    assert "artifact://secret/raw-tool-payload" not in body
    assert "id: 1" not in resumed
    assert "id: 2" in resumed


@pytest.mark.integration
async def test_sse_unknown_event_does_not_forward_raw_summary() -> None:
    event = AgentEvent(
        id=1,
        event_key="unknown-event",
        trace_id="trace-1",
        run_id="run-1",
        user_id=SCOPE.user_id,
        event_type="INTERNAL_TOOL_PAYLOAD",
        summary='{"secret_tool_input":"do-not-stream"}',
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        created_at=datetime.now(UTC),
    )
    runs = FakeRunManager(runs={"run-1": _run(status=RunStatus.COMPLETED)})
    app, _, _, _ = _app(runs, FakeEvents(events=[event]))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1/events")

    assert response.status_code == 200
    assert "do-not-stream" not in response.text
    assert '"event_type":"PROGRESS"' in response.text


@pytest.mark.integration
async def test_sse_exposes_safe_degradation_event_notice() -> None:
    event = AgentEvent(
        id=1,
        event_key="retrieval-degraded",
        trace_id="trace-1",
        run_id="run-1",
        user_id=SCOPE.user_id,
        event_type="RETRIEVAL_DEGRADED",
        summary="degraded",
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        created_at=datetime.now(UTC),
    )
    runs = FakeRunManager(runs={"run-1": _run(status=RunStatus.COMPLETED)})
    app, _, _, _ = _app(runs, FakeEvents(events=[event]))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1/events")

    assert response.status_code == 200
    assert '"event_type":"RETRIEVAL_DEGRADED"' in response.text
    assert '"summary":"degraded"' in response.text


@pytest.mark.integration
async def test_unconfigured_memory_list_fails_closed() -> None:
    class UnavailableMemory(FakeMemory):
        async def list(self, scope: UserScope) -> list[MemoryRecord]:
            del scope
            raise OSError("provider unavailable")

    app, _, _, _ = _app(memory=UnavailableMemory())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/memories")

    assert response.status_code == 503
    assert response.json()["error_code"] == "MEMORY_UNAVAILABLE"


@pytest.mark.integration
async def test_sync_query_timeout_returns_202_without_second_run() -> None:
    runs = FakeRunManager()
    app, manager, _, _ = _app(runs)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/query", json={"query": "complex research", "wait_seconds": 0}
        )

    assert response.status_code == 202
    assert response.json()["run_id"] == "run-2"
    assert len(manager.creates) == 1


@pytest.mark.integration
async def test_completed_run_response_exposes_only_audited_answer_projection() -> None:
    runs = FakeRunManager(
        runs={
            "run-1": replace(
                _run(status=RunStatus.COMPLETED),
                answer={"segments": [{"text": "notice applies", "evidence_ids": ["e1"]}]},
            )
        }
    )
    app, _, _, _ = _app(runs)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1")

    assert response.status_code == 200
    assert response.json()["answer"]["segments"][0]["evidence_ids"] == ["e1"]


@pytest.mark.integration
async def test_memory_list_and_delete_are_scoped_and_tombstone_backed() -> None:
    memory = FakeMemory()
    app, _, _, memory = _app(memory=memory)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/v1/memories")
        deleted = await client.delete("/v1/memories/memory-1")

    assert listed.status_code == 200
    assert listed.json()["memories"][0]["id"] == "memory-1"
    assert deleted.status_code == 204
    assert memory.deleted == ["api-user:memory-1"]
