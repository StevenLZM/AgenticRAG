"""Deterministic HTTP contracts for the Phase 4 query runtime APIs.

The real MySQL/Redis/Mem0 adapters are covered by their own integration suites;
these tests keep the API boundary executable with injected ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from agentic_rag.api.errors import register_error_handlers
from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.memory.models import MemoryRecord, MemoryType
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.repositories import ActiveRunConflict, AgentEvent, QueryRun
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SCOPE = UserScope(user_id="api-user")


@pytest.mark.parametrize("route", ["chat", "fast_rag", "research", "private prompt"])
def test_sse_route_projects_only_safe_route_enum(route: str) -> None:
    from agentic_rag.api.query_runs import _sse_event
    from agentic_rag.observability.logging import sanitize_summary

    event = AgentEvent(
        id=1, event_key="route", trace_id="run-1", run_id="run-1",
        user_id="api-user", event_type="QUERY_ROUTED", node_name="query_routed",
        summary=sanitize_summary(route), runtime_config_snapshot_id="snapshot",
    )
    payload = json.loads(_sse_event(event).split("data: ", 1)[1])
    if route == "private prompt":
        assert "route" not in payload
        assert "private prompt" not in json.dumps(payload)
    else:
        assert payload.get("route") == route


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
    artifacts: object | None = None,
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
        artifacts=artifacts,
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
async def test_sse_public_event_does_not_forward_raw_summary() -> None:
    """A public event name never authorizes prompt/provider/tool summary text."""
    event = AgentEvent(
        id=1,
        event_key="public-raw-summary",
        trace_id="trace-1",
        run_id="run-1",
        user_id=SCOPE.user_id,
        event_type="TOOL_COMPLETED",
        summary="Bearer provider response with prompt and raw tool_input",
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        created_at=datetime.now(UTC),
    )
    runs = FakeRunManager(runs={"run-1": _run(status=RunStatus.COMPLETED)})
    app, _, _, _ = _app(runs, FakeEvents(events=[event]))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1/events")

    assert response.status_code == 200
    assert "Bearer " not in response.text
    assert "prompt" not in response.text
    assert "tool_input" not in response.text
    assert '"summary":"telemetry event"' in response.text


@pytest.mark.integration
async def test_sensitive_research_exception_is_absent_from_run_api_and_sse() -> None:
    secret = "Authorization: Bearer research-secret provider_response=https://private"
    event = AgentEvent(
        id=1,
        event_key="research-provider-unavailable",
        trace_id="trace-1",
        run_id="run-1",
        user_id=SCOPE.user_id,
        event_type="TOOL_COMPLETED",
        summary=secret,
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        created_at=datetime.now(UTC),
    )
    runs = FakeRunManager(runs={
        "run-1": replace(
            _run(status=RunStatus.COMPLETED),
            answer={
                "status": "cannot_answer",
                "raw": secret,
                "provider_response": secret,
            },
        )
    })
    app, _, _, _ = _app(runs, FakeEvents(events=[event]))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_response = await client.get("/v1/query-runs/run-1")
        sse_response = await client.get("/v1/query-runs/run-1/events")

    assert run_response.status_code == sse_response.status_code == 200
    assert run_response.json()["answer"]["status"] == "cannot_answer"
    for marker in ("research-secret", "provider_response", "https://private"):
        assert marker not in run_response.text
        assert marker not in sse_response.text


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
async def test_sse_exposes_only_allowlisted_degradation_attributes(tmp_path: Path) -> None:
    """A console can render reason/outcome/retryability without raw provider data."""
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    payload_ref = artifacts.put_json(
        "observability/events/degradation.json",
        {
            "attributes": {
                "attempt": 1,
                "component": "dense",
                "outcome": "degraded",
                "prompt": "never expose this prompt",
                "provider_response": "Bearer secret-provider-text",
                "reason": "lane_timeout",
                "retryable": True,
                "tool_input": "never expose this tool payload",
            }
        },
    )
    event = AgentEvent(
        id=1,
        event_key="retrieval-degraded-attributes",
        trace_id="trace-1",
        run_id="run-1",
        user_id=SCOPE.user_id,
        event_type="RETRIEVAL_DEGRADED",
        summary="degraded",
        payload_ref=payload_ref.uri,
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        created_at=datetime.now(UTC),
    )
    runs = FakeRunManager(runs={"run-1": _run(status=RunStatus.COMPLETED)})
    app, _, _, _ = _app(runs, FakeEvents(events=[event]), artifacts=artifacts)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1/events")

    assert response.status_code == 200
    payload = next(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    )
    assert payload["attributes"] == {
        "attempt": 1,
        "component": "dense",
        "outcome": "degraded",
        "reason": "lane_timeout",
        "retryable": True,
    }
    assert "never expose" not in response.text
    assert "Bearer " not in response.text


@pytest.mark.integration
async def test_sse_exposes_safe_llm_diagnostics(tmp_path: Path) -> None:
    """LLM retry metadata reaches operators without exposing provider payloads."""
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    payload_ref = artifacts.put_json(
        "observability/events/model-retry.json",
        {
            "attributes": {
                "attempt": 1,
                "client_timeout_seconds": 30.0,
                "component": "llm",
                "error_class": "APITimeoutError",
                "http_status": 503,
                "operation": "graph.node.fast_rag.llm",
                "outcome": "degraded",
                "protocol": "auto",
                "provider_request_id": "req_123",
                "reason": "provider_outage",
                "requested_model": "deepseek-v4-flash",
                "retryable": True,
                "error_message": "Bearer provider-secret",
            }
        },
    )
    event = AgentEvent(
        id=1,
        event_key="model-retry-diagnostics",
        trace_id="trace-1",
        run_id="run-1",
        user_id=SCOPE.user_id,
        event_type="MODEL_RETRY",
        summary="degraded",
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        node_name="graph.node.fast_rag.llm",
        payload_ref=payload_ref.uri,
        created_at=datetime.now(UTC),
    )
    runs = FakeRunManager(runs={"run-1": _run(status=RunStatus.COMPLETED)})
    app, _, _, _ = _app(runs, FakeEvents(events=[event]), artifacts=artifacts)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1/events")

    assert response.status_code == 200
    payload = next(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    )
    assert payload["attributes"] == {
        "attempt": 1,
        "client_timeout_seconds": 30.0,
        "component": "llm",
        "error_class": "APITimeoutError",
        "http_status": 503,
        "operation": "graph.node.fast_rag.llm",
        "outcome": "degraded",
        "protocol": "auto",
        "provider_request_id": "req_123",
        "reason": "provider_outage",
        "requested_model": "deepseek-v4-flash",
        "retryable": True,
    }
    assert "provider-secret" not in response.text


@pytest.mark.integration
@pytest.mark.parametrize("error_type", (OSError, TimeoutError, ConnectionError))
async def test_memory_list_provider_errors_fail_closed(
    error_type: type[OSError],
) -> None:
    class UnavailableMemory(FakeMemory):
        async def list(self, scope: UserScope) -> list[MemoryRecord]:
            del scope
            raise error_type("provider unavailable")

    app, _, _, _ = _app(memory=UnavailableMemory())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/memories")

    assert response.status_code == 503
    assert response.json()["error_code"] == "MEMORY_UNAVAILABLE"


@pytest.mark.integration
@pytest.mark.parametrize("error_type", (OSError, TimeoutError, ConnectionError))
async def test_memory_delete_provider_errors_fail_closed(
    error_type: type[OSError],
) -> None:
    class UnavailableMemory(FakeMemory):
        async def delete(self, scope: UserScope, memory_id: str) -> None:
            del scope, memory_id
            raise error_type("provider unavailable")

    app, _, _, _ = _app(memory=UnavailableMemory())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/v1/memories/memory-1")

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
    secret = "Authorization: Bearer api-answer-secret"
    runs = FakeRunManager(
        runs={
            "run-1": replace(
                _run(status=RunStatus.COMPLETED),
                answer={
                    "audited": True,
                    "segments": [{
                        "kind": "content",
                        "text": "notice applies",
                        "evidence_ids": ["e1"],
                        "tool_input": secret,
                    }],
                    "evidence_parent_ids": ["parent-1"],
                    "route": "fast_rag",
                    "runtime_config_snapshot_id": SNAPSHOT.snapshot_id,
                    "client_provenance": "real_query_api",
                    "prompt": secret,
                    "provider_response": secret,
                    "raw": secret,
                    "unknown": secret,
                },
            )
        }
    )
    app, _, _, _ = _app(runs)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1")

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer == {
        "status": None,
        "audited": True,
        "segments": [
            {"kind": "content", "text": "notice applies", "evidence_ids": ["e1"]}
        ],
        "evidence_parent_ids": ["parent-1"],
        "route": "fast_rag",
        "runtime_config_snapshot_id": SNAPSHOT.snapshot_id,
        "client_provenance": "real_query_api",
        "citation_coverage": None,
    }
    assert secret not in response.text


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
