"""User phases describe real execution boundaries and never control execution."""

import asyncio
import importlib.util
from datetime import UTC, datetime

import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import AgentEventEmitter
from agentic_rag.persistence.repositories import agent_events, AgentEvent
from tests.unit.query.test_graph import EventLog, _deps, _state
from tests.unit.query.test_audit import (
    SNAPSHOT,
    SCOPE,
    packed_evidence,
    authorization,
    FakeAuthorizationResolver,
)


def phase_api():
    assert importlib.util.find_spec("agentic_rag.query.phases"), "phases missing"
    from agentic_rag.query import phases

    return phases


async def test_phase_emission_preserves_order_and_unique_keys_without_payloads():
    log = EventLog()
    producer = phase_api().QueryPhaseEmitter(
        run_id="r",
        scope=SCOPE,
        snapshot=SNAPSHOT,
        event_emitter=AgentEventEmitter(
            log, None, runtime_config_snapshot_id=SNAPSHOT.snapshot_id
        ),
    )
    for phase in ("processing", "auditing", "processing", "auditing", "SECRET"):
        await producer.report(phase)
    assert [e.summary for e in log.events] == [
        "processing",
        "auditing",
        "processing",
        "auditing",
    ]
    assert len({e.event_key for e in log.events}) == 4
    assert all(e.payload_ref is None for e in log.events)


async def test_phase_reporting_failure_is_isolated_but_cancellation_propagates():
    class Broken:
        async def append(self, event):
            raise OSError("unavailable")

    producer = phase_api().QueryPhaseEmitter(
        run_id="r", scope=SCOPE, snapshot=SNAPSHOT, event_repository=Broken()
    )
    await producer.report("retrieving")

    class Cancelled:
        async def append(self, event):
            raise asyncio.CancelledError()

    producer = phase_api().QueryPhaseEmitter(
        run_id="r", scope=SCOPE, snapshot=SNAPSHOT, event_repository=Cancelled()
    )
    with pytest.raises(asyncio.CancelledError):
        await producer.report("retrieving")


async def test_audit_phase_precedes_actual_audit_and_repeats_on_revision():
    from agentic_rag.query.audit import (
        generate_with_mandatory_audits,
        CitationValidator,
        FaithfulnessAudit,
    )
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment

    calls = []

    class Generator:
        async def generate(self, *args, **kwargs):
            calls.append("generate")
            return AnswerDraft(
                segments=(
                    AnswerSegment(
                        kind="content", text="Fact", evidence_ids=("evidence-1",)
                    ),
                )
            )

    class Auditor:
        async def audit(self, *args, **kwargs):
            passed = "audit_fail" in calls
            calls.append("audit_pass" if passed else "audit_fail")
            return FaithfulnessAudit(passed=passed)

    async def report(phase):
        calls.append(phase)

    result = await generate_with_mandatory_audits(
        question="q",
        state={},
        packed_evidence=packed_evidence(),
        scope=SCOPE,
        snapshot=SNAPSHOT,
        generator=Generator(),
        faithfulness_auditor=Auditor(),
        citation_validator=CitationValidator(),
        authorization={},
        authorization_resolver=FakeAuthorizationResolver(authorization()["evidence-1"]),
        report_phase=report,
    )
    assert calls == [
        "processing",
        "generate",
        "auditing",
        "audit_fail",
        "processing",
        "generate",
        "auditing",
        "audit_pass",
    ]
    assert result["answer"]["audited"] and result["revision_count"] == 1


@pytest.mark.parametrize("broken", [False, True])
async def test_fast_path_phases_precede_retrieval_and_grading_without_affecting_result(
    broken,
):
    from agentic_rag.query.fast_rag import FastRagDependencies, run_fast_rag
    from tests.unit.query.test_router_fast_path import (
        _initial_state,
        FakeRetrieval,
        FakeGrader,
        RouteDecision,
    )
    from agentic_rag.models.schemas import EvidenceGrade
    from agentic_rag.query.evidence_builder import EvidenceBuilder

    calls = []

    class Retrieval(FakeRetrieval):
        async def retrieve(self, *args):
            calls.append("retrieve")
            return await super().retrieve(*args)

    class Grader(FakeGrader):
        async def grade(self, *args, **kwargs):
            calls.append("grade")
            return await super().grade(*args, **kwargs)

    async def report(phase):
        calls.append(phase)
        if broken:
            raise OSError("offline")

    state = _initial_state()
    state["route"] = RouteDecision(
        route="fast_rag", normalized_query="q", reason_code="simple"
    ).model_dump(mode="json")
    result = await run_fast_rag(
        state,
        FastRagDependencies(
            Retrieval(), EvidenceBuilder(), Grader(EvidenceGrade(decision="sufficient"))
        ),
        report_phase=report,
    )
    assert result["next_node"] == "generate"
    assert calls == ["retrieving", "retrieve", "auditing", "grade"]


@pytest.mark.parametrize("route", ["chat", "fast_rag", "research"])
async def test_graph_phases_follow_executed_path(route):
    from agentic_rag.query.graph import build_query_graph, query_checkpoint_config

    deps, _, _, log = _deps(route=route)
    if route == "chat":
        deps.gateway.answer_values[:] = [{"text": "你好"}]
    state = _state()
    await build_query_graph(deps).ainvoke(state, query_checkpoint_config(state))
    phases = [e.summary for e in log.events if e.event_type == "QUERY_PHASE_CHANGED"]
    assert phases[0] == "processing"
    if route == "chat":
        assert set(phases) == {"processing"}
    if route == "research":
        assert "researching" in phases and "retrieving" not in phases
    if route == "fast_rag":
        assert phases[1:5] == ["retrieving", "auditing", "processing", "auditing"]
        assert phases[5:] in ([], ["processing", "auditing"])


async def test_reader_batches_scoped_latest_valid_phases():
    assert importlib.util.find_spec("agentic_rag.runtime.query_phase_reader"), (
        "reader missing"
    )
    from agentic_rag.runtime.query_phase_reader import QueryPhaseReader

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(agent_events.create)
            for i, (run, user, phase) in enumerate(
                [
                    ("r", "a", "retrieving"),
                    ("r", "a", "auditing"),
                    ("r", "a", "PRIVATE"),
                    ("r", "b", "researching"),
                    ("x", "a", "researching"),
                ],
                1,
            ):
                await conn.execute(
                    insert(agent_events).values(
                        id=i,
                        event_key=str(i),
                        trace_id=run,
                        run_id=run,
                        user_id=user,
                        event_type="QUERY_PHASE_CHANGED",
                        summary=phase,
                        runtime_config_snapshot_id="s",
                        created_at=datetime.now(UTC),
                    )
                )
        assert await QueryPhaseReader(async_sessionmaker(engine)).latest(
            UserScope(user_id="a"), ["r", "none"]
        ) == {"r": "auditing"}
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "phase", ["processing", "retrieving", "researching", "auditing", "PRIVATE"]
)
def test_sse_projects_only_fixed_phase_enum(phase):
    from agentic_rag.api.query_runs import _sse_event
    import json

    event = AgentEvent(
        id=1,
        event_key="key",
        trace_id="r",
        run_id="r",
        user_id="a",
        event_type="QUERY_PHASE_CHANGED",
        summary=phase,
        runtime_config_snapshot_id="s",
        payload_ref="artifact://secret",
    )
    value = json.loads(_sse_event(event).split("data: ")[1])
    assert "attributes" not in value and "PRIVATE" not in str(value)
    if phase != "PRIVATE":
        assert value["phase"] == phase
    else:
        assert "phase" not in value
