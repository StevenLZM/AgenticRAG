"""The fixed, checkpoint-safe macro graph for audited query runs.

Only JSON-compatible values cross a LangGraph checkpoint.  Services are held
by :class:`QueryGraphDependencies` and captured by node closures instead of
being placed in ``QueryState``.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.memory.models import PublicMessage
from agentic_rag.query.chat import run_chat
from agentic_rag.memory.service import MemoryService
from agentic_rag.observability.logging import (
    AgentEventEmitter,
    event_emission_scope,
    sanitize_summary,
    stable_event_key,
)
from agentic_rag.observability.tracing import TraceRecorder
from agentic_rag.persistence.repositories import AgentEvent, EventRepository
from agentic_rag.query.audit import (
    CitationValidator,
    EvidenceAuthorizationResolver,
    FaithfulnessAuditor,
    generate_with_mandatory_audits,
)
from agentic_rag.query.evidence_builder import (
    EvidenceBuilder,
    EvidenceCoverageTarget,
    PackedEvidence,
)
from agentic_rag.query.fast_rag import FastRagDependencies, EvidenceGrader as EvidenceGraderPort, run_fast_rag
from agentic_rag.query.generation import AnswerGenerator
from agentic_rag.query.research_loop import ResearchAgentLoop
from agentic_rag.query.router import MemoryContextLoader, route_query
from agentic_rag.models.schemas import EvidenceGrade, RouteAssessment
from agentic_rag.query.audit import EvidenceGradingUnavailable
from agentic_rag.query.routing_context import (
    ConversationReader, RoutingContext, RoutingContextUnavailable, fresh_routing_state,
    prepare_routing_state, reasoning_question,
)
from agentic_rag.query.routing_policy import RuntimeCapabilities, decide_grade, policy_update, technical_failure
from agentic_rag.query.state import QueryState, question_from_state, scope_from_state, snapshot_from_state
from agentic_rag.retrieval.graph import RetrievalService
from agentic_rag.retrieval.models import EvidenceBatch
from agentic_rag.runtime.concurrency import ConcurrencyManager
from agentic_rag.runtime.model_gateway import ModelGateway


class QueryGraphEventRecorder(Protocol):
    """Small event port kept here to make unit graphs independent of SQL."""

    async def append(self, event: AgentEvent) -> int: ...


@dataclass(frozen=True, slots=True)
class QueryGraphDependencies:
    """Process-owned collaborators captured by graph nodes, never checkpointed."""

    memory: MemoryService
    gateway: ModelGateway
    retrieval: RetrievalService
    evidence_builder: EvidenceBuilder
    evidence_grader: EvidenceGraderPort
    research_loop: ResearchAgentLoop
    generator: AnswerGenerator
    faithfulness_auditor: FaithfulnessAuditor
    citation_validator: CitationValidator
    authorization_resolver: EvidenceAuthorizationResolver
    event_repository: EventRepository | QueryGraphEventRecorder | None = None
    trace_recorder: TraceRecorder | None = None
    event_emitter: AgentEventEmitter | None = None
    concurrency: ConcurrencyManager | None = None
    owned_resources: tuple[object, ...] = ()
    capabilities: RuntimeCapabilities | None = None
    conversations: ConversationReader | None = None


def query_checkpoint_config(state: QueryState) -> RunnableConfig:
    """Build the only safe checkpoint namespace from server-owned query state.

    A bare client thread id is intentionally never accepted by this helper.
    Task 8 uses the same ``query:{user_id}:{thread_id}`` convention when it
    owns a durable Run and resumes one of its checkpoints.
    """
    scope = scope_from_state(state)
    request = state.get("request", {})
    thread_id = request.get("thread_id") if isinstance(request, Mapping) else None
    safe_thread = thread_id if isinstance(thread_id, str) and thread_id.strip() else state["run_id"]
    return {"configurable": {"thread_id": f"query:{scope.user_id}:{safe_thread}"}, "recursion_limit": 50}


def build_query_graph(
    dependencies: QueryGraphDependencies,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[QueryState, None, QueryState, QueryState]:
    """Compile the approved, fixed topology for a single query run.

    Fast RAG owns its one retrieval and first grading attempt from Task 3.  The
    macro graph records that grade and takes over grading after Research loops;
    both paths converge on the exact same audited answer publication nodes.
    """
    memory_loader = MemoryContextLoader(dependencies.memory)
    fast_dependencies = FastRagDependencies(
        retrieval=dependencies.retrieval,
        evidence_builder=dependencies.evidence_builder,
        evidence_grader=dependencies.evidence_grader,
        capabilities=dependencies.capabilities,
    )

    async def load_memory(state: QueryState, config: RunnableConfig) -> dict[str, object]:
        _assert_server_checkpoint_namespace(state, config)
        prepared: dict[str, object] = {}
        if dependencies.capabilities is not None:
            prepared = fresh_routing_state(state)
            state = cast(QueryState, {**state, **prepared})
            try:
                prepared.update(await prepare_routing_state(state, capabilities=dependencies.capabilities,
                                                           conversations=dependencies.conversations))
            except RoutingContextUnavailable:
                return {**prepared, **policy_update(state, technical_failure("routing_context_unavailable")),
                        "errors": [{"code": "routing_context_unavailable"}]}
            state = cast(QueryState, {**state, **prepared})
        async with _trace_span(dependencies, state, "graph.node.memory_loader"):
            async with _trace_span(dependencies, state, "memory"):
                update = await memory_loader.load(state)
        await _event(dependencies, state, "MEMORY_LOADED", "memory context loaded")
        return {**prepared, **update}

    async def route(state: QueryState) -> dict[str, object]:
        if dependencies.capabilities is not None and state.get("response_mode") == "technical_error":
            return {"next_node": "chat"}
        async with _trace_span(dependencies, state, "graph.node.route"):
            async with _trace_span(dependencies, state, "llm"):
                update = await route_query(state, dependencies.gateway, capabilities=dependencies.capabilities)
        await _event(dependencies, state, "QUERY_ROUTED", str(update.get("next_node", "research_agent")))
        return update

    async def chat(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.chat"):
            return await run_chat(state, dependencies.gateway)

    async def fast_rag(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.fast_rag"):
            async with _trace_span(dependencies, state, "retrieval"):
                async with _trace_span(dependencies, state, "rerank"):
                    update = await run_fast_rag(state, fast_dependencies)
        await _event(
            dependencies,
            cast(QueryState, {**state, **update}),
            "FAST_RAG_COMPLETED",
            str(update.get("next_node", "research_agent")),
        )
        return update

    async def record_fast_grade(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.record_fast_grade"):
            await _event(dependencies, state, "EVIDENCE_GRADED", _grade_summary(state))
        return {}

    async def research_agent_loop(state: QueryState) -> dict[str, object]:
        if dependencies.capabilities is not None and state.get("last_evidence_grade"):
            grade = EvidenceGrade.model_validate(state["last_evidence_grade"])
            if grade.gap_type in {"external_realtime_required", "external_lookup_required"}:
                assessment = RouteAssessment.model_validate(state["route_assessment"]) if state.get("route_assessment") else None
                return policy_update(state, decide_grade(grade, assessment, dependencies.capabilities,
                    research_attempts=state.get("research_attempt_count", 0), max_research_rounds=snapshot_from_state(state).max_research_rounds))
        async with _trace_span(dependencies, state, "graph.node.research_agent_loop"):
            update = await dependencies.research_loop.ainvoke(state)
        await _event(
            dependencies,
            cast(QueryState, {**state, **update}),
            "PROGRESS" if update.get("next_node") == "research_agent" else "RESEARCH_LOOP_COMPLETED",
            str(update.get("next_node", "end")),
        )
        return update

    async def evidence_builder(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.evidence_builder"):
            async with _trace_span(dependencies, state, "rerank"):
                raw_batches = state.get("retrieval_batches")
                if not isinstance(raw_batches, list) or not raw_batches:
                    return {
                        "packed_context": _empty_pack(state).model_dump(mode="json"),
                        "evidence": [],
                        "errors": [*state.get("errors", []), {"code": "research_batches_missing"}],
                        "termination_reason": "refuse",
                        "next_node": "end",
                    }
                try:
                    batches = tuple(EvidenceBatch.model_validate(value) for value in raw_batches)
                except (TypeError, ValueError):
                    return {
                        "packed_context": _empty_pack(state).model_dump(mode="json"),
                        "evidence": [],
                        "errors": [*state.get("errors", []), {"code": "research_batches_invalid"}],
                        "termination_reason": "refuse",
                        "next_node": "end",
                    }
                try:
                    packed = dependencies.evidence_builder.build(
                        batches,
                        _coverage_targets(state),
                        scope_from_state(state),
                        snapshot_from_state(state),
                    )
                except (TypeError, ValueError):
                    return {
                        "packed_context": _empty_pack(state).model_dump(mode="json"),
                        "evidence": [],
                        "errors": [*state.get("errors", []), {"code": "research_evidence_invalid"}],
                        "termination_reason": "refuse",
                        "next_node": "end",
                    }
                return {
                    "packed_context": packed.model_dump(mode="json"),
                    "evidence": [item.model_dump(mode="json") for item in packed.items],
                }

    async def evidence_grader(state: QueryState) -> dict[str, object]:
        if dependencies.capabilities is not None:
            try:
                async with _trace_span(dependencies, state, "graph.node.evidence_grader"):
                    grade_v2 = await dependencies.evidence_grader.grade(
                        reasoning_question(state), _packed_from_state(state), scope=scope_from_state(state),
                        snapshot=snapshot_from_state(state), capabilities=dependencies.capabilities,
                        routing_context=RoutingContext.model_validate(state["routing_context"]) if state.get("routing_context") else None)
                grade_v2 = EvidenceGrade.model_validate(_unwrap(grade_v2))
            except (EvidenceGradingUnavailable, OSError, TimeoutError, ValueError, TypeError):
                return {**policy_update(state, technical_failure("evidence_grader_unavailable")),
                        "errors": [*state.get("errors", []), {"code": "evidence_grader_unavailable"}]}
            assessment = RouteAssessment.model_validate(state["route_assessment"]) if state.get("route_assessment") else None
            await _event(dependencies, state, "EVIDENCE_GRADED", grade_v2.decision)
            decision_v2 = decide_grade(grade_v2, assessment, dependencies.capabilities,
                research_attempts=state.get("research_attempt_count", 0), max_research_rounds=snapshot_from_state(state).max_research_rounds)
            return {**policy_update(state, decision_v2), "last_evidence_grade": grade_v2.model_dump(mode="json"),
                    "research": {**state.get("research", {}), "gaps": list(grade_v2.gaps)}}
        async with _trace_span(dependencies, state, "graph.node.evidence_grader"):
            async with _trace_span(dependencies, state, "llm"):
                packed = _packed_from_state(state)
                grade = await dependencies.evidence_grader.grade(
                    question_from_state(state), packed,
                    scope=scope_from_state(state), snapshot=snapshot_from_state(state),
                )
        decision = _grade_decision(grade)
        await _event(dependencies, state, "EVIDENCE_GRADED", decision)
        if decision == "sufficient":
            return {"next_node": "generate"}
        if decision == "insufficient":
            gaps = list(getattr(_unwrap(grade), "gaps", ()))
            return {"research": {**_mapping(state.get("research")), "gaps": gaps}, "next_node": "research_agent"}
        return {
            "answer": {"status": decision}, "termination_reason": decision,
            "next_node": "end",
        }

    async def generate(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.generate"):
            async with _trace_span(dependencies, state, "llm"):
                async with _trace_span(dependencies, state, "audit"):
                    packed = _packed_from_state(state)
                    if not packed.items:
                        return {
                            "answer": {},
                            "errors": [*state.get("errors", []), {"code": "verified_evidence_missing"}],
                            "termination_reason": "refuse",
                            "next_node": "end",
                        }
                    # `generate_with_mandatory_audits` is the sole owner of revision
                    # policy; subsequent graph nodes expose its completed gates only.
                    update = await generate_with_mandatory_audits(
                        question=reasoning_question(state), state=state, packed_evidence=packed,
                        scope=scope_from_state(state), snapshot=snapshot_from_state(state),
                        generator=dependencies.generator, faithfulness_auditor=dependencies.faithfulness_auditor,
                        citation_validator=dependencies.citation_validator, authorization={},
                        authorization_resolver=dependencies.authorization_resolver,
                    )
        await _event(dependencies, state, "ANSWER_GENERATED", "audited draft generated")
        return update

    async def faithfulness(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.faithfulness"):
            async with _trace_span(dependencies, state, "audit"):
                audit = _latest_audit(state, "faithfulness")
                await _event(dependencies, state, "FAITHFULNESS_AUDITED", "passed" if audit.get("passed") else "failed")
        return {}

    async def citation(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.citation"):
            async with _trace_span(dependencies, state, "audit"):
                audit = _latest_audit(state, "citation")
                await _event(dependencies, state, "CITATION_VALIDATED", "passed" if audit.get("passed") else "failed")
        return {}

    async def finalize(state: QueryState) -> dict[str, object]:
        async with _trace_span(dependencies, state, "graph.node.finalize"):
            async with _trace_span(dependencies, state, "memory"):
                termination = state.get("termination_reason")
                completed = "completed" if termination is None else str(termination)
                await _event(dependencies, state, "ANSWER_FINALIZED", completed)
                # Publication precedes best-effort memory extraction.  A memory outage
                # must never retract an answer that has already passed audits.
                if snapshot_from_state(state).evaluation is not None:
                    return {"termination_reason": completed, "next_node": "end"}
                try:
                    await dependencies.memory.extract_and_store(
                        scope_from_state(state), state["run_id"], _public_messages(state)
                    )
                except asyncio.CancelledError:
                    raise
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    return {
                        "termination_reason": completed,
                        "errors": [*state.get("errors", []), {"code": "memory_finalize_unavailable", "detail": type(error).__name__}],
                        "next_node": "end",
                    }
                return {"termination_reason": completed, "next_node": "end"}

    def after_route(state: QueryState) -> str:
        if state.get("next_node") == "chat":
            return "chat"
        return "fast_rag" if state.get("next_node") == "fast_rag" else "research_agent_loop"

    def after_fast_grade(state: QueryState) -> str:
        if state.get("next_node") == "chat":
            return "chat"
        next_node = state.get("next_node")
        if next_node == "generate":
            return "generate"
        if next_node == "research_agent":
            return "research_agent_loop"
        return "finalize"

    def after_research(state: QueryState) -> str:
        if state.get("next_node") == "chat":
            return "chat"
        if state.get("next_node") == "research_agent":
            return "research_agent_loop"
        return "evidence_builder" if state.get("next_node") == "generate" else "finalize"

    def after_evidence_builder(state: QueryState) -> str:
        return "finalize" if state.get("next_node") == "end" else "evidence_grader"

    def after_grade(state: QueryState) -> str:
        if state.get("next_node") == "chat":
            return "chat"
        if state.get("next_node") == "generate":
            return "generate"
        if state.get("next_node") == "research_agent":
            return "research_agent_loop"
        return "finalize"

    def tracked(name, function):
        async def invoke(state: QueryState, config: RunnableConfig):
            if name == "memory_loader":
                update = await function(state, config)
            else:
                update = await function(state)
            if dependencies.capabilities is not None:
                previous = [] if name == "memory_loader" and state.get("routing_owner_run_id") != state["run_id"] else state.get("executed_path", [])
                update["executed_path"] = [*previous, name][-64:]
            return update
        return invoke

    builder = StateGraph(QueryState)
    for name, function in [("memory_loader", load_memory), ("route", route), ("chat", chat),
                           ("fast_rag", fast_rag), ("record_fast_grade", record_fast_grade),
                           ("research_agent_loop", research_agent_loop), ("evidence_builder", evidence_builder),
                           ("evidence_grader", evidence_grader), ("generate", generate),
                           ("faithfulness", faithfulness), ("citation", citation), ("finalize", finalize)]:
        builder.add_node(name, tracked(name, function))
    builder.add_edge(START, "memory_loader")
    builder.add_edge("memory_loader", "route")
    builder.add_conditional_edges("route", after_route, {"chat": "chat", "fast_rag": "fast_rag", "research_agent_loop": "research_agent_loop"})
    builder.add_edge("chat", "finalize")
    builder.add_edge("fast_rag", "record_fast_grade")
    builder.add_conditional_edges("record_fast_grade", after_fast_grade, {"chat": "chat", "generate": "generate", "research_agent_loop": "research_agent_loop", "finalize": "finalize"})
    builder.add_conditional_edges("research_agent_loop", after_research, {"chat": "chat", "research_agent_loop": "research_agent_loop", "evidence_builder": "evidence_builder", "finalize": "finalize"})
    builder.add_conditional_edges("evidence_builder", after_evidence_builder, {"evidence_grader": "evidence_grader", "finalize": "finalize"})
    builder.add_conditional_edges("evidence_grader", after_grade, {"chat": "chat", "generate": "generate", "research_agent_loop": "research_agent_loop", "finalize": "finalize"})
    builder.add_edge("generate", "faithfulness")
    builder.add_edge("faithfulness", "citation")
    builder.add_edge("citation", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="QueryGraph")


def _assert_server_checkpoint_namespace(state: QueryState, config: RunnableConfig) -> None:
    """Reject a supplied bare/external checkpoint key while allowing unit calls."""
    configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
    thread_id = configurable.get("thread_id") if isinstance(configurable, Mapping) else None
    if thread_id is not None and (not isinstance(thread_id, str) or not thread_id.startswith(f"query:{scope_from_state(state).user_id}:")):
        raise ValueError("query checkpoint thread_id must use the server query namespace")


@asynccontextmanager
async def _trace_span(
    dependencies: QueryGraphDependencies, state: QueryState, name: str
) -> AsyncIterator[None]:
    """Record a safe local span only when it matches the Run's snapshot."""
    snapshot_id = snapshot_from_state(state).snapshot_id
    recorder = dependencies.trace_recorder
    emitter = dependencies.event_emitter
    safe_recorder = (
        recorder if recorder is not None and recorder.runtime_config_snapshot_id == snapshot_id else None
    )
    safe_emitter = (
        emitter if emitter is not None and emitter.runtime_config_snapshot_id == snapshot_id else None
    )
    if safe_emitter is None:
        if safe_recorder is None:
            yield
            return
        async with safe_recorder.span(name, run_id=state["run_id"]):
            yield
        return
    async with event_emission_scope(
        safe_emitter,
        state["run_id"],
        name,
        user_id=scope_from_state(state).user_id,
    ):
        await _span_event(safe_emitter, state, name, "GRAPH_NODE_STARTED")
        started = time.perf_counter()
        try:
            if safe_recorder is None:
                yield
            else:
                async with safe_recorder.span(name, run_id=state["run_id"]):
                    yield
        finally:
            await _span_event(
                safe_emitter,
                state,
                name,
                "GRAPH_NODE_COMPLETED",
                attributes={"node_latency_seconds": time.perf_counter() - started},
            )


async def _span_event(
    emitter: AgentEventEmitter,
    state: QueryState,
    name: str,
    event_type: str,
    *,
    attributes: Mapping[str, object] | None = None,
) -> None:
    """Emit fixed graph-boundary events without retaining graph state."""
    try:
        await emitter.emit(
            run_id=state["run_id"],
            user_id=scope_from_state(state).user_id,
            event_type=event_type,
            node_name=name,
            summary="started" if event_type.endswith("STARTED") else "completed",
            attributes=attributes,
            event_key=stable_event_key(
                state["run_id"],
                name,
                event_type,
                str(state.get("revision_count", 0)),
                str(len(state.get("audit_results", []))),
                str(len(state.get("retrieval_batches", []))),
            ),
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        pass


async def _event(dependencies: QueryGraphDependencies, state: QueryState, event_type: str, summary: str) -> None:
    if event_type == "QUERY_ROUTED":
        summary = {"chat": "chat", "fast_rag": "fast_rag", "research_agent": "research"}.get(summary, "completed")
    if dependencies.event_emitter is not None:
        if dependencies.event_emitter.runtime_config_snapshot_id != snapshot_from_state(state).snapshot_id:
            return
        try:
            await dependencies.event_emitter.emit(
                run_id=state["run_id"],
                user_id=scope_from_state(state).user_id,
                event_type=event_type,
                node_name=event_type.lower(),
                summary=summary if event_type == "QUERY_ROUTED" else "completed",
                attributes=_event_attributes(state, event_type),
                event_key=stable_event_key(
                    state["run_id"],
                    event_type,
                    str(state.get("revision_count", 0)),
                    str(state.get("research_attempt_count", 0)),
                    str(len(state.get("audit_results", []))),
                    str(state.get("next_node", "")),
                ),
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
        return
    if dependencies.event_repository is None:
        return
    snapshot = snapshot_from_state(state)
    safe_event_type = event_type if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", event_type) else "PROGRESS"
    safe_summary = sanitize_summary(summary)
    event = AgentEvent(
        event_key=stable_event_key(
            state["run_id"],
            safe_event_type,
            str(state.get("revision_count", 0)),
            str(state.get("research_attempt_count", 0)),
            str(len(state.get("audit_results", []))),
            str(len(state.get("retrieval_batches", []))),
            str(state.get("next_node", "")),
        ),
        trace_id=state["run_id"], run_id=state["run_id"], user_id=scope_from_state(state).user_id,
        event_type=safe_event_type, summary=safe_summary, runtime_config_snapshot_id=snapshot.snapshot_id,
        node_name=safe_event_type.lower(),
    )
    await dependencies.event_repository.append(event)


def _event_attributes(state: Mapping[str, object], event_type: str) -> dict[str, object]:
    """Derive only numeric/enumerated projection fields from trusted graph state."""
    attributes: dict[str, object] = {}
    if event_type == "FAST_RAG_COMPLETED":
        attributes["retrieval_rounds"] = 1
    if event_type in {"FAST_RAG_COMPLETED", "RESEARCH_LOOP_COMPLETED"}:
        batches = state.get("retrieval_batches")
        if isinstance(batches, list):
            attributes["retrieval_rounds"] = len(batches)
        elif event_type == "RESEARCH_LOOP_COMPLETED":
            attributes["retrieval_rounds"] = 0
        evidence = state.get("evidence")
        if isinstance(evidence, list):
            attributes["candidate_count"] = len(evidence)
        elif event_type == "RESEARCH_LOOP_COMPLETED":
            attributes["candidate_count"] = 0
    if event_type == "RESEARCH_LOOP_COMPLETED":
        attempts = state.get("research_attempt_count")
        attributes["research_attempt_count"] = (
            attempts if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 0 else 0
        )
        research = state.get("research")
        todos = research.get("todos") if isinstance(research, Mapping) else None
        attributes["todo_count"] = len(todos) if isinstance(todos, list) else 0
    if event_type == "CITATION_VALIDATED":
        answer = state.get("answer")
        segments = answer.get("segments") if isinstance(answer, Mapping) else None
        if isinstance(segments, list):
            content = [segment for segment in segments if isinstance(segment, Mapping) and segment.get("kind") == "content"]
            attributes["claim_count"] = len(content)
            attributes["cited_claim_count"] = sum(
                1 for segment in content if isinstance(segment.get("evidence_ids"), list) and segment["evidence_ids"]
            )
        else:
            attributes["claim_count"] = 0
            attributes["cited_claim_count"] = 0
    if event_type == "ANSWER_FINALIZED":
        termination = state.get("termination_reason")
        attributes["termination_reason"] = termination if isinstance(termination, str) else "completed"
        revision_count = state.get("revision_count")
        if isinstance(revision_count, int) and revision_count > 0:
            attributes["repair_count"] = revision_count
    return attributes


def _empty_pack(state: QueryState) -> PackedEvidence:
    return PackedEvidence(
        items=(), manifest={}, rendered_context="", token_count=0,
        index_generation=snapshot_from_state(state).index_generation,
    )


def _coverage_targets(state: QueryState) -> tuple[EvidenceCoverageTarget, ...]:
    targets: list[EvidenceCoverageTarget] = [
        EvidenceCoverageTarget(target_id=f"query:{state['run_id']}", description=question_from_state(state))
    ]
    research = state.get("research")
    todos = research.get("todos") if isinstance(research, Mapping) else None
    if isinstance(todos, list):
        for todo in todos:
            if not isinstance(todo, Mapping):
                continue
            todo_id = todo.get("id")
            title = todo.get("title")
            if isinstance(todo_id, str) and isinstance(title, str) and title.strip():
                targets.append(EvidenceCoverageTarget(target_id=todo_id, description=title.strip()))
    return tuple(dict((target.target_id, target) for target in targets).values())


def _packed_from_state(state: QueryState) -> PackedEvidence:
    packed = state.get("packed_context")
    if not isinstance(packed, Mapping):
        return _empty_pack(state)
    try:
        result = PackedEvidence.model_validate(packed)
    except (TypeError, ValueError):
        return _empty_pack(state)
    snapshot = snapshot_from_state(state)
    return result if result.index_generation == snapshot.index_generation else _empty_pack(state)


def _grade_decision(value: object) -> str:
    decision = getattr(_unwrap(value), "decision", None)
    if decision not in {"sufficient", "insufficient", "clarify", "refuse"}:
        raise ValueError("evidence grader returned an invalid decision")
    return cast(str, decision)


def _unwrap(value: object) -> object:
    return getattr(value, "value", value)


def _grade_summary(state: QueryState) -> str:
    if state.get("termination_reason") in {"clarify", "refuse"}:
        return str(state["termination_reason"])
    return "sufficient" if state.get("next_node") == "generate" else "insufficient"


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _latest_audit(state: QueryState, key: str) -> dict[str, object]:
    records = state.get("audit_results", [])
    if not isinstance(records, Sequence) or not records:
        return {"passed": False}
    last = records[-1]
    if not isinstance(last, Mapping):
        return {"passed": False}
    value = last.get(key)
    return dict(value) if isinstance(value, Mapping) else {"passed": False}


def _public_messages(state: QueryState) -> list[PublicMessage]:
    messages: list[PublicMessage] = []
    for index, value in enumerate(state.get("messages", [])):
        if not isinstance(value, Mapping):
            continue
        try:
            messages.append(PublicMessage.model_validate({"id": value.get("id", f"message-{index}"), **value}))
        except (TypeError, ValueError):
            continue
    return messages
