"""Production QueryGraph composition with experiment-local write boundaries.

No HTTP/Outbox/Worker timing is claimed. No gold reaches the graph. Provider,
retriever, reranker, SQL Parent hydration, prompts and graph nodes are real.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import datetime, UTC
import time
from types import SimpleNamespace
from uuid import uuid4

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.tracing import TraceRecorder
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.query.graph import build_query_graph, query_checkpoint_config
from agentic_rag.query.public_answer import project_public_answer
from agentic_rag.query.state import new_query_state
from agentic_rag.runtime.models import EvaluationMetadata, RuntimeConfigSnapshot
from agentic_rag.runtime.query_composition import build_query_dependencies
from agentic_rag.tool_runtime.models import ToolError
from evals.costs_v2 import provider_requests
from evals.gold_rebinding import _write
from evals.paired_graph import GraderObserver
from evals.trace_v2 import project_case_trace


def arm_snapshot(base, arm, dataset_sha256, corpus_snapshot_id):
    if arm not in {"ik", "ik_max_word"}:
        raise ValueError("unsupported graph arm")
    changes = {"evaluation": EvaluationMetadata(session_id="paired-graph-" + arm,
        dataset_sha256=dataset_sha256, corpus_snapshot_id=corpus_snapshot_id)}
    if arm == "ik_max_word":
        changes["retrieval_config_version"] = base.retrieval_config_version + ":eval-query-ik-max-word-v1"
    return base.model_copy(update=changes)


class DisabledMemory:
    def __getattr__(self, name):
        raise RuntimeError("evaluation must not access memory: " + name)


class BlockedExternalAdapter:
    """Keep declared capabilities; never contact unrelated remote tools for this corpus test."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.adapter_id = adapter.adapter_id
        self.config = getattr(adapter, "config", None)

    async def list_tools(self, context):
        raise ToolError("tool_unavailable")

    async def call(self, *args, **kwargs):
        raise ToolError("tool_unavailable")

    async def aclose(self):
        await self.adapter.aclose()


class LocalEvents:
    def __init__(self, artifacts):
        self.artifacts, self.path, self.records = artifacts, None, []

    def begin(self, path):
        if path.exists():
            raise ValueError("event journal already exists")
        self.path, self.records = path, []

    async def append(self, event):
        if self.path is None:
            raise RuntimeError("event journal not bound")
        record = asdict(event)
        record["created_at"] = (event.created_at or datetime.now(UTC)).isoformat()
        record["attributes"] = self.artifacts.read_json(self.artifacts.describe(event.payload_ref))["attributes"] if event.payload_ref else {}
        self.records.append(record)
        _write(self.path, self.records)
        return len(self.records)


class ProviderBudget:
    """One shared limit for real LLM+embedding sends, including SDK/gateway retries."""
    def __init__(self, limit=100):
        self.limit, self.path, self.records = limit, None, []

    def begin(self, path):
        if path.exists():
            raise ValueError("provider journal already exists")
        self.path, self.records = path, []

    def wrap(self, resource, kind):
        original = resource.create
        async def create(*args, **kwargs):
            if self.path is None or len(self.records) >= self.limit:
                raise RuntimeError("query provider request budget exhausted")
            record = {"kind": kind, "requested_model": kwargs.get("model"), "status": "running"}
            self.records.append(record)
            _write(self.path, self.records)
            try:
                result = await original(*args, **kwargs)
                record["status"] = "returned"  # Streaming consumption is measured by production telemetry.
                return result
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__)
                raise
            finally:
                _write(self.path, self.records)
        resource.create = create


def protect_sql_reads(engine):
    """Fail closed if any evaluator/composition path attempts a SQL mutation."""
    from sqlalchemy import event
    def before_execute(conn, cursor, statement, parameters, context, executemany):
        if not statement.lstrip().upper().startswith("SELECT "):
            raise ValueError("evaluation SQL boundary is read-only")
    event.listen(engine, "before_cursor_execute", before_execute)


async def compose(settings, engine, transport, root):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from agentic_rag.tool_runtime.native import NativeAdapter
    artifacts = LocalArtifactStore(root / "runtime-artifacts")
    events, budget = LocalEvents(artifacts), ProviderBudget()
    local_settings = settings.model_copy(update={"tool_invocation_path": root / "tool-invocations.sqlite"})
    container = SimpleNamespace(elasticsearch=transport, artifacts=artifacts, event_repository=events,
        memory_service=DisabledMemory(), repositories=SimpleNamespace(session_factory=async_sessionmaker(engine, expire_on_commit=False)))
    deps = await build_query_dependencies(container, local_settings, child_index=transport.index)
    # Real provider clients are the first two resources owned by production composition.
    llm, embedding = deps.owned_resources[:2]
    budget.wrap(llm.chat.completions, "llm")
    budget.wrap(llm.responses, "llm")
    budget.wrap(embedding.embeddings, "embedding")
    if deps.tool_runtime is not None:
        runtime = deps.tool_runtime
        runtime.adapters = tuple(a if isinstance(a, NativeAdapter) else BlockedExternalAdapter(a) for a in runtime.adapters)
        runtime._by_id = {a.adapter_id: a for a in runtime.adapters}
    return deps, events, budget


def project_graph_result(state, *, error_code):
    snapshot = RuntimeConfigSnapshot.model_validate(state["runtime_config_snapshot"])
    route = state.get("route", {}).get("route")
    termination = state.get("termination_reason")
    answer = dict(state.get("answer") or {"status": termination})
    if termination == "completed":
        answer.pop("status", None)
    else:
        answer["status"] = termination
    public = None if error_code else project_public_answer(answer,
        evidence_parent_ids=(list(dict.fromkeys(i["parent_id"] for i in state.get("evidence", [])))
                             if answer.get("audited") is True else []),
        route=route, runtime_config_snapshot_id=snapshot.snapshot_id if termination == "completed" else None,
        require_audited=termination == "completed" and route != "chat")
    run = {"id": state["run_id"], "user_id": state["scope"]["user_id"], "question": state["request"]["question"],
           "runtime_config_snapshot_id": snapshot.snapshot_id, "status": "failed" if error_code else "completed",
           "error_code": error_code, "route": route, "answer": public.model_dump(mode="json") if public else None}
    trace = project_case_trace(run, state, snapshot_id=snapshot.snapshot_id)
    mode = state.get("response_mode")
    if not error_code and mode in {"technical_error", "clarify", "capability_unavailable"}:
        trace["answer_origin"] = "controlled_response"
        if mode == "technical_error":
            from agentic_rag.query.routing_policy import PolicyDecision, technical_failure
            policy = PolicyDecision.model_validate(state["policy_decision"])
            # Revalidate against production's bounded technical failure codes;
            # preserve the public terminal text, but never score it as model output.
            technical_failure(policy.reason_code)
            trace["outcome"] = policy.reason_code
    trace["client_provenance"] = "real_query_graph"
    return trace


async def execute_graph(*, deps, events, budget, transport, snapshot, arm, question, user_id, directory):
    """One fresh graph execution, with unchanged production deadlines and actions."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    events.begin(directory / "events.json")
    budget.begin(directory / "provider-budget.json")
    transport.begin(arm, directory / "searches.json")
    grader = GraderObserver(deps.evidence_grader, directory / "grades.json")
    base_id = snapshot.model_copy(update={"evaluation": None}).snapshot_id
    scoped = replace(deps, evidence_grader=grader, deployment_snapshot_id=base_id,
        event_emitter=deps.event_emitter.with_snapshot(base_id),
        trace_recorder=TraceRecorder(runtime_config_snapshot_id=snapshot.snapshot_id))
    graph = build_query_graph(scoped)
    state = new_query_state(run_id=str(uuid4()), question=question, scope=UserScope(user_id=user_id), snapshot=snapshot)
    _write(directory / "state.json", state)
    history, error_code = [], None
    start = time.monotonic()
    try:
        async with asyncio.timeout(snapshot.query_run_timeout_seconds):
            async for current in graph.astream(state, config=query_checkpoint_config(state), stream_mode="values"):
                state = current
                _write(directory / "state.json", {k: v for k, v in state.items() if k not in {"messages", "memory_context"}})
                history.append({"elapsed_seconds": time.monotonic() - start, "next_node": state.get("next_node"),
                    "research_attempt_count": state["research_attempt_count"], "termination_reason": state.get("termination_reason"),
                    "todos": state.get("research", {}).get("todos", []), "retrieval_calls": len(state.get("retrieval_batches", []))})
                _write(directory / "history.json", history)
    except Exception as error:
        error_code = type(error).__name__
    if not state.get("termination_reason") and not error_code:
        error_code = "graph_without_terminal_state"
    trace = project_graph_result(state, error_code=error_code)
    trace.update(total_seconds=time.monotonic() - start, grade_history=grader.records, history=history,
                 provider_requests=provider_requests(events.records), provider_budget_attempts=len(budget.records))
    requests = trace["provider_requests"]
    if len(requests["requests"]) != len(budget.records):
        requests["status"] = "incomplete"
    trace["provider_usage"] = {"status": requests["status"], "cost": None,
        **{key: sum(r[key] for r in requests["requests"]) if requests["status"] == "available" else None
           for key in ("input_tokens", "output_tokens")}}
    return trace
