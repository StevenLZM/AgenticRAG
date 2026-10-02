"""Checkpoint-sized tool actions shared by all three execution strategies.

Models choose business arguments; identity, deadline, call IDs, publication
and adapter selection are controlled by the application.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter

from agentic_rag.query.evidence_builder import EvidenceBuilder, EvidenceCoverageTarget
from agentic_rag.query.state import QueryState, question_from_state, scope_from_state, snapshot_from_state
from agentic_rag.retrieval.models import EvidenceBatch
from agentic_rag.runtime.model_gateway import ModelCall, load_prompt
from agentic_rag.tool_runtime.models import ToolContext, ToolDefinition, ToolError, ToolResult


class DiscoverTools(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["discover_tools"]
    query: str = Field(min_length=1, max_length=1000)


class CallTool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["call_tool"]
    tool_id: str = Field(min_length=1, max_length=256)
    arguments: dict[str, Any]


class FinishTools(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["finish"]
    result_ids: tuple[str, ...] = Field(default=(), max_length=12)
    max_items: int = Field(default=5, ge=1, le=8)


class ToolReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["answer", "clarify", "cannot_answer"]
    text: str = Field(min_length=1, max_length=8000)


class EscalateTools(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["research"]


ToolAction = Annotated[DiscoverTools | CallTool | FinishTools | ToolReply | EscalateTools,
                       Field(discriminator="action")]
ACTION: TypeAdapter[Any] = TypeAdapter(ToolAction)


class ToolActionSchema(RootModel[ToolAction]):
    pass


def has_external_requirement(state: QueryState) -> bool:
    assessment = dict(state.get("route_assessment") or {})
    grade = state.get("last_evidence_grade") or {}
    sources = assessment.get("required_sources", [])
    return bool(set(sources if isinstance(sources, list) else []) & {"external_lookup", "external_realtime"}
                or grade.get("gap_type") in {"external_lookup_required", "external_realtime_required"})


def tool_context(state: QueryState) -> ToolContext:
    tool_state = state.get("tool_state") or {}
    snapshot = snapshot_from_state(state)
    deadline = tool_state.get("deadline")
    if not isinstance(deadline, (int, float)):
        deadline = time.time() + snapshot.query_run_timeout_seconds
    return ToolContext(scope=scope_from_state(state), run_id=state["run_id"],
                       session_id=str(state.get("request", {}).get("thread_id") or state["run_id"]),
                       snapshot=snapshot, deadline=float(deadline))


def tool_results(state: QueryState) -> list[ToolResult]:
    results = (state.get("tool_state") or {}).get("results", [])
    return [ToolResult.model_validate(value) for value in results] if isinstance(results, list) else []


def _state(state: QueryState) -> dict[str, Any]:
    value = dict(state.get("tool_state") or {})
    value.setdefault("steps", 0)
    value.setdefault("strategy_steps", {})
    value.setdefault("loaded", [])
    value.setdefault("results", [])
    value.setdefault("observations", [])
    value.setdefault("deadline", tool_context(state).deadline)
    return value


def _next(strategy: str) -> str:
    return "research_agent" if strategy == "research" else strategy


def _escalate(state: QueryState, tool_state: dict[str, Any]) -> dict[str, Any]:
    route = dict(state.get("route") or {})
    route.update(route="research", reason_code="tool_work_requires_research")
    tool_state["active"] = True
    return {"tool_state": tool_state, "route": route, "response_mode": None, "next_node": "research_agent"}


def _safe_end(tool_state: dict[str, Any], code: str, *, text: str | None = None,
              status: str = "cannot_answer", route: str = "chat") -> dict[str, Any]:
    answer: dict[str, Any] = {"status": status}
    # Clarification is not a factual answer or a document audit.
    if text and status == "clarify":
        answer.update(route="chat", segments=[{"kind": "content", "text": text, "evidence_ids": []}])
        route = "chat"
    return {"tool_state": tool_state, "answer": answer, "termination_reason": status,
            "next_node": "end", "response_mode": None,
            **({"route": {"route": route, "reason_code": code}} if text and status == "clarify" else {})}


def tool_prompt_context(state: QueryState) -> dict[str, Any]:
    value = _state(state)
    # Provider payloads and document batches stay outside the model transcript.
    observations = value["observations"][-12:]
    return {"question": question_from_state(state), "route_assessment": state.get("route_assessment", {}),
            "conversation": state.get("routing_context", {}), "memory": state.get("memory_context", {}),
            "capabilities": state.get("capabilities", {}), "loaded_tools": value["loaded"],
            "observations": observations, "remaining_actions": max(0, 24 - value["steps"])}


async def run_tool_step(state: QueryState, gateway: Any, runtime: Any, *, strategy: str) -> dict[str, Any]:
    value = _state(state)
    if value["steps"] >= 24 or time.time() >= value["deadline"]:
        return _safe_end(value, "tool_budget_exhausted")
    strategy_steps = value["strategy_steps"].get(strategy, 0)
    if strategy in {"chat", "fast_rag"} and strategy_steps >= (4 if strategy == "chat" else 6):
        return _escalate(state, value)
    call = ModelCall(model_role="main" if strategy == "research" else "light",
                     snapshot=snapshot_from_state(state),
                     temperature=0, max_output_tokens=4096,
                     timeout_seconds=max(0.1, value["deadline"] - time.time()),
                     messages=({"role": "system", "content": load_prompt("tool_agent_v1").content},
                               {"role": "user", "content": json.dumps(tool_prompt_context(state), ensure_ascii=False)}))
    try:
        response = await gateway.complete_structured(call, ToolActionSchema)
        raw = getattr(response, "value", response)
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json")
        action = ACTION.validate_python(raw)
    except asyncio.CancelledError:
        raise
    except Exception:
        return _safe_end(value, "tool_model_unavailable")
    return await execute_tool_action(state, action, runtime, strategy=strategy)


async def execute_tool_action(state: QueryState, action: Any, runtime: Any, *, strategy: str) -> dict[str, Any]:
    value = _state(state)
    if value["steps"] >= 24 or time.time() >= value["deadline"]:
        return _safe_end(value, "tool_budget_exhausted")
    value["steps"] += 1
    value["strategy_steps"] = {**value["strategy_steps"], strategy: value["strategy_steps"].get(strategy, 0) + 1}
    context = tool_context({**state, "tool_state": value})
    update: dict[str, Any] = {"tool_state": value, "next_node": _next(strategy)}
    observation: dict[str, Any] = {}
    try:
        if isinstance(action, DiscoverTools):
            definitions = await runtime.discover(action.query, context, limit=5)
            loaded = {item["tool_id"]: item for item in value["loaded"]}
            for definition in definitions:
                loaded[definition.tool_id] = definition.model_dump(mode="json")
            value["loaded"] = list(loaded.values())[-12:]
            observation = {"kind": "discovery", "tool_ids": [d.tool_id for d in definitions]}
        elif isinstance(action, CallTool):
            loaded = {item["tool_id"]: ToolDefinition.model_validate(item) for item in value["loaded"]}
            if action.tool_id not in loaded:
                raise ToolError("tool_not_loaded")
            result = await runtime.call(action.tool_id, action.arguments, context,
                                        call_id=f"tool:{state['run_id']}:{value['steps']}")
            value["results"] = [*value["results"], result.model_dump(mode="json")]
            observation = _observation(result)
            if result.status == "success" and result.source_kind == "document":
                update.update(_document_update(state, result, action.arguments))
                value["document_queries"] = [*value.get("document_queries", []), str(action.arguments.get("query", ""))][-8:]
        elif isinstance(action, EscalateTools):
            if strategy != "research":
                return _escalate(state, value)
            observation = {"kind": "error", "error_code": "already_researching"}
        elif isinstance(action, ToolReply):
            if action.action == "clarify":
                return _safe_end(value, "clarification_required", text=action.text, status="clarify")
            if action.action == "cannot_answer":
                return _safe_end(value, "tool_information_unavailable")
            if value["results"] or has_external_requirement(state) or (state.get("route_assessment") or {}).get("required_sources") == ["knowledge_base"]:
                return finish_tools({**state, "tool_state": value}, strategy=strategy)
            # Only an ordinary conversation may publish an uncited model reply.
            if strategy == "chat":
                return {"tool_state": value, "next_node": "end", "termination_reason": None,
                        "answer": {"route": "chat", "segments": [{"kind": "content", "text": action.text, "evidence_ids": []}]}}
            return _safe_end(value, "verified_evidence_missing")
        elif isinstance(action, FinishTools):
            return finish_tools({**state, "tool_state": value}, strategy=strategy,
                                result_ids=action.result_ids, max_items=action.max_items)
    except asyncio.CancelledError:
        raise
    except ToolError as error:
        observation = {"kind": "error", "error_code": error.code, "retryable": error.retryable}
    except (TypeError, ValueError):
        observation = {"kind": "error", "error_code": "tool_result_invalid", "retryable": False}
    value["observations"] = [*value["observations"], observation][-16:]
    return update


def _observation(result: ToolResult) -> dict[str, Any]:
    if result.status != "success":
        return {"kind": "tool", "tool_id": result.tool_id, "call_id": result.call_id,
                "error_code": result.error_code, "retryable": result.retryable}
    data = result.data
    if result.source_kind == "document":
        batch = EvidenceBatch.model_validate(data["batch"])
        body: Any = [{"parent_id": p.parent_id, "content": p.content[:2000]} for p in batch.parents[:6]]
    else:
        body = data
    encoded = json.dumps(body, ensure_ascii=False)
    return {"kind": "tool", "tool_id": result.tool_id, "call_id": result.call_id,
            "trust": "untrusted_data", "content": encoded[:8000], "truncated": len(encoded) > 8000}


def _document_update(state: QueryState, result: ToolResult, arguments: Mapping[str, Any]) -> dict[str, Any]:
    batch = EvidenceBatch.model_validate(result.data["batch"])
    target_id = f"query:{state['run_id']}"
    batch = batch.model_copy(update={"target_ids": tuple(dict.fromkeys((*batch.target_ids, target_id)))})
    batches = [EvidenceBatch.model_validate(x) for x in state.get("retrieval_batches", [])]
    batches.append(batch)
    packed = EvidenceBuilder().build(batches, [EvidenceCoverageTarget(target_id=target_id,
        description=str(arguments.get("query") or question_from_state(state)))], scope_from_state(state), snapshot_from_state(state))
    return {"retrieval_batches": [b.model_dump(mode="json") for b in batches],
            "packed_context": packed.model_dump(mode="json"), "evidence": [p.model_dump(mode="json") for p in packed.items]}


def finish_tools(state: QueryState, *, strategy: str, result_ids: tuple[str, ...] = (), max_items: int = 5) -> dict[str, Any]:
    from agentic_rag.query.tool_answers import build_tool_answer
    value = _state(state)
    results = tool_results(state)
    if result_ids:
        if not set(result_ids).issubset({result.call_id for result in results}):
            return _safe_end(value, "unknown_tool_result")
        # Document evidence follows its own packed manifest, even if the model
        # selects only the final map output for presentation.
        results = [r for r in results if r.call_id in result_ids or r.source_kind == "document"]
    value["selected_result_ids"] = [r.call_id for r in results]
    value["max_items"] = max_items
    raw_required = (state.get("route_assessment") or {}).get("required_sources", [])
    required = set(raw_required) if isinstance(raw_required, list) else set()
    external = [r for r in results if r.status == "success" and r.source_kind == "external"]
    if required & {"external_lookup", "external_realtime"}:
        # Transport success alone is not usable factual evidence. Require the
        # domain projection before allowing document-only generation to finish.
        external_answer = build_tool_answer(external, route=strategy, max_cards=max_items)
        if not external_answer or external_answer.get("tool_audited") is not True:
            return _safe_end(value, "external_evidence_missing")
    if "knowledge_base" in required and not state.get("evidence"):
        return _safe_end(value, "document_evidence_missing")
    if state.get("evidence"):
        value["finalizing"] = True
        return {"tool_state": value, "next_node": "generate", "response_mode": None}
    answer = build_tool_answer(results, route=strategy, max_cards=max_items)
    if answer is None:
        return _safe_end(value, "verified_tool_result_missing")
    return {"tool_state": value, "answer": answer, "next_node": "end", "response_mode": None,
            "termination_reason": None}


def document_question(state: QueryState) -> str | None:
    value = state.get("tool_state") or {}
    if not value.get("finalizing") or not has_external_requirement(state):
        return None
    raw_queries = value.get("document_queries", [])
    queries = raw_queries if isinstance(raw_queries, list) else []
    return "请仅基于文档回答以下资料查询，外部地图结果由系统另外展示：\n" + "\n".join(str(q) for q in queries) if queries else None
