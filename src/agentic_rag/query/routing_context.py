"""Bounded conversational input, separate from this run's memory writes."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.domain.models import UserScope
from agentic_rag.query.state import QueryState, question_from_state, scope_from_state, snapshot_from_state
from agentic_rag.query.evidence_builder import PackedEvidence


class RoutingTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    role: Literal["user", "assistant"]
    content: str
    truncated: bool = False


class RoutingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    user_id: str
    thread_id: str | None
    requested_at: str
    timezone: str = "Asia/Shanghai"
    history: tuple[RoutingTurn, ...] = ()
    history_available: bool = True
    map_references: tuple[dict[str, Any], ...] = Field(default=(), max_length=24)


class RoutingContextUnavailable(RuntimeError):
    pass


class ConversationReader(Protocol):
    async def load(self, scope: UserScope, *, run_id: str, thread_id: str) -> RoutingContext: ...


def bound_history(turns: Sequence[RoutingTurn]) -> tuple[RoutingTurn, ...]:
    remaining = 8000
    kept = []
    for turn in reversed(turns[-6:]):
        if remaining <= 0:
            break
        content = turn.content[-remaining:]
        kept.append(turn.model_copy(update={"content": content, "truncated": len(content) < len(turn.content)}))
        remaining -= len(content)
    return tuple(reversed(kept))


def reasoning_question(state: QueryState) -> str:
    original = question_from_state(state)
    if state.get("routing_owner_run_id") != state["run_id"]:
        return original
    assessment = state.get("route_assessment") or {}
    context = state.get("routing_context") or {}
    return json.dumps({"trust": "untrusted_data", "original_question": original,
                       "resolved_question": assessment.get("normalized_query", original),
                       "history": context.get("history", []),
                       "requested_at": context.get("requested_at")}, ensure_ascii=False)


def fresh_routing_state(state: QueryState) -> dict[str, object]:
    if state.get("routing_owner_run_id") == state["run_id"]:
        return {}
    return {"routing_owner_run_id": state["run_id"], "routing_context": {},
            "route_assessment": {}, "route": {}, "policy_decision": {},
            "response_mode": None, "last_evidence_grade": {}, "executed_path": [],
            "initial_route": "", "memory_context": None, "evidence": [], "retrieval_batches": [],
            "packed_context": PackedEvidence(items=(), manifest={}, rendered_context="", token_count=0,
                index_generation=snapshot_from_state(state).index_generation).model_dump(mode="json"),
            "research": {}, "tool_state": {}, "answer": {}, "errors": [],
            "termination_reason": None, "research_attempt_count": 0, "revision_count": 0,
            "audit_results": [], "next_node": "route", "routing_policy_version": "routing-v2"}


async def prepare_routing_state(state: QueryState, *, capabilities, conversations: ConversationReader | None) -> dict[str, object]:
    scope = scope_from_state(state)
    cached = state.get("routing_context")
    if cached:
        context = RoutingContext.model_validate(cached)
        if context.run_id == state["run_id"] and context.user_id == scope.user_id:
            return {"capabilities": capabilities.model_dump(mode="json")}
    thread = state.get("request", {}).get("thread_id")
    if conversations is not None and isinstance(thread, str) and thread:
        context = await conversations.load(scope, run_id=state["run_id"], thread_id=thread)
    else:
        # Direct graph calls have no durable Run; capture time once in checkpoint.
        context = RoutingContext(run_id=state["run_id"], user_id=scope.user_id,
                                 thread_id=thread if isinstance(thread, str) else None,
                                 requested_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                 history_available=False)
    return {"routing_context": context.model_dump(mode="json"), "capabilities": capabilities.model_dump(mode="json")}
