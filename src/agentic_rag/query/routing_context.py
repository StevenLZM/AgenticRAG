"""Bounded conversational input, separate from this run's memory writes."""

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from agentic_rag.domain.models import UserScope
from agentic_rag.query.state import QueryState, question_from_state


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
