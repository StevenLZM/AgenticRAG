"""JSON-only state contracts for the QueryGraph entry path.

Query state can be checkpointed or streamed.  It consequently contains only
JSON primitives, lists and dictionaries: clients and compiled graphs stay in
the process-owned dependency objects that close over graph nodes.
"""

from __future__ import annotations

import json
from typing import Any, NotRequired, TypedDict, cast

from agentic_rag.domain.models import UserScope
from agentic_rag.runtime.models import RuntimeConfigSnapshot


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class QueryState(TypedDict):
    """Checkpoint-safe state shared by the Phase 4 query runtime."""

    request: dict[str, JsonValue]
    run_id: str
    scope: dict[str, JsonValue]
    messages: list[dict[str, JsonValue]]
    memory_context: NotRequired[dict[str, JsonValue]]
    route: NotRequired[dict[str, JsonValue]]
    research: NotRequired[dict[str, JsonValue]]
    evidence: NotRequired[list[dict[str, JsonValue]]]
    retrieval_batches: NotRequired[list[dict[str, JsonValue]]]
    packed_context: NotRequired[dict[str, JsonValue]]
    runtime_config_snapshot: dict[str, JsonValue]
    answer: NotRequired[dict[str, JsonValue]]
    audit_results: list[dict[str, JsonValue]]
    revision_count: int
    errors: list[dict[str, JsonValue]]
    termination_reason: str | None
    next_node: NotRequired[str]


class InvalidQueryState(ValueError):
    """Raised when a graph node receives incomplete or non-serializable state."""


def new_query_state(
    *,
    run_id: str,
    question: str,
    scope: UserScope,
    snapshot: RuntimeConfigSnapshot,
    messages: list[dict[str, Any]] | None = None,
) -> QueryState:
    """Create a minimal, serializable state for a server-validated query run."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be blank")
    state: QueryState = {
        "request": {"question": normalized_question},
        "run_id": run_id,
        "scope": cast(dict[str, JsonValue], json_safe(scope.model_dump(mode="json"))),
        "messages": cast(list[dict[str, JsonValue]], json_safe(messages or [])),
        "runtime_config_snapshot": cast(
            dict[str, JsonValue], json_safe(snapshot.model_dump(mode="json"))
        ),
        "audit_results": [],
        "revision_count": 0,
        "errors": [],
        "termination_reason": None,
    }
    return state


def json_safe(value: Any) -> JsonValue:
    """Round-trip through JSON to reject non-checkpointable graph values."""
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise InvalidQueryState("query state must contain JSON-compatible values") from error
    return decoded


def scope_from_state(state: QueryState) -> UserScope:
    """Reconstruct the immutable server-owned scope at a graph boundary."""
    try:
        return UserScope.model_validate(state["scope"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidQueryState("state does not include a valid user scope") from error


def snapshot_from_state(state: QueryState) -> RuntimeConfigSnapshot:
    """Reconstruct the immutable runtime snapshot at a graph boundary."""
    try:
        return RuntimeConfigSnapshot.model_validate(state["runtime_config_snapshot"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidQueryState("state does not include a valid runtime snapshot") from error


def question_from_state(state: QueryState) -> str:
    """Return the normalized server query while rejecting arbitrary structures."""
    request = state.get("request")
    question = request.get("question") if isinstance(request, dict) else None
    if not isinstance(question, str) or not (normalized := question.strip()):
        raise InvalidQueryState("state does not include a valid question")
    return normalized
