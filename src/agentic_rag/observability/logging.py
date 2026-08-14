"""Safe structured logging and durable Agent Event emission.

Observability data is informational only.  These adapters never inspect or
modify runtime state and intentionally retain only bounded, non-sensitive
metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from agentic_rag.persistence.artifacts import ArtifactStore
from agentic_rag.persistence.repositories import AgentEvent, EventRepository


logger = logging.getLogger(__name__)


_SENSITIVE_VALUE = re.compile(
    r"(?:\b(?:api[_ -]?key|authorization|bearer|password|secret)\b|\bsk-[A-Za-z0-9_-]+)",
    flags=re.IGNORECASE,
)
_SAFE_SUMMARIES = frozenset(
    {
        "completed",
        "started",
        "cancelled",
        "failed",
        "sufficient",
        "insufficient",
        "clarify",
        "refuse",
        "progress update",
        "degraded",
        "refused",
        "dlq",
        "model_completed",
        "queue_waited",
        "retrieval_completed",
        "citation_validated",
    }
)
_EVENT_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EVENT_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_NODE_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_NUMERIC_ATTRIBUTES = frozenset(
    {
        "queue_wait_seconds",
        "run_latency_seconds",
        "node_latency_seconds",
        "retrieval_rounds",
        "candidate_count",
        "concurrency_active",
        "concurrency_limit",
        "concurrent_saturation",
        "input_tokens",
        "output_tokens",
        "estimated_cost",
        "cited_claim_count",
        "claim_count",
        "repair_count",
        "attempts",
        "latency_ms",
    }
)


@dataclass(slots=True)
class _EmissionScope:
    emitter: "AgentEventEmitter"
    run_id: str
    operation: str
    user_id: str | None
    sequence: int = 0


_EMISSION_SCOPE: ContextVar[_EmissionScope | None] = ContextVar(
    "agentic_rag_event_emission_scope", default=None
)
_TERMINATION_REASONS = frozenset(
    {
        "completed",
        "clarify",
        "refuse",
        "cannot_answer",
        "research_action_invalid",
        "audit_failed",
        "research_round_limit",
        "cancelled",
        "failed",
    }
)
_DEGRADED_COMPONENTS = frozenset(
    {
        "dense",
        "bm25",
        "reranker",
        "memory",
        "mem0",
        "llm",
        "elasticsearch",
        "redis",
        "artifact_store",
        "retrieval",
        "router",
        "generation",
        "audit",
        "citation",
        "outbox",
        "query_worker",
        "worker",
        "mysql",
        "checkpoint",
        "unknown",
    }
)
_DEGRADATION_REASONS = frozenset(
    {
        "lane_failure",
        "lane_timeout",
        "retrieval_unavailable",
        "reranker_unavailable",
        "memory_unavailable",
        "router_unavailable",
        "router_schema_invalid",
        "model_unavailable",
        "model_schema_invalid",
        "generation_unavailable",
        "audit_failed",
        "authorization_unavailable",
        "provider_outage",
        "circuit_open",
        "outbox_retry",
        "lease_lost",
        "cancelled",
        "worker_timeout",
        "worker_dlq",
        "invalid_input",
        "unknown",
    }
)
_DEGRADATION_OUTCOMES = frozenset({"degraded", "refused", "dlq"})
_SAFE_CONTEXT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return JSON-compatible telemetry attributes with unsafe fields removed.

    Drop rather than mask sensitive values.  This prevents the sensitive key
    itself from becoming a searchable durable field and makes accidental raw
    Tool or prompt capture fail closed.
    """
    result: dict[str, Any] = {}
    if not isinstance(attributes, Mapping):
        return result
    for key, value in attributes.items():
        if key in _NUMERIC_ATTRIBUTES and _safe_number(value):
            result[key] = value
        elif key == "attempt" and _safe_number(value) and int(value) == value:
            result[key] = int(value)
        elif key == "retryable" and type(value) is bool:
            result[key] = value
        elif key == "component" and value in _DEGRADED_COMPONENTS:
            result[key] = value
        elif key == "reason" and value in _DEGRADATION_REASONS:
            result[key] = value
        elif key == "outcome" and value in _DEGRADATION_OUTCOMES:
            result[key] = value
        elif key == "termination_reason" and value in _TERMINATION_REASONS:
            result[key] = value
        elif key == "rating" and value in {"up", "down"}:
            result[key] = value
        elif key == "degraded_components" and _safe_components(value):
            result[key] = list(value)
    return result


def sanitize_summary(summary: str) -> str:
    """Keep a bounded operational summary without credential-like content."""
    candidate = summary.strip()[:1_000]
    if _SENSITIVE_VALUE.search(candidate):
        return "telemetry event"
    return candidate if candidate.casefold() in _SAFE_SUMMARIES else "telemetry event"


def structured_log_record(
    event: str, *, attributes: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON log record using the same redaction boundary as events."""
    return {
        "event": sanitize_summary(event),
        "attributes": sanitize_attributes(attributes),
    }


class AgentEventEmitter:
    """Adapt safe telemetry fields to the existing durable Agent Event log."""

    def __init__(
        self,
        repository: EventRepository,
        artifacts: ArtifactStore | None,
        *,
        runtime_config_snapshot_id: str,
    ) -> None:
        if not runtime_config_snapshot_id.strip():
            raise ValueError("runtime_config_snapshot_id must not be blank")
        self._repository = repository
        self._artifacts = artifacts
        self._snapshot_id = runtime_config_snapshot_id

    @property
    def runtime_config_snapshot_id(self) -> str:
        """Expose the immutable baseline identity for safe producer injection."""
        return self._snapshot_id

    async def emit(
        self,
        *,
        run_id: str,
        user_id: str,
        event_type: str,
        attributes: Mapping[str, Any] | None = None,
        summary: str = "",
        node_name: str | None = None,
        trace_id: str | None = None,
        event_key: str | None = None,
        created_at: datetime | None = None,
    ) -> int:
        """Persist one sanitized Event and return its existing repository cursor.

        Attribute artifacts are optional so callers can retain the existing
        compact Event row contract.  Artifact writes occur only after redaction.
        """
        if not run_id.strip() or not user_id.strip() or not event_type.strip():
            raise ValueError("run_id, user_id, and event_type must not be blank")
        if _EVENT_TYPE.fullmatch(event_type) is None:
            raise ValueError("event_type must be a safe uppercase identifier")
        if node_name is not None and _NODE_NAME.fullmatch(node_name) is None:
            raise ValueError("node_name must be a safe identifier")
        if summary and sanitize_summary(summary) == "telemetry event":
            raise ValueError("summary must use the safe telemetry summary allowlist")
        safe_attributes = sanitize_attributes(attributes)
        key = event_key or uuid4().hex
        if _EVENT_KEY.fullmatch(key) is None:
            raise ValueError("event_key must use only letters, digits, underscores, or hyphens")
        payload_ref: str | None = None
        if safe_attributes and self._artifacts is not None:
            payload = {"attributes": safe_attributes}
            digest = _artifact_digest(user_id, run_id, key, payload)
            path = f"observability/events/{digest}.json"
            try:
                ref = self._artifacts.describe(f"artifact://{path}")
                if self._artifacts.read_json(ref) != payload:
                    raise RuntimeError("observability artifact hash collision")
            except FileNotFoundError:
                ref = self._artifacts.put_json(path, payload)
            payload_ref = ref.uri
        event = AgentEvent(
            event_key=key,
            trace_id=trace_id or run_id,
            run_id=run_id,
            user_id=user_id,
            event_type=event_type.upper(),
            summary=sanitize_summary(summary),
            runtime_config_snapshot_id=self._snapshot_id,
            node_name=node_name[:128] if node_name else None,
            payload_ref=payload_ref,
            # Let SqlAlchemyEventRepository assign the first-write timestamp so
            # an otherwise identical replay cannot raise EventKeyConflict.
            created_at=created_at,
        )
        return await self._repository.append(event)


@asynccontextmanager
async def event_emission_scope(
    emitter: AgentEventEmitter,
    run_id: str,
    operation: str,
    *,
    user_id: str | None = None,
) -> AsyncIterator[None]:
    """Bind one safe Event emitter to the current async ModelGateway task."""
    if not run_id.strip() or _NODE_NAME.fullmatch(operation) is None:
        raise ValueError("run_id and operation must be safe non-blank identifiers")
    if user_id is not None and not user_id.strip():
        raise ValueError("user_id must be non-blank when supplied")
    parent = _EMISSION_SCOPE.get()
    if (
        parent is not None
        and parent.emitter is emitter
        and parent.run_id == run_id
        and (parent.user_id == user_id or user_id is None)
    ):
        composed = f"{parent.operation}.{operation}"
        if _NODE_NAME.fullmatch(composed) is not None:
            operation = composed
        if user_id is None:
            user_id = parent.user_id
    token: Token[_EmissionScope | None] = _EMISSION_SCOPE.set(
        _EmissionScope(emitter, run_id, operation, user_id)
    )
    try:
        yield
    finally:
        _EMISSION_SCOPE.reset(token)


async def emit_model_usage(
    *, input_tokens: int, output_tokens: int, attempts: int, latency_ms: int
) -> None:
    """Write provider-derived model counters when an approved task scope exists."""
    scope = _EMISSION_SCOPE.get()
    if scope is None or scope.user_id is None:
        return
    scope.sequence += 1
    try:
        await scope.emitter.emit(
            run_id=scope.run_id,
            user_id=scope.user_id,
            event_type="LLM_COMPLETED",
            node_name=scope.operation,
            summary="model_completed",
            event_key=stable_event_key(
                scope.run_id, scope.operation, str(scope.sequence), "LLM_COMPLETED"
            ),
            attributes={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "attempts": attempts,
                "latency_ms": latency_ms,
            },
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return


async def emit_degradation(
    *,
    component: str,
    reason: str,
    run_id: str | None,
    snapshot_id: str,
    attempt: int,
    retryable: bool,
    outcome: Literal["degraded", "refused", "dlq"],
    event_type: str | None = None,
) -> None:
    """Emit one bounded signal for a fallback, refusal, or circuit outcome.

    This helper is best-effort: telemetry failure must never turn a safe
    business fallback into a failed query.  Both the logger and durable event
    contain only allowlisted enums and non-sensitive server identifiers.
    When called inside :func:`event_emission_scope`, the event is persisted only
    if the scope's run and snapshot match the supplied values.
    """
    safe_component = component if component in _DEGRADED_COMPONENTS else "unknown"
    safe_reason = reason if reason in _DEGRADATION_REASONS else "unknown"
    safe_outcome = outcome if outcome in _DEGRADATION_OUTCOMES else "degraded"
    safe_attempt = attempt if type(attempt) is int and attempt >= 0 else 0
    safe_retryable = retryable if type(retryable) is bool else False
    raw_snapshot = snapshot_id.strip() if isinstance(snapshot_id, str) else ""
    safe_snapshot = raw_snapshot if _SAFE_CONTEXT_ID.fullmatch(raw_snapshot) else "unknown"
    scope = _EMISSION_SCOPE.get()
    safe_run = (
        run_id.strip()
        if isinstance(run_id, str) and _SAFE_CONTEXT_ID.fullmatch(run_id.strip())
        else None
    )
    if scope is not None:
        if not raw_snapshot:
            safe_snapshot = scope.emitter.runtime_config_snapshot_id
        if safe_run is None:
            safe_run = scope.run_id
        elif safe_run != scope.run_id:
            safe_run = None
        if safe_snapshot != scope.emitter.runtime_config_snapshot_id:
            safe_snapshot = "unknown"
    resolved_event_type = event_type or _degradation_event_type(
        safe_component, safe_reason, safe_outcome
    )
    if _EVENT_TYPE.fullmatch(resolved_event_type) is None:
        resolved_event_type = "COMPONENT_DEGRADED"
    logger.warning(
        "degradation component=%s reason=%s outcome=%s retryable=%s attempt=%s run_id=%s snapshot_id=%s",
        safe_component,
        safe_reason,
        safe_outcome,
        safe_retryable,
        safe_attempt,
        safe_run or "none",
        safe_snapshot,
    )
    if scope is None or safe_run is None or safe_snapshot == "unknown":
        return
    if scope.emitter.runtime_config_snapshot_id != safe_snapshot:
        return
    try:
        await scope.emitter.emit(
            run_id=safe_run,
            user_id=scope.user_id or "unknown",
            event_type=resolved_event_type,
            summary=safe_outcome,
            event_key=stable_event_key(
                safe_run,
                "degradation",
                resolved_event_type,
                safe_component,
                safe_reason,
                str(safe_attempt),
                safe_outcome,
            ),
            attributes={
                "attempt": safe_attempt,
                "component": safe_component,
                "reason": safe_reason,
                "retryable": safe_retryable,
                "outcome": safe_outcome,
            },
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        # The warning above is the fallback when the durable repository/artifact
        # boundary itself is unavailable.
        return


def _degradation_event_type(component: str, reason: str, outcome: str) -> str:
    if reason == "circuit_open":
        return "CIRCUIT_OPEN"
    if outcome == "dlq":
        return "WORKER_DLQ"
    if component in {"dense", "bm25", "retrieval"}:
        return "RETRIEVAL_DEGRADED"
    if component == "outbox":
        return "OUTBOX_RETRY"
    if outcome == "refused":
        return "COMPONENT_REFUSED"
    return "COMPONENT_DEGRADED"


def stable_event_key(*parts: str) -> str:
    """Return a deterministic key from non-sensitive, server-owned semantics."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:64]


def _safe_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _safe_components(value: object) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(component, str) and component in _DEGRADED_COMPONENTS
        for component in value
    )


def _artifact_digest(
    user_id: str, run_id: str, event_key: str, payload: Mapping[str, Any]
) -> str:
    """Address artifact content without exposing caller-controlled path segments."""
    encoded = json.dumps(
        {
            "user_id": user_id,
            "run_id": run_id,
            "event_key": event_key,
            "payload": payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
