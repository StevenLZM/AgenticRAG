"""Safe structured logging and durable Agent Event emission.

Observability data is informational only.  These adapters never inspect or
modify runtime state and intentionally retain only bounded, non-sensitive
metadata.
"""

from __future__ import annotations

import re
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentic_rag.persistence.artifacts import ArtifactStore
from agentic_rag.persistence.repositories import AgentEvent, EventRepository


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
    }
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
    {"dense", "bm25", "reranker", "memory", "llm", "elasticsearch", "redis", "artifact_store"}
)


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
