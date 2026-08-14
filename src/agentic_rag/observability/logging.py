"""Safe structured logging and durable Agent Event emission.

Observability data is informational only.  These adapters never inspect or
modify runtime state and intentionally retain only bounded, non-sensitive
metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import uuid4

from agentic_rag.persistence.artifacts import ArtifactStore
from agentic_rag.persistence.repositories import AgentEvent, EventRepository


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "password",
    "prompt",
    "reasoning",
    "secret",
    "tool_args",
    "tool_payload",
    "tool_result",
)
_SENSITIVE_VALUE = re.compile(
    r"(?:\b(?:api[_ -]?key|authorization|bearer|password|secret)\b|\bsk-[A-Za-z0-9_-]+)",
    flags=re.IGNORECASE,
)
_MAX_ATTRIBUTE_DEPTH = 8
_MAX_ATTRIBUTE_STRING_LENGTH = 1_000


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return JSON-compatible telemetry attributes with unsafe fields removed.

    Drop rather than mask sensitive values.  This prevents the sensitive key
    itself from becoming a searchable durable field and makes accidental raw
    Tool or prompt capture fail closed.
    """
    if not isinstance(attributes, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_key, value in attributes.items():
        if not isinstance(raw_key, str) or _is_sensitive_key(raw_key):
            continue
        sanitized = _sanitize_value(value, depth=0)
        if sanitized is not _OMIT:
            result[raw_key] = sanitized
    return result


def sanitize_summary(summary: str) -> str:
    """Keep a bounded operational summary without credential-like content."""
    candidate = summary.strip()[:1_000]
    return "redacted telemetry event" if _SENSITIVE_VALUE.search(candidate) else candidate


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
        safe_attributes = sanitize_attributes(attributes)
        key = event_key or uuid4().hex
        if len(key) > 64:
            raise ValueError("event_key must be at most 64 characters")
        payload_ref: str | None = None
        if safe_attributes and self._artifacts is not None:
            ref = self._artifacts.put_json(
                f"observability/events/{key}.json",
                {"attributes": safe_attributes},
            )
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


class _Omit:
    pass


_OMIT = _Omit()


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in {"token", "access_token", "auth_token"} or any(
        part in normalized for part in _SENSITIVE_KEY_PARTS
    )


def _sanitize_value(value: Any, *, depth: int) -> Any:
    if depth >= _MAX_ATTRIBUTE_DEPTH:
        return _OMIT
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if _SENSITIVE_VALUE.search(value):
            return _OMIT
        return value[:_MAX_ATTRIBUTE_STRING_LENGTH]
    if isinstance(value, Mapping):
        return sanitize_attributes(value)
    if isinstance(value, (list, tuple)):
        items = [_sanitize_value(item, depth=depth + 1) for item in value]
        return [item for item in items if item is not _OMIT]
    return _OMIT
