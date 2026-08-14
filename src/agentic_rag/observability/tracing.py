"""OpenTelemetry-compatible local span records without an external collector."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue

from agentic_rag.observability.logging import sanitize_attributes


class SpanRecord(BaseModel):
    """The local subset of a completed OpenTelemetry span useful for operations."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    run_id: str
    name: str
    started_at: datetime
    ended_at: datetime | None
    status: Literal["ok", "error", "cancelled"]
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    baseline_label: str
    runtime_config_snapshot_id: str

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Default to JSON mode so event records are safely serializable."""
        kwargs.setdefault("mode", "json")
        return super().model_dump(**kwargs)


_ACTIVE_SPANS: ContextVar[tuple[SpanRecord, ...] | None] = ContextVar(
    "agentic_rag_active_spans", default=None
)


class TraceRecorder:
    """Collect process-local spans while preserving task-local context hierarchy."""

    def __init__(
        self, *, runtime_config_snapshot_id: str, baseline_label: str = "default"
    ) -> None:
        if not runtime_config_snapshot_id.strip():
            raise ValueError("runtime_config_snapshot_id must not be blank")
        if not baseline_label.strip():
            raise ValueError("baseline_label must not be blank")
        self.runtime_config_snapshot_id = runtime_config_snapshot_id
        self.baseline_label = baseline_label
        self.events: list[SpanRecord] = []

    def span(
        self,
        name: str,
        *,
        run_id: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> "LocalSpan":
        """Start an async span; its status is finalized as its context exits."""
        if not name.strip() or not run_id.strip():
            raise ValueError("span name and run_id must not be blank")
        return LocalSpan(self, name, run_id, attributes)


class LocalSpan:
    """Context manager returned by :meth:`TraceRecorder.span`."""

    def __init__(
        self,
        recorder: TraceRecorder,
        name: str,
        run_id: str,
        attributes: Mapping[str, Any] | None,
    ) -> None:
        self._recorder = recorder
        self._name = name
        self._run_id = run_id
        self._attributes = sanitize_attributes(attributes)
        self._record: SpanRecord | None = None
        self._index: int | None = None
        self._root_index: int | None = None
        self._token: Token[tuple[SpanRecord, ...] | None] | None = None

    @property
    def record(self) -> SpanRecord:
        if self._record is None:
            raise RuntimeError("span has not been entered")
        return self._record

    async def __aenter__(self) -> "LocalSpan":
        active = _ACTIVE_SPANS.get()
        if active:
            parent = active[-1]
            trace_id = parent.trace_id
            stack = active
        else:
            implicit_root = SpanRecord(
                trace_id=uuid4().hex,
                span_id=uuid4().hex,
                parent_span_id=None,
                run_id=self._run_id,
                name="run",
                started_at=datetime.now(UTC),
                ended_at=None,
                status="ok",
                baseline_label=self._recorder.baseline_label,
                runtime_config_snapshot_id=self._recorder.runtime_config_snapshot_id,
            )
            parent = implicit_root
            trace_id = implicit_root.trace_id
            stack = (implicit_root,)
            self._root_index = len(self._recorder.events)
            self._recorder.events.append(implicit_root)
        self._record = SpanRecord(
            trace_id=trace_id,
            span_id=uuid4().hex,
            parent_span_id=parent.span_id,
            run_id=self._run_id,
            name=self._name,
            started_at=datetime.now(UTC),
            ended_at=None,
            status="ok",
            attributes=self._attributes,
            baseline_label=self._recorder.baseline_label,
            runtime_config_snapshot_id=self._recorder.runtime_config_snapshot_id,
        )
        self._index = len(self._recorder.events)
        self._recorder.events.append(self._record)
        self._token = _ACTIVE_SPANS.set((*stack, self._record))
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del traceback
        status: Literal["ok", "error", "cancelled"] = "ok"
        if exc_type is not None:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        self._finish(status)
        if self._root_index is not None:
            root = self._recorder.events[self._root_index]
            self._recorder.events[self._root_index] = root.model_copy(
                update={"status": status, "ended_at": datetime.now(UTC)}
            )
        if self._token is not None:
            _ACTIVE_SPANS.reset(self._token)
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        """Add one safe scalar or JSON value to this span."""
        self._update_attributes({key: value})

    def record_usage(
        self, *, input_tokens: int, output_tokens: int, estimated_cost: float
    ) -> None:
        """Record model usage only; it deliberately cannot affect model decisions."""
        if input_tokens < 0 or output_tokens < 0 or estimated_cost < 0:
            raise ValueError("usage values must be non-negative")
        self._update_attributes(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": estimated_cost,
            }
        )

    def _update_attributes(self, attributes: Mapping[str, Any]) -> None:
        record = self.record
        safe = sanitize_attributes(attributes)
        if not safe:
            return
        updated = record.model_copy(update={"attributes": {**record.attributes, **safe}})
        self._replace(updated)

    def _finish(self, status: Literal["ok", "error", "cancelled"]) -> None:
        record = self.record
        self._replace(
            record.model_copy(update={"status": status, "ended_at": datetime.now(UTC)})
        )

    def _replace(self, record: SpanRecord) -> None:
        if self._index is None:
            raise RuntimeError("span has not been entered")
        self._record = record
        self._recorder.events[self._index] = record
