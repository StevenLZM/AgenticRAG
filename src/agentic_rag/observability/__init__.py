"""Local, privacy-preserving observability primitives for Agentic RAG."""

from agentic_rag.observability.logging import (
    AgentEventEmitter,
    emit_degradation,
    emit_model_usage,
    event_emission_scope,
    stable_event_key,
    structured_log_record,
)
from agentic_rag.observability.metrics import MetricsProjector, MetricsWindow
from agentic_rag.observability.tracing import SpanRecord, TraceRecorder

__all__ = [
    "AgentEventEmitter",
    "emit_degradation",
    "emit_model_usage",
    "event_emission_scope",
    "MetricsProjector",
    "MetricsWindow",
    "SpanRecord",
    "TraceRecorder",
    "structured_log_record",
    "stable_event_key",
]
