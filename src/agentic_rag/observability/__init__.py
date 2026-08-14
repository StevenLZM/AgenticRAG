"""Local, privacy-preserving observability primitives for Agentic RAG."""

from agentic_rag.observability.logging import AgentEventEmitter, structured_log_record
from agentic_rag.observability.metrics import MetricsProjector, MetricsWindow
from agentic_rag.observability.tracing import SpanRecord, TraceRecorder

__all__ = [
    "AgentEventEmitter",
    "MetricsProjector",
    "MetricsWindow",
    "SpanRecord",
    "TraceRecorder",
    "structured_log_record",
]
