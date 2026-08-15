"""Contracts for local, privacy-preserving trace and event recording."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agentic_rag.observability.logging import (
    AgentEventEmitter,
    sanitize_attributes,
    sanitize_summary,
    structured_log_record,
)
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.repositories import AgentEvent
from agentic_rag.observability.tracing import TraceRecorder


@pytest.fixture
def trace_recorder() -> TraceRecorder:
    return TraceRecorder(runtime_config_snapshot_id="snapshot-1", baseline_label="v1")


async def test_model_span_records_usage_without_prompt_or_secret(
    trace_recorder: TraceRecorder,
) -> None:
    """LLM usage is useful telemetry; request content and credentials are not."""
    async with trace_recorder.span(
        "llm",
        run_id="r1",
        attributes={
            "api_key": "secret",
            "prompt": "do not persist this request",
            "provider": "local-model",
        },
    ) as span:
        span.record_usage(input_tokens=10, output_tokens=4, estimated_cost=0.01)

    event = trace_recorder.events[-1]
    assert event.parent_span_id
    assert event.attributes["input_tokens"] == 10
    assert event.attributes["output_tokens"] == 4
    assert event.attributes["estimated_cost"] == 0.01
    assert "secret" not in json.dumps(event.model_dump())
    assert "do not persist" not in json.dumps(event.model_dump())


async def test_nested_spans_share_trace_and_preserve_parent_hierarchy(
    trace_recorder: TraceRecorder,
) -> None:
    """Removing context propagation would make node latency attribution impossible."""
    async with trace_recorder.span("graph.node", run_id="run-1"):
        async with trace_recorder.span("retrieval", run_id="run-1") as retrieval_span:
            retrieval_span.set_attribute("candidate_count", 6)

    root_event, graph_event, retrieval_event = trace_recorder.events
    assert root_event.parent_span_id is None
    assert graph_event.parent_span_id == root_event.span_id
    assert graph_event.trace_id == retrieval_event.trace_id
    assert retrieval_event.parent_span_id == graph_event.span_id
    assert retrieval_event.attributes["candidate_count"] == 6
    assert graph_event.status == "ok"


async def test_span_marks_error_without_storing_exception_message(
    trace_recorder: TraceRecorder,
) -> None:
    """Exception status remains observable without leaking a provider response."""
    with pytest.raises(RuntimeError, match="private provider payload"):
        async with trace_recorder.span("tool", run_id="run-1"):
            raise RuntimeError("private provider payload")

    event = trace_recorder.events[-1]
    assert event.status == "error"
    assert "private provider payload" not in json.dumps(event.model_dump())


async def test_trace_context_rejects_cross_recorder_or_cross_run_nesting() -> None:
    """A run must never inherit trace context from a different recorder or run."""
    first = TraceRecorder(runtime_config_snapshot_id="snapshot-1")
    second = TraceRecorder(runtime_config_snapshot_id="snapshot-2")

    async with first.span("queue", run_id="run-1"):
        with pytest.raises(RuntimeError, match="recorder"):
            async with second.span("graph.node", run_id="run-1"):
                pass
        with pytest.raises(RuntimeError, match="run_id"):
            async with first.span("graph.node", run_id="run-2"):
                pass


def test_structured_log_record_removes_sensitive_fields() -> None:
    """The log adapter must use the same privacy boundary as trace attributes."""
    record = structured_log_record(
        "model_completed",
        attributes={"authorization": "Bearer secret", "input_tokens": 8},
    )

    assert record["event"] == "model_completed"
    assert record["attributes"] == {"input_tokens": 8}
    assert "secret" not in json.dumps(record)


def test_privacy_sanitizers_fail_closed_for_raw_prompt_and_tool_fields() -> None:
    """Unknown summaries and all raw user/model/tool channels are never telemetry."""
    attributes = sanitize_attributes(
        {
            "tool_input": {"query": "private"},
            "tool_output": "private result",
            "tool_response": "private response",
            "chain_of_thought": "private reasoning",
            "hidden_reasoning": "private reasoning",
            "messages": [{"content": "private prompt"}],
            "provider_payload": {"response": "private"},
            "input_tokens": 8,
        }
    )

    assert attributes == {"input_tokens": 8}
    assert sanitize_summary("summarize this private user prompt") == "telemetry event"
    assert sanitize_summary("completed") == "completed"


async def test_event_emitter_accepts_only_allowlisted_attributes_and_safe_identifiers() -> None:
    """Unknown input-like telemetry fields must be discarded or rejected before storage."""
    repository = RecordingEventRepository()
    emitter = AgentEventEmitter(repository, None, runtime_config_snapshot_id="snapshot-1")
    attributes = {
        "input_tokens": 8,
        "tool_calls": [{"arguments": "private"}],
        "arguments": "private",
        "result": "private",
        "analysis": "private",
        "thinking": "private",
        "instruction": "private",
        "input": "private",
    }
    assert sanitize_attributes(attributes) == {"input_tokens": 8}
    assert sanitize_attributes({"estimated_cost": math.nan, "input_tokens": math.inf}) == {}
    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="LLM_COMPLETED",
        attributes=attributes,
    )
    assert repository.events[0].payload_ref is None

    with pytest.raises(ValueError, match="event_type"):
        await emitter.emit(
            run_id="run-1", user_id="user-1", event_type="summarize user prompt"
        )
    with pytest.raises(ValueError, match="summary"):
        await emitter.emit(
            run_id="run-1",
            user_id="user-1",
            event_type="LLM_COMPLETED",
            summary="summarize user private prompt",
        )


class RecordingEventRepository:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def append(self, event: AgentEvent) -> int:
        self.events.append(event)
        return len(self.events)

    async def list_after(self, *args: object, **kwargs: object) -> list[AgentEvent]:
        del args, kwargs
        return []


async def test_event_emitter_replay_leaves_timestamp_to_repository() -> None:
    """A caller-supplied event key must replay without a timestamp conflict."""
    repository = RecordingEventRepository()
    emitter = AgentEventEmitter(
        repository,
        None,
        runtime_config_snapshot_id="snapshot-1",
    )

    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="LLM_COMPLETED",
        event_key="replay-key",
    )
    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="LLM_COMPLETED",
        event_key="replay-key",
    )

    assert [event.created_at for event in repository.events] == [None, None]


async def test_emitter_artifact_path_is_immutable_and_not_controlled_by_event_key(
    tmp_path: Path,
) -> None:
    """A conflicting replay must not rewrite the first event's safe artifact."""
    repository = RecordingEventRepository()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    emitter = AgentEventEmitter(
        repository, artifacts, runtime_config_snapshot_id="snapshot-1"
    )
    first_id = await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="LLM_COMPLETED",
        event_key="same-key",
        attributes={"input_tokens": 1},
    )
    first = repository.events[first_id - 1]
    assert first.payload_ref is not None
    first_ref = artifacts.describe(first.payload_ref)

    await emitter.emit(
        run_id="run-1",
        user_id="user-1",
        event_type="LLM_COMPLETED",
        event_key="same-key",
        attributes={"input_tokens": 2},
    )
    second = repository.events[-1]
    assert second.payload_ref is not None
    assert second.payload_ref != first.payload_ref
    assert "same-key" not in first.payload_ref
    assert artifacts.read_json(first_ref) == {"attributes": {"input_tokens": 1}}

    third_id = await emitter.emit(
        run_id="run-1",
        user_id="user-2",
        event_type="LLM_COMPLETED",
        event_key="same-key",
        attributes={"input_tokens": 1},
    )
    assert repository.events[third_id - 1].payload_ref != first.payload_ref

    with pytest.raises(ValueError, match="event_key"):
        await emitter.emit(
            run_id="run-1",
            user_id="user-1",
            event_type="LLM_COMPLETED",
            event_key="../path-alias",
        )
