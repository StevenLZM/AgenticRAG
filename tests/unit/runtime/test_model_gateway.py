"""Behavioral tests for the single-owner model invocation boundary."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from agentic_rag.models.schemas import RouteDecision
from agentic_rag.runtime.model_gateway import (
    ModelCall,
    ModelGateway,
    StructuredOutputValidationError,
    load_prompt,
    prompt_hashes,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="graph-v1",
    prompt_version="prompt-v1",
    main_model_id="main-model",
    light_model_id="light-model",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="reranker-v1",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)
ROUTE_CALL = ModelCall(
    messages=({"role": "user", "content": "What is the route?"},),
    model_role="light",
    snapshot=SNAPSHOT,
)


@dataclass
class _Usage:
    input_tokens: int = 7
    output_tokens: int = 3


@dataclass
class _Response:
    output_text: str
    model: str = "provider-resolved-model"
    usage: _Usage = field(default_factory=_Usage)


class FakeResponsesApi:
    def __init__(self, values: list[object]) -> None:
        self.values = values
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return _Response(output_text=str(value))


class FakeClient:
    def __init__(self, values: list[object]) -> None:
        self.responses = FakeResponsesApi(values)


@dataclass
class UsageEmitter:
    calls: list[dict[str, object]] = field(default_factory=list)

    async def emit(self, **kwargs: object) -> int:
        self.calls.append(dict(kwargs))
        return len(self.calls)


class FakeChatCompletionsApi(FakeResponsesApi):
    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return {"model": "chat-model", "choices": [{"message": {"content": value}}]}


class FakeChatClient:
    def __init__(self, values: list[object]) -> None:
        self.chat = type(
            "Chat", (), {"completions": FakeChatCompletionsApi(values)}
        )()


class DualProtocolClient(FakeChatClient):
    def __init__(self, chat_values: list[object], response_values: list[object]) -> None:
        super().__init__(chat_values)
        self.responses = FakeResponsesApi(response_values)


@dataclass
class DiagnosticEmitter:
    runtime_config_snapshot_id: str
    calls: list[dict[str, object]] = field(default_factory=list)

    async def emit(self, **kwargs: object) -> int:
        self.calls.append(dict(kwargs))
        return len(self.calls)


class APIConnectionError(Exception):
    """A transport-shaped SDK error that does not inherit ConnectionError."""


async def test_structured_call_repairs_invalid_schema_without_returning_partial() -> None:
    client = FakeClient(
        [
            '{"route":"unknown"}',
            '{"route":"fast_rag","normalized_query":"q","reason_code":"simple"}',
        ]
    )

    result = await ModelGateway(client, sleep=lambda _: _no_sleep()).complete_structured(
        ROUTE_CALL, RouteDecision
    )

    assert result.value == RouteDecision(
        route="fast_rag", normalized_query="q", reason_code="simple"
    )
    assert result.attempts == 2
    assert (result.input_tokens, result.output_tokens) == (14, 6)
    assert len(client.responses.calls) == 2
    assert "schema validation error" in str(client.responses.calls[1]["input"]).lower()


async def test_transient_failure_is_retried_by_gateway_once_per_attempt() -> None:
    client = FakeClient([asyncio.TimeoutError(), "hello"])

    result = await ModelGateway(client, sleep=lambda _: _no_sleep()).complete(ROUTE_CALL)

    assert result.value == "hello"
    assert result.attempts == 2
    assert result.requested_model == "light-model"
    assert result.actual_model == "provider-resolved-model"
    assert (result.input_tokens, result.output_tokens) == (7, 3)


async def test_gateway_emits_real_usage_only_inside_task_local_event_scope() -> None:
    """The gateway emits provider-derived counts without receiving prompt content."""
    from agentic_rag.observability.logging import event_emission_scope

    emitter = UsageEmitter()
    async with event_emission_scope(emitter, "run-1", "llm", user_id="user-1"):
        await ModelGateway(FakeClient(["hello"]), sleep=lambda _: _no_sleep()).complete(ROUTE_CALL)

    assert len(emitter.calls) == 1
    event = emitter.calls[0]
    assert event["event_type"] == "LLM_COMPLETED"
    assert event["attributes"] == {"input_tokens": 7, "output_tokens": 3, "attempts": 1, "latency_ms": event["attributes"]["latency_ms"]}
    assert event["event_key"]


async def test_sdk_connection_error_is_retried_without_retrying_value_errors() -> None:
    client = FakeClient([APIConnectionError("temporary connection"), "hello"])

    result = await ModelGateway(client, sleep=lambda _: _no_sleep()).complete(ROUTE_CALL)

    assert result.value == "hello"
    assert result.attempts == 2


async def test_non_transient_failure_is_not_retried() -> None:
    client = FakeClient([ValueError("bad request")])

    with pytest.raises(ValueError, match="bad request"):
        await ModelGateway(client, sleep=lambda _: _no_sleep()).complete(ROUTE_CALL)

    assert len(client.responses.calls) == 1


async def test_gateway_supports_an_injected_chat_completions_client() -> None:
    client = FakeChatClient(["chat response"])

    result = await ModelGateway(client, sleep=lambda _: _no_sleep()).complete(ROUTE_CALL)

    assert result.value == "chat response"
    assert result.actual_model == "chat-model"


@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_gateway_explicit_protocol_selects_requested_api(protocol: str) -> None:
    client = DualProtocolClient(
        ["chat response"],
        ["responses response"],
    )
    call = ROUTE_CALL.model_copy(update={"protocol": protocol})

    result = await ModelGateway(client, sleep=lambda _: _no_sleep()).complete(call)

    expected = "chat response" if protocol == "chat" else "responses response"
    assert result.value == expected
    assert len(client.chat.completions.calls) == (1 if protocol == "chat" else 0)
    assert len(client.responses.calls) == (1 if protocol == "responses" else 0)


async def test_gateway_auto_prefers_chat_when_both_protocols_are_available() -> None:
    client = DualProtocolClient(["chat response"], ["responses response"])

    result = await ModelGateway(client, sleep=lambda _: _no_sleep()).complete(ROUTE_CALL)

    assert result.value == "chat response"
    assert len(client.chat.completions.calls) == 1
    assert len(client.responses.calls) == 0


async def test_structured_chat_call_requests_json_object() -> None:
    client = DualProtocolClient(
        ['{"route":"fast_rag","normalized_query":"q","reason_code":"simple"}'],
        ["unused"],
    )
    call = ROUTE_CALL.model_copy(update={"protocol": "chat"})

    await ModelGateway(client, sleep=lambda _: _no_sleep()).complete_structured(
        call, RouteDecision
    )

    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


async def test_structured_call_accepts_a_single_json_markdown_fence() -> None:
    client = FakeClient(
        [
            '```json\n{"route":"fast_rag","normalized_query":"q","reason_code":"simple"}\n```'
        ]
    )

    result = await ModelGateway(client, sleep=lambda _: _no_sleep()).complete_structured(
        ROUTE_CALL, RouteDecision
    )

    assert result.value.route == "fast_rag"
    assert result.attempts == 1


async def test_schema_exhaustion_diagnostic_contains_only_safe_metadata() -> None:
    raw_output = "private prompt and hidden reasoning"
    client = FakeClient([raw_output, raw_output])
    emitter = DiagnosticEmitter(SNAPSHOT.snapshot_id)

    from agentic_rag.observability.logging import event_emission_scope

    with pytest.raises(StructuredOutputValidationError):
        async with event_emission_scope(emitter, "run-1", "answer", user_id="user-1"):
            await ModelGateway(client, sleep=lambda _: _no_sleep()).complete_structured(
                ROUTE_CALL, RouteDecision
            )

    diagnostic = next(item for item in emitter.calls if item["event_type"] == "MODEL_REPAIR_EXHAUSTED")
    assert diagnostic["attributes"]["schema_name"] == "RouteDecision"
    assert diagnostic["attributes"]["output_length"] == len(raw_output)
    assert diagnostic["attributes"]["output_sha256"]
    assert raw_output not in json.dumps(diagnostic, ensure_ascii=False)


@pytest.mark.parametrize(
    "name",
    [
        "router_v1",
        "research_agent_v1",
        "evidence_grader_v1",
        "generator_v1",
        "faithfulness_v1",
        "context_compactor_v1",
        "memory_extractor_v1",
    ],
)
def test_versioned_prompt_loader_returns_stable_content_hash(name: str) -> None:
    prompt = load_prompt(name)

    assert prompt.version == "v1"
    assert len(prompt.content_hash) == 64
    assert "# ROLE" in prompt.content
    assert "# FAIL-CLOSED RULES" in prompt.content


def test_all_versioned_prompt_hashes_are_captured_by_an_immutable_snapshot() -> None:
    names = (
        "router_v1",
        "research_agent_v1",
        "evidence_grader_v1",
        "generator_v1",
        "faithfulness_v1",
        "context_compactor_v1",
        "memory_extractor_v1",
    )
    hashes = prompt_hashes(names)
    snapshot = RuntimeConfigSnapshot(
        **SNAPSHOT.model_dump(exclude={"prompt_hashes"}), prompt_hashes=hashes
    )

    assert snapshot.prompt_hash_map == hashes
    assert snapshot.model_dump()["prompt_hashes"] == tuple(sorted(hashes.items()))
    with pytest.raises(TypeError):
        snapshot.prompt_hashes[0] = ("router_v1", "0" * 64)


async def _no_sleep() -> None:
    return None
