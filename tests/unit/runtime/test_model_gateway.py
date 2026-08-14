"""Behavioral tests for the single-owner model invocation boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from agentic_rag.models.schemas import RouteDecision
from agentic_rag.runtime.model_gateway import (
    ModelCall,
    ModelGateway,
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
