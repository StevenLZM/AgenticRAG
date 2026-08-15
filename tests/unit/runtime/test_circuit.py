"""Process-owned circuit breaker contracts."""

from __future__ import annotations

import asyncio

import pytest

from agentic_rag.runtime.circuit import CircuitOpenError, CircuitState
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main",
    light_model_id="light",
    embedding_model="embedding",
    embedding_dimensions=1024,
    reranker_version="reranker",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)


class _FailingResponses:
    async def create(self, **_: object) -> object:
        raise asyncio.TimeoutError()


class _FailingClient:
    responses = _FailingResponses()


def test_circuit_opens_after_bounded_failures_and_allows_one_probe() -> None:
    now = [0.0]
    circuit = CircuitState(
        failure_threshold=3,
        reset_timeout_seconds=10.0,
        monotonic=lambda: now[0],
    )

    assert circuit.allow_call() is True
    assert circuit.record_failure() is False
    assert circuit.record_failure() is False
    assert circuit.record_failure() is True
    assert circuit.is_open is True
    assert circuit.allow_call() is False

    now[0] = 10.0
    assert circuit.allow_call() is True
    assert circuit.allow_call() is False
    circuit.record_success()
    assert circuit.allow_call() is True
    assert circuit.is_open is False


@pytest.mark.asyncio
async def test_model_gateway_short_circuits_after_provider_failures() -> None:
    circuit = CircuitState(failure_threshold=1, reset_timeout_seconds=30.0)
    gateway = ModelGateway(_FailingClient(), max_retries=0, circuit=circuit)
    call = ModelCall(
        model_role="light",
        snapshot=SNAPSHOT,
        messages=({"role": "user", "content": "safe"},),
    )

    with pytest.raises(asyncio.TimeoutError):
        await gateway.complete(call)
    with pytest.raises(CircuitOpenError):
        await gateway.complete(call)
