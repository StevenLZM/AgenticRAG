"""Deterministic concurrency/backpressure acceptance tests."""

from __future__ import annotations

import pytest

from agentic_rag.testing.e2e_harness import BackpressureHarness


pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_backpressure_never_exceeds_configured_limits() -> None:
    harness = BackpressureHarness(run_limit=4, llm_limit=8, reranker_limit=1)

    result = await harness.submit_parallel_queries(12)

    assert result.max_observed.run <= 4
    assert result.max_observed.llm <= 8
    assert result.max_observed.reranker <= 1
    assert result.completed_queries == 12
    assert result.api_liveness is True
    assert result.worker_liveness is True


@pytest.mark.asyncio
async def test_backpressure_records_waiters_without_dropping_work() -> None:
    harness = BackpressureHarness(run_limit=2, llm_limit=3, reranker_limit=1)

    result = await harness.submit_parallel_queries(7)

    assert result.max_observed.run <= 2
    assert result.queue_wait_samples == 7
    assert result.completed_queries == 7
    assert result.failed_queries == 0
