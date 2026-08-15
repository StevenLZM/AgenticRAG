"""Deterministic concurrency/backpressure acceptance tests."""

from __future__ import annotations

import pytest

from agentic_rag.testing.e2e_harness import BackpressureHarness


pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_backpressure_never_exceeds_configured_limits() -> None:
    api_health_calls = 0
    worker_health_calls = 0

    async def api_health() -> bool:
        nonlocal api_health_calls
        api_health_calls += 1
        return True

    async def worker_health() -> bool:
        nonlocal worker_health_calls
        worker_health_calls += 1
        return True

    harness = BackpressureHarness(
        run_limit=4,
        llm_limit=8,
        reranker_limit=1,
        api_health_probe=api_health,
        worker_health_probe=worker_health,
    )

    result = await harness.submit_parallel_queries(12)

    assert result.max_observed.run <= 4
    assert result.max_observed.llm <= 8
    assert result.max_observed.reranker <= 1
    assert result.max_observed.llm == 8
    assert result.llm_tasks_submitted >= 8
    assert result.completed_queries == 12
    assert result.api_liveness is True
    assert result.worker_liveness is True
    assert api_health_calls == 1
    assert worker_health_calls == 1


@pytest.mark.asyncio
async def test_backpressure_records_waiters_without_dropping_work() -> None:
    harness = BackpressureHarness(
        run_limit=2,
        llm_limit=3,
        reranker_limit=1,
        api_health_probe=lambda: _healthy(),
        worker_health_probe=lambda: _healthy(),
    )

    result = await harness.submit_parallel_queries(7)

    assert result.max_observed.run <= 2
    assert result.queue_wait_samples > 0
    assert result.queue_wait_seconds_total > 0
    assert result.completed_queries == 7
    assert result.failed_queries == 0


@pytest.mark.asyncio
async def test_backpressure_reports_injected_health_failure() -> None:
    harness = BackpressureHarness(
        run_limit=2,
        llm_limit=3,
        reranker_limit=1,
        api_health_probe=lambda: _healthy(),
        worker_health_probe=lambda: _unhealthy(),
    )

    result = await harness.submit_parallel_queries(4)

    assert result.api_liveness is True
    assert result.worker_liveness is False


async def _healthy() -> bool:
    return True


async def _unhealthy() -> bool:
    return False
