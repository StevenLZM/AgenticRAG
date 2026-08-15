"""Query Worker owns the query Outbox dispatch loop."""

from __future__ import annotations

import asyncio

import pytest

from agentic_rag.runtime.query_worker import QueryWorker


class EmptyBroker:
    async def reclaim(self, *args: object) -> list[object]:
        del args
        return []

    async def consume(self, *args: object) -> list[object]:
        del args
        return []


class StopAfterDispatch:
    def __init__(self, stop: asyncio.Event) -> None:
        self.calls = 0
        self._stop = stop

    async def dispatch_once(self, limit: int = 100) -> int:
        del limit
        self.calls += 1
        self._stop.set()
        return 1


@pytest.mark.asyncio
async def test_query_worker_runs_query_outbox_dispatcher_alongside_consumer() -> None:
    stop = asyncio.Event()
    dispatcher = StopAfterDispatch(stop)

    worker = QueryWorker(
        runs=object(),
        broker=EmptyBroker(),
        graph_factory=lambda **_: object(),
        worker_id="query-worker",
        outbox_dispatcher=dispatcher,
        outbox_interval_seconds=0.001,
    )

    await asyncio.wait_for(worker.run_forever(stop_event=stop), timeout=1.0)

    assert dispatcher.calls >= 1
