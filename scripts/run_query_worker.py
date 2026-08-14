#!/usr/bin/env python3
"""Launch the single durable Query Worker.

Project deployments must inject query graph dependencies explicitly: they own
model and retrieval adapters and must never be constructed from stream input.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agentic_rag.bootstrap import AppContainer, build_container  # noqa: E402
from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.query.graph import QueryGraphDependencies  # noqa: E402
from agentic_rag.runtime.query_worker import (  # noqa: E402
    QueryWorker,
    build_graph_factory,
)
from agentic_rag.runtime.run_manager import TransactionalRunRepository  # noqa: E402


DependenciesFactory = Callable[[AppContainer, Settings], Awaitable[QueryGraphDependencies]]


async def run(settings: Settings, dependencies_factory: DependenciesFactory) -> None:
    """Run the worker with deployment-owned QueryGraph dependencies."""
    container = build_container(settings)
    try:
        dependencies = await dependencies_factory(container, settings)
        async with container.checkpoints.open_query() as checkpointer:
            worker = QueryWorker(
                runs=TransactionalRunRepository(container.repositories.session_factory),
                broker=container.broker,
                graph_factory=build_graph_factory(dependencies, checkpointer),
                worker_id=f"{socket.gethostname()}:{os.getpid()}",
                trace_recorder=dependencies.trace_recorder,
                event_emitter=dependencies.event_emitter,
            )
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for event in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(event, stop.set)
                except NotImplementedError:
                    pass
            await worker.run_forever(stop_event=stop)
    finally:
        await container.close()


def main() -> int:
    raise SystemExit(
        "Query Worker dependencies are deployment-owned. Import run(settings, "
        "dependencies_factory) from this module and inject QueryGraphDependencies."
    )


if __name__ == "__main__":
    main()
