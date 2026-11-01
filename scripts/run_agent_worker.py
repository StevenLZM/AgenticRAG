#!/usr/bin/env python3
"""Run one supervisor-v1 role in an independent process."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import signal
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.bootstrap import build_container  # noqa: E402
from agentic_rag.config import get_settings  # noqa: E402
from agentic_rag.orchestration.composition import build_executor  # noqa: E402
from agentic_rag.orchestration.service import CoordinationService, check_schema  # noqa: E402
from agentic_rag.orchestration.store import CoordinationStore  # noqa: E402
from agentic_rag.orchestration.worker import AgentWorker, ROLE_STAGES  # noqa: E402


async def run(role: str) -> None:
    settings = get_settings()
    container, clients = build_container(settings), []
    stop = asyncio.Event()
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{role}"
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await check_schema(container.mysql_engine)
        from agentic_rag.runtime.context_composition import validate_context_role
        if role == "coordinator":
            await validate_context_role(container, role)
            service = CoordinationService(CoordinationStore(container.repositories.session_factory, max_active_runs=settings.max_concurrent_query_runs, context_sources=container.context_sources, query_compiler=container.query_compiler), container.broker, worker_id=worker_id)
            await service.run_forever(stop)
        else:
            executor, clients = await build_executor(container, settings, role)
            await validate_context_role(container, role)
            worker = AgentWorker(executor.store, executor, worker_id=worker_id, role=role, broker=container.broker)
            await worker.run_forever(stop)
    finally:
        for client in clients:
            await client.close()
        await container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["coordinator", *ROLE_STAGES])
    asyncio.run(run(parser.parse_args().role))
