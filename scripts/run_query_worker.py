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
from agentic_rag.config import Settings, get_settings  # noqa: E402
from agentic_rag.persistence.outbox import OutboxDispatcher  # noqa: E402
from agentic_rag.persistence.elasticsearch import (  # noqa: E402
    ElasticsearchChildIndexStore,
)
from agentic_rag.persistence.repositories import (  # noqa: E402
    OutboxRecord,
    SqlAlchemyOutboxRepository,
)
from agentic_rag.query.graph import QueryGraphDependencies  # noqa: E402
from agentic_rag.runtime.concurrency import ConcurrencyManager  # noqa: E402
from agentic_rag.runtime.query_worker import (  # noqa: E402
    QUERY_STREAM,
    QueryWorker,
    build_graph_factory,
)
from agentic_rag.runtime.query_composition import (  # noqa: E402
    build_query_dependencies,
    close_query_dependencies,
)
from agentic_rag.runtime.run_manager import TransactionalRunRepository  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker  # noqa: E402


DependenciesFactory = Callable[[AppContainer, Settings], Awaitable[QueryGraphDependencies]]


async def ensure_active_child_alias(
    container: AppContainer, settings: Settings
) -> None:
    """Repair a missing configured Child alias before consuming Query Outbox."""
    await ElasticsearchChildIndexStore(container.elasticsearch).ensure_active_alias(
        settings.index_generation
    )


class TransactionalQueryOutboxAdapter:
    """Open a short SQL transaction for Query Outbox lifecycle operations."""

    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        *,
        user_id: str | None = None,
        stream_name: str = QUERY_STREAM,
    ) -> None:
        if user_id is not None and not user_id.strip():
            raise ValueError("query outbox user_id must not be blank")
        if not stream_name.strip():
            raise ValueError("query outbox stream name must not be blank")
        if stream_name.startswith("agenticrag:e2e:") and user_id is None:
            # Acceptance runs share MySQL with ordinary workers.  A private
            # Redis stream prevents broker delivery collisions, while this
            # required user predicate prevents an accidentally reused stream
            # name from leasing another acceptance user's outbox row.
            raise ValueError("private query outbox stream requires a user_id")
        self._factory = factory
        self._user_id = user_id
        self._stream_name = stream_name

    async def list_pending(
        self, limit: int, *, aggregate_type: str | None = None
    ) -> list[OutboxRecord]:
        async with self._factory() as session:
            return await SqlAlchemyOutboxRepository(session).list_pending(
                limit,
                aggregate_type=aggregate_type,
                user_id=self._user_id,
                stream_name=self._stream_name,
            )

    async def claim_pending(
        self, limit: int, *, aggregate_type: str | None = None
    ) -> list[OutboxRecord]:
        async with self._factory.begin() as session:
            return await SqlAlchemyOutboxRepository(session).claim_pending(
                limit,
                aggregate_type=aggregate_type,
                user_id=self._user_id,
                stream_name=self._stream_name,
            )

    async def mark_dispatched(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).mark_dispatched(
                outbox_id,
                user_id=self._user_id,
                stream_name=self._stream_name,
            )

    async def schedule_retry(self, outbox_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyOutboxRepository(session).schedule_retry(
                outbox_id,
                user_id=self._user_id,
                stream_name=self._stream_name,
            )


async def run(
    settings: Settings,
    dependencies_factory: DependenciesFactory = build_query_dependencies,
) -> None:
    """Run the worker with deployment-owned QueryGraph dependencies."""
    container = build_container(settings)
    dependencies: QueryGraphDependencies | None = None
    try:
        from agentic_rag.persistence.schema_readiness import check_chat_schema
        await check_chat_schema(container.mysql_engine)
        await ensure_active_child_alias(container, settings)
        dependencies = await dependencies_factory(container, settings)
        shared_concurrency = dependencies.concurrency or ConcurrencyManager()
        async with container.checkpoints.open_query() as checkpointer:
            worker = QueryWorker(
                runs=TransactionalRunRepository(container.repositories.session_factory),
                broker=container.broker,
                graph_factory=build_graph_factory(dependencies, checkpointer),
                worker_id=f"{socket.gethostname()}:{os.getpid()}",
                concurrency=shared_concurrency,
                trace_recorder=dependencies.trace_recorder,
                event_emitter=dependencies.event_emitter,
                outbox_dispatcher=OutboxDispatcher(
                    TransactionalQueryOutboxAdapter(
                        container.repositories.session_factory
                    ),
                    container.broker,
                    aggregate_type="query_run",
                ),
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
        if dependencies is not None:
            await close_query_dependencies(dependencies)
        await container.close()


def main() -> int:
    try:
        asyncio.run(run(get_settings()))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    main()
