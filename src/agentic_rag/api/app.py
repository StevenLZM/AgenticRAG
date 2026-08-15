"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from agentic_rag.api.errors import register_error_handlers
from agentic_rag.api.health import health_router
from agentic_rag.api.documents import documents_router
from agentic_rag.api.feedback import feedback_router
from agentic_rag.api.memories import memories_router
from agentic_rag.api.query_runs import query_runs_router
from agentic_rag.bootstrap import build_container
from agentic_rag.config import Settings


def create_app(settings: Settings, *, container: Any | None = None) -> FastAPI:
    """Build an app, optionally reusing a deployment-owned container.

    The normal API process owns the container it constructs.  Worker/API
    composition and in-process acceptance runners may inject the already
    composed container so Mem0, event sinks and runtime snapshots are not
    initialized twice with competing provider resources.
    """
    owns_container = container is None
    if container is None:
        container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_container:
                await container.close()

    app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
    app.state.container = container
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(query_runs_router)
    app.include_router(memories_router)
    app.include_router(feedback_router)
    register_error_handlers(app)
    return app
