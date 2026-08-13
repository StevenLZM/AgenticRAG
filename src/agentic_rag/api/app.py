"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_rag.api.errors import register_error_handlers
from agentic_rag.api.health import health_router
from agentic_rag.api.documents import documents_router
from agentic_rag.api.feedback import feedback_router
from agentic_rag.api.memories import memories_router
from agentic_rag.api.query_runs import query_runs_router
from agentic_rag.bootstrap import build_container
from agentic_rag.config import Settings


def create_app(settings: Settings) -> FastAPI:
    """Build one explicit application instance and own its process resources."""
    container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
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
