"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag.api.errors import register_error_handlers
from agentic_rag.api.health import health_router
from agentic_rag.api.documents import documents_router
from agentic_rag.api.feedback import feedback_router
from agentic_rag.api.memories import memories_router
from agentic_rag.api.query_runs import query_runs_router
from agentic_rag.api.chat_sessions import chat_sessions_router
from agentic_rag.api.runtime_summary import runtime_summary_router
from agentic_rag.bootstrap import build_container
from agentic_rag.config import Settings


class ConsoleStaticFiles(StaticFiles):
    """Serve only the console assets, never the HTML source through ``/static``."""

    _allowed_assets = frozenset({"app.css", "app.js", "chat-state.js", "chat-api.js", "chat-view.js", "console-tools.js"})

    async def get_response(self, path: str, scope: Any):
        if path not in self._allowed_assets:
            raise HTTPException(status_code=404)
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


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

    static_dir = Path(__file__).with_name("static")
    app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
    app.state.container = container
    app.mount("/static", ConsoleStaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def console() -> FileResponse:
        """Serve the browser console from the same API origin."""
        return FileResponse(static_dir / "index.html")

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(query_runs_router)
    app.include_router(chat_sessions_router)
    app.include_router(memories_router)
    app.include_router(feedback_router)
    app.include_router(runtime_summary_router)
    register_error_handlers(app)
    return app
