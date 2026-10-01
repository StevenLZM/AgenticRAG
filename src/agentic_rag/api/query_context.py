"""Shared server-owned query scope, dependencies and immutable configuration."""

from collections.abc import Mapping
from typing import Any
from fastapi import Request, status
from agentic_rag.api.errors import ApiException
from agentic_rag.domain.models import UserScope
from agentic_rag.runtime.models import RuntimeConfigSnapshot


def request_scope(request: Request) -> UserScope:
    settings = getattr(request.app.state.container, "settings", None)
    user_id = getattr(settings, "default_user_id", "default_user")
    try:
        return UserScope(user_id=str(user_id))
    except (TypeError, ValueError) as error:
        raise ApiException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INVALID_SERVER_SCOPE",
            message="The server user scope is invalid.",
        ) from error


def require_dependency(request: Request, name: str) -> Any:
    dependency = getattr(request.app.state.container, name, None)
    if dependency is None:
        raise ApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DEPENDENCY_UNAVAILABLE",
            message="The query runtime is temporarily unavailable.",
            retryable=True,
            degraded_components=(name,),
        )
    return dependency


def runtime_snapshot(request: Request) -> RuntimeConfigSnapshot:
    container = request.app.state.container
    candidate = getattr(container, "runtime_snapshot", None)
    if candidate is None:
        candidate = getattr(container, "query_snapshot", None)
    if isinstance(candidate, RuntimeConfigSnapshot):
        return candidate
    if isinstance(candidate, Mapping):
        try:
            return RuntimeConfigSnapshot.model_validate(candidate)
        except (TypeError, ValueError):
            pass

    settings = container.settings
    # Settings are immutable input for a Run snapshot.  Prompt hashes are
    # intentionally empty in the bootstrap fallback; a production deployment
    # should inject the loaded, content-addressed prompt map on the container.
    try:
        return RuntimeConfigSnapshot(
            app_version="0.1.0",
            graph_version="query-v1",
            prompt_version="prompt-v1",
            main_model_id=str(getattr(settings, "main_model", "deepseek-v4-pro")),
            light_model_id=str(getattr(settings, "light_model", "deepseek-v4-flash")),
            embedding_model=str(
                getattr(settings, "embedding_model", "text-embedding-v3")
            ),
            # RuntimeConfigSnapshot intentionally fixes the vector contract at
            # 1024 dimensions; reject a misconfigured Settings value rather
            # than silently snapshotting an incompatible index.
            embedding_dimensions=1024,
            reranker_version=str(getattr(settings, "reranker_model", "reranker-v1")),
            retrieval_config_version="retrieval-v1",
            index_generation=str(getattr(settings, "index_generation", "index-v1")),
            memory_config_version="memory-v1",
            max_research_rounds=int(getattr(settings, "max_research_rounds", 6)),
            max_answer_revisions=int(getattr(settings, "max_answer_revisions", 1)),
            query_run_timeout_seconds=int(
                getattr(settings, "query_run_timeout_seconds", 300)
            ),
            max_evidence_tokens=int(getattr(settings, "max_evidence_tokens", 12_000)),
            research_context_soft_limit_tokens=int(
                getattr(settings, "research_context_soft_limit_tokens", 16_000)
            ),
            max_parallel_subagents_per_run=int(
                getattr(settings, "max_parallel_subagents_per_run", 3)
            ),
        )
    except (TypeError, ValueError) as error:
        raise ApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="RUNTIME_SNAPSHOT_UNAVAILABLE",
            message="The query runtime configuration is unavailable.",
            retryable=True,
        ) from error
