"""Strict cleanup contract used by live acceptance runners."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agentic_rag.bootstrap import AppContainer, Repositories


class _FailingElasticsearch:
    async def close(self) -> None:
        raise OSError("close failed")


class _Redis:
    async def aclose(self) -> None:
        return None


class _Engine:
    async def dispose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_container_strict_close_reports_a_boundary_failure_after_siblings_close() -> None:
    container = AppContainer(
        settings=cast(Any, object()),
        repositories=Repositories(session_factory=cast(Any, object())),
        broker=cast(Any, object()),
        artifacts=cast(Any, object()),
        checkpoints=cast(Any, object()),
        elasticsearch=cast(Any, _FailingElasticsearch()),
        readiness_checks=cast(Any, object()),
        document_service=cast(Any, object()),
        mysql_engine=cast(Any, _Engine()),
        redis=cast(Any, _Redis()),
        reranker_initialized=True,
        runtime_snapshot=cast(Any, object()),
    )

    with pytest.raises(RuntimeError, match="container cleanup failed: OSError"):
        await container.close(raise_on_error=True)
