"""Explicit, cancellable concurrency budgets for one query runtime process."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class ConcurrencyManager:
    """Own global budgets and create a fresh bounded subagent budget per Run.

    The injected semaphores let the application share limits among all active
    runs.  A subagent dispatcher must obtain a ``subagent_slot`` as well, so a
    single Run cannot consume more than its server-configured allowance.
    """

    def __init__(
        self,
        *,
        run_limit: int = 4,
        llm_limit: int = 8,
        reranker_limit: int = 1,
        per_run_subagent_limit: int = 3,
        run_semaphore: asyncio.Semaphore | None = None,
        llm_semaphore: asyncio.Semaphore | None = None,
        reranker_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        if min(run_limit, llm_limit, reranker_limit, per_run_subagent_limit) < 1:
            raise ValueError("concurrency limits must be positive")
        self._run = run_semaphore or asyncio.Semaphore(run_limit)
        self._llm = llm_semaphore or asyncio.Semaphore(llm_limit)
        self._reranker = reranker_semaphore or asyncio.Semaphore(reranker_limit)
        self._per_run_subagent_limit = per_run_subagent_limit

    @property
    def per_run_subagent_limit(self) -> int:
        return self._per_run_subagent_limit

    def new_subagent_semaphore(self, requested: int) -> asyncio.Semaphore:
        if requested < 1:
            raise ValueError("requested subagent parallelism must be positive")
        return asyncio.Semaphore(min(requested, self._per_run_subagent_limit))

    @asynccontextmanager
    async def run_slot(self) -> AsyncIterator[None]:
        async with self._run:
            yield

    @asynccontextmanager
    async def llm_slot(self) -> AsyncIterator[None]:
        async with self._llm:
            yield

    @asynccontextmanager
    async def reranker_slot(self) -> AsyncIterator[None]:
        async with self._reranker:
            yield

    @asynccontextmanager
    async def subagent_slot(self, semaphore: asyncio.Semaphore) -> AsyncIterator[None]:
        async with semaphore:
            yield
