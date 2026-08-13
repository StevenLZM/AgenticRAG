"""Transactional creation and cancellation of durable query runs."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.persistence.repositories import ActiveRunConflict, QueryRun, RunRepository, SqlAlchemyRunRepository
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class TransactionFactory(Protocol):
    def begin(self) -> AbstractAsyncContextManager[object]: ...


class RunManager:
    """Persist a query and its outbox notification under one database transaction.

    ``RunRepository.create_queued`` is deliberately the sole creation port: its
    SQL adapter inserts both the Run and the matching outbox record together.
    """

    def __init__(self, *, session_factory: TransactionFactory, runs: RunRepository) -> None:
        self.session_factory = session_factory
        self.runs = runs

    async def create(
        self,
        scope: UserScope,
        thread_id: str,
        question: str,
        snapshot: RuntimeConfigSnapshot,
    ) -> QueryRun:
        normalized_thread = thread_id.strip()
        normalized_question = question.strip()
        if not normalized_thread:
            raise ValueError("thread_id must not be blank")
        if not normalized_question:
            raise ValueError("question must not be blank")
        try:
            async with self.session_factory.begin() as transaction:
                return await self.runs.create_queued(
                    scope, normalized_thread, snapshot,
                    question=normalized_question, transaction=transaction,  # type: ignore[arg-type]
                )
        except ActiveRunConflict as error:
            # A database uniqueness race rolls back the create transaction.
            # Resolve the current active row through the scoped repository port
            # so API callers receive a Location header without provider detail.
            resolver = getattr(self.runs, "get_active", None)
            if callable(resolver) and getattr(error, "existing_run_id", None) is None:
                try:
                    existing = await resolver(scope, normalized_thread)
                except Exception:
                    existing = None
                if existing is not None:
                    error.existing_run_id = existing.id
            raise

    async def request_cancel(self, scope: UserScope, run_id: str) -> RunStatus:
        return await self.runs.request_cancel(run_id, scope)

    async def get(self, run_id: str, scope: UserScope) -> QueryRun | None:
        """Read one user-owned Run for API status and cancellation responses."""
        return await self.runs.get(run_id, scope)

    async def get_for_delivery(self, run_id: str) -> QueryRun | None:
        """Read one trusted broker-delivered Run for worker coordination."""
        return await self.runs.get_for_delivery(run_id)


class TransactionalRunRepository:
    """Open a short MySQL transaction for each worker lifecycle operation."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def create_queued(
        self, scope: UserScope, thread_id: str, snapshot: RuntimeConfigSnapshot,
        *, question: str = "", transaction: AsyncSession | None = None,
    ) -> QueryRun:
        if transaction is not None:
            return await SqlAlchemyRunRepository(transaction).create_queued(
                scope, thread_id, snapshot, question=question, transaction=transaction,
            )
        async with self._factory.begin() as session:
            return await SqlAlchemyRunRepository(session).create_queued(
                scope, thread_id, snapshot, question=question,
            )

    async def get(self, run_id: str, scope: UserScope) -> QueryRun | None:
        async with self._factory() as session:
            return await SqlAlchemyRunRepository(session).get(run_id, scope)

    async def get_for_delivery(self, run_id: str) -> QueryRun | None:
        async with self._factory() as session:
            return await SqlAlchemyRunRepository(session).get_for_delivery(run_id)

    async def get_active(self, scope: UserScope, thread_id: str) -> QueryRun | None:
        async with self._factory() as session:
            return await SqlAlchemyRunRepository(session).get_active(scope, thread_id)

    async def claim(self, run_id: str, owner: str, lease_seconds: int) -> QueryRun | None:
        async with self._factory.begin() as session:
            return await SqlAlchemyRunRepository(session).claim(
                run_id, owner, lease_seconds
            )

    async def heartbeat(
        self, run_id: str, owner: str, lease_seconds: int, *, claim_generation: int
    ) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyRunRepository(session).heartbeat(
                run_id, owner, lease_seconds, claim_generation=claim_generation,
            )

    async def request_cancel(self, run_id: str, scope: UserScope) -> RunStatus:
        async with self._factory.begin() as session:
            return await SqlAlchemyRunRepository(session).request_cancel(run_id, scope)

    async def finish(
        self, run_id: str, status: RunStatus, result_ref: str | None,
        error_code: str | None, *, owner: str, claim_generation: int,
        answer: dict[str, object] | None = None,
    ) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyRunRepository(session).finish(
                run_id, status, result_ref, error_code,
                owner=owner, claim_generation=claim_generation,
                answer=answer,
            )
