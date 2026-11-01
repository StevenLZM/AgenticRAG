"""SQL context accounting and first-result acceptance under the durable Job fence."""

from __future__ import annotations
import math
from datetime import UTC, datetime
from uuid import uuid4
from sqlalchemy import (
    JSON,
    Column,
    Float,
    Integer,
    String,
    Table,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import IntegrityError
from agentic_rag.persistence.repositories import metadata, ingestion_jobs
from agentic_rag.ingestion.worker import _live_claim_predicates, LeaseLost
from agentic_rag.ingestion.contextualization.store import BudgetExhausted
from agentic_rag.models.contextual import ContextArtifactRef

context_build_budgets = Table(
    "context_build_budgets",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("document_version_id", String(36), nullable=False),
    Column("build_fingerprint", String(64), nullable=False),
    Column("request_count", Integer, nullable=False),
    Column("reserved_tokens", Integer, nullable=False),
    Column("consumed_tokens", Integer, nullable=False),
    Column("reserved_cost", Float, nullable=False),
    Column("consumed_cost", Float, nullable=False),
)
context_build_items = Table(
    "context_build_items",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("build_id", String(64), nullable=False, index=True),
    Column("source_child_id", String(255), nullable=False),
    Column("status", String(16), nullable=False),
    Column("attempt_count", Integer, nullable=False),
    Column("artifact_ref", JSON),
    Column("accepted_generation", Integer),
)
context_build_reservations = Table(
    "context_build_reservations",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("build_id", String(64), nullable=False, index=True),
    Column("item_id", String(64), nullable=False),
    Column("claim_generation", Integer, nullable=False),
    Column("tokens", Integer, nullable=False),
    Column("cost", Float, nullable=False),
    Column("status", String(16), nullable=False),
)


class SqlAlchemyContextBuildStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._factory = session_factory

    async def _fence(self, session, key, claim):
        key.check_claim(claim)
        # A write locks the durable Job until this SAME transaction commits.
        # A separate assert_lease read cannot protect the accepted pointer.
        result = await session.execute(
            update(ingestion_jobs)
            .where(*_live_claim_predicates(claim, datetime.now(UTC)))
            .values(heartbeat_at=ingestion_jobs.c.heartbeat_at)
        )
        if result.rowcount != 1:
            raise LeaseLost("context build lease lost")

    async def _insert_absent(self, session, table, values):
        # Savepoint keeps a concurrent first-insert conflict from invalidating
        # the outer lease-fenced transaction. No request has dispatched yet.
        try:
            async with session.begin_nested():
                await session.execute(insert(table).values(**values))
        except IntegrityError:
            pass

    async def _item(self, session, key):
        await self._insert_absent(
            session,
            context_build_items,
            dict(
                id=key.item_id,
                build_id=key.build_id,
                source_child_id=key.source_child_id,
                status="pending",
                attempt_count=0,
            ),
        )
        return (
            (
                await session.execute(
                    select(context_build_items)
                    .where(context_build_items.c.id == key.item_id)
                    .with_for_update()
                )
            )
            .mappings()
            .one()
        )

    async def get_ready(self, key, claim):
        async with self._factory.begin() as session:
            await self._fence(session, key, claim)
            row = (
                (
                    await session.execute(
                        select(context_build_items).where(
                            context_build_items.c.id == key.item_id
                        )
                    )
                )
                .mappings()
                .first()
            )
            return (
                ContextArtifactRef.model_validate(row["artifact_ref"])
                if row and row["status"] == "ready"
                else None
            )

    async def reserve(self, key, claim, *, tokens, cost, build):
        if tokens < 0 or not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid reservation usage")
        async with self._factory.begin() as session:
            await self._fence(session, key, claim)
            await self._insert_absent(
                session,
                context_build_budgets,
                dict(
                    id=key.build_id,
                    user_id=key.user_id,
                    document_id=key.document_id,
                    document_version_id=key.document_version_id,
                    build_fingerprint=key.build_fingerprint,
                    request_count=0,
                    reserved_tokens=0,
                    consumed_tokens=0,
                    reserved_cost=0,
                    consumed_cost=0,
                ),
            )
            # Conditional arithmetic is atomic even when distinct Job rows share
            # this document/build. Reading counters then writing loses the cap.
            charged = await session.execute(
                update(context_build_budgets)
                .where(
                    context_build_budgets.c.id == key.build_id,
                    context_build_budgets.c.request_count + 1 <= build.max_requests,
                    context_build_budgets.c.reserved_tokens
                    + context_build_budgets.c.consumed_tokens
                    + tokens
                    <= build.max_tokens,
                    context_build_budgets.c.reserved_cost
                    + context_build_budgets.c.consumed_cost
                    + cost
                    <= build.max_cost,
                )
                .values(
                    request_count=context_build_budgets.c.request_count + 1,
                    reserved_tokens=context_build_budgets.c.reserved_tokens + tokens,
                    reserved_cost=context_build_budgets.c.reserved_cost + cost,
                )
            )
            if charged.rowcount != 1:
                raise BudgetExhausted("context document budget exhausted")
            await self._item(session, key)
            await session.execute(
                update(context_build_items)
                .where(context_build_items.c.id == key.item_id)
                .values(attempt_count=context_build_items.c.attempt_count + 1)
            )
            rid = uuid4().hex
            await session.execute(
                insert(context_build_reservations).values(
                    id=rid,
                    build_id=key.build_id,
                    item_id=key.item_id,
                    claim_generation=claim.claim_generation,
                    tokens=tokens,
                    cost=cost,
                    status="reserved",
                )
            )
            return rid

    async def settle(self, key, claim, reservation_id, *, tokens, cost):
        if tokens < 0 or not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid settled usage")
        async with self._factory.begin() as session:
            await self._fence(session, key, claim)
            row = (
                (
                    await session.execute(
                        select(context_build_reservations).where(
                            context_build_reservations.c.id == reservation_id,
                            context_build_reservations.c.build_id == key.build_id,
                            context_build_reservations.c.item_id == key.item_id,
                            context_build_reservations.c.claim_generation
                            == claim.claim_generation,
                        )
                    )
                )
                .mappings()
                .one()
            )
            if row["status"] == "settled":
                return
            await session.execute(
                update(context_build_budgets)
                .where(context_build_budgets.c.id == key.build_id)
                .values(
                    reserved_tokens=context_build_budgets.c.reserved_tokens
                    - row["tokens"],
                    reserved_cost=context_build_budgets.c.reserved_cost - row["cost"],
                    consumed_tokens=context_build_budgets.c.consumed_tokens + tokens,
                    consumed_cost=context_build_budgets.c.consumed_cost + cost,
                )
            )
            await session.execute(
                update(context_build_reservations)
                .where(context_build_reservations.c.id == reservation_id)
                .values(tokens=tokens, cost=cost, status="settled")
            )

    async def accept(self, key, claim, ref):
        async with self._factory.begin() as session:
            await self._fence(session, key, claim)
            row = await self._item(session, key)
            if row and row["status"] == "ready":
                return ContextArtifactRef.model_validate(row["artifact_ref"])
            await self._fence(session, key, claim)
            await session.execute(
                update(context_build_items)
                .where(context_build_items.c.id == key.item_id)
                .values(
                    status="ready",
                    artifact_ref=ref.model_dump(mode="json"),
                    accepted_generation=claim.claim_generation,
                )
            )
            return ref
