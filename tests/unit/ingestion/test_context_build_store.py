"""Real SQL transactions fence accepted pointers and preserve crash spend."""

from datetime import UTC, datetime, timedelta
import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from agentic_rag.persistence.context_builds import (
    SqlAlchemyContextBuildStore,
    context_build_budgets,
    context_build_items,
)
from agentic_rag.persistence.repositories import metadata, ingestion_jobs
from agentic_rag.ingestion.contextualization.store import BuildKey, BudgetExhausted
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.models.contextual import ContextArtifactRef
from tests.unit.ingestion.test_worker_store import _seed
from tests.unit.ingestion.test_context_service import config


@pytest.fixture
async def sql(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/context.db")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await _seed(
        factory, job_status="running", version_status="building", attempt_count=1
    )
    claim = IngestionJobClaim(
        job_id="job-1",
        user_id="user-1",
        document_id="doc-1",
        document_version_id="version-1",
        owner="worker-1",
        claim_generation=1,
    )
    key = BuildKey(
        user_id=claim.user_id,
        document_id=claim.document_id,
        document_version_id=claim.document_version_id,
        build_fingerprint="a" * 64,
        source_child_id="raw",
    )
    yield factory, claim, key
    await engine.dispose()


@pytest.mark.asyncio
async def test_crash_reservation_is_not_reset_by_new_store_or_child(sql):
    factory, claim, key = sql
    build = config().model_copy(update={"max_requests": 1})
    await SqlAlchemyContextBuildStore(factory).reserve(
        key, claim, tokens=100, cost=0.01, build=build
    )
    with pytest.raises(BudgetExhausted):
        await SqlAlchemyContextBuildStore(factory).reserve(
            key.model_copy(update={"source_child_id": "other"}),
            claim,
            tokens=100,
            cost=0.01,
            build=build,
        )
    async with factory() as session:
        row = (await session.execute(select(context_build_budgets))).mappings().one()
        assert row["request_count"] == 1 and row["reserved_tokens"] == 100


@pytest.mark.asyncio
async def test_accept_is_first_winner_and_old_lease_cannot_replace_it(sql):
    factory, claim, key = sql
    store = SqlAlchemyContextBuildStore(factory)
    first = ContextArtifactRef(
        uri="artifact://first.json", sha256="a" * 64, size_bytes=1
    )
    second = ContextArtifactRef(
        uri="artifact://second.json", sha256="b" * 64, size_bytes=2
    )
    assert await store.accept(key, claim, first) == first
    assert await store.accept(key, claim, second) == first
    async with factory.begin() as session:
        await session.execute(
            update(ingestion_jobs).values(
                attempt_count=2,
                lease_owner="worker-2",
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )
    with pytest.raises(RuntimeError, match="lease"):
        await store.accept(key, claim, second)
    next_claim = claim.model_copy(update={"claim_generation": 2, "owner": "worker-2"})
    assert await store.get_ready(key, next_claim) == first
    async with factory() as session:
        row = (await session.execute(select(context_build_items))).mappings().one()
        assert row["accepted_generation"] == 1


@pytest.mark.asyncio
async def test_settlement_is_idempotent_and_cross_scope_rejected(sql):
    factory, claim, key = sql
    store = SqlAlchemyContextBuildStore(factory)
    rid = await store.reserve(key, claim, tokens=100, cost=0.01, build=config())
    await store.settle(key, claim, rid, tokens=40, cost=0.004)
    await store.settle(key, claim, rid, tokens=40, cost=0.004)
    async with factory() as session:
        row = (await session.execute(select(context_build_budgets))).mappings().one()
        assert row["reserved_tokens"] == 0 and row["consumed_tokens"] == 40
    with pytest.raises(ValueError):
        await store.accept(
            key.model_copy(update={"user_id": "other"}),
            claim,
            ContextArtifactRef(uri="artifact://x.json", sha256="b" * 64, size_bytes=1),
        )


@pytest.mark.asyncio
async def test_concurrent_reservations_cannot_exceed_document_cap(sql):
    import asyncio

    factory, claim, key = sql
    build = config().model_copy(update={"max_requests": 1})
    outcomes = await asyncio.gather(
        *(
            SqlAlchemyContextBuildStore(factory).reserve(
                key.model_copy(update={"source_child_id": str(i)}),
                claim,
                tokens=100,
                cost=0.01,
                build=build,
            )
            for i in range(4)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(value, str) for value in outcomes) == 1
    assert sum(isinstance(value, BudgetExhausted) for value in outcomes) == 3
