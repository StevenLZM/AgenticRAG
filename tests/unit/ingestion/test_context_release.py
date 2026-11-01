"""Offline release admission and restart boundaries use real SQL state."""

from datetime import UTC, datetime, timedelta
import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from agentic_rag.persistence.repositories import metadata, ingestion_jobs
from tests.unit.ingestion.test_worker_store import _seed
from agentic_rag.persistence.context_release import (
    ReleaseStore,
    ReleaseNotReady,
    check_context_admission,
)


async def install(store, factory, roles=("api", "query", "ingestion")):
    from tests.unit.ingestion.test_maintenance_receipt import signed_receipt

    inventory = [{"instance": role, "role": role} for role in roles]
    envelope, verifier = signed_receipt(
        from_generation="old-v1",
        to_generation="old-v1",
        config_hash="a" * 64,
        instances=inventory,
        previous_instances=inventory,
    )
    store.verifier = verifier
    await store.install(
        generation="old-v1", config_hash="a" * 64, roles=roles, receipt=envelope
    )
    for role in roles:
        factory.configure(
            info={"context_config_hash": "a" * 64, "context_instance_id": role}
        )
        await store.attest(role, "a" * 64)
    await store.reopen()


@pytest.fixture
async def release_sql(tmp_path):
    from agentic_rag.persistence import context_registry, context_builds  # noqa: F401

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/release.db")
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        from agentic_rag.orchestration.schema import metadata as orchestration_metadata

        await connection.run_sync(orchestration_metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_gate_blocks_stale_process_and_requires_all_role_receipts(release_sql):
    store = ReleaseStore(release_sql)
    await install(store, release_sql)
    release_sql.configure(info={"context_config_hash": "a" * 64})
    async with release_sql.begin() as db:
        await check_context_admission(db)
    await store.pause()
    async with release_sql.begin() as db:
        with pytest.raises(ReleaseNotReady):
            await check_context_admission(db)
    await store.assert_drained()
    await store.set_configuration("new-v1", "b" * 64)
    release_sql.configure(info={"context_config_hash": "b" * 64})
    for role in ("api", "query"):
        release_sql.configure(
            info={"context_config_hash": "b" * 64, "context_instance_id": role}
        )
        await store.attest(role, "b" * 64)
    with pytest.raises(ReleaseNotReady, match="roles"):
        await store.reopen()
    release_sql.configure(
        info={"context_config_hash": "b" * 64, "context_instance_id": "ingestion"}
    )
    await store.attest("ingestion", "b" * 64)
    await store.reopen()
    release_sql.configure(info={"context_config_hash": "a" * 64})
    async with release_sql.begin() as db:
        with pytest.raises(ReleaseNotReady, match="configuration"):
            await check_context_admission(db)
    release_sql.configure(info={"context_config_hash": "b" * 64})
    async with release_sql.begin() as db:
        await check_context_admission(db)


@pytest.mark.asyncio
async def test_pause_drain_checks_durable_ingestion_including_pre_pause_queue(
    release_sql,
):
    await _seed(
        release_sql, job_status="queued", version_status="uploaded", attempt_count=0
    )
    store = ReleaseStore(release_sql)
    await install(store, release_sql, ("api",))
    await store.pause()
    with pytest.raises(ReleaseNotReady, match="ingestion"):
        await store.assert_drained()
    async with release_sql.begin() as db:
        await db.execute(update(ingestion_jobs).values(status="failed"))
        from agentic_rag.persistence.repositories import document_versions

        await db.execute(update(document_versions).values(status="failed"))
    await store.assert_drained()


@pytest.mark.asyncio
async def test_disabled_session_never_requires_context_schema(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/empty.db")
    async with async_sessionmaker(engine).begin() as db:
        await check_context_admission(db)
    await engine.dispose()


@pytest.mark.asyncio
async def test_rebuild_lease_survives_restart_and_never_uses_active_upload_completion(
    release_sql,
):
    from agentic_rag.persistence.context_registry import (
        RebuildJobStore,
        retrieval_rebuild_jobs,
    )
    from agentic_rag.persistence.repositories import documents
    from agentic_rag.persistence.context_builds import SqlAlchemyContextBuildStore
    from agentic_rag.ingestion.contextualization.store import BuildKey
    from agentic_rag.models.contextual import ContextArtifactRef

    await _seed(release_sql, job_status="completed", version_status="active")
    async with release_sql.begin() as db:
        await db.execute(
            update(documents).values(status="active", active_version_id="version-1")
        )
    jobs = RebuildJobStore(release_sql)
    await jobs.enqueue(
        owner="user-1",
        document="doc-1",
        version="version-1",
        generation="new-v1",
        payload={"config": "immutable"},
    )
    claim, payload = await jobs.claim_next("worker-a", "new-v1")
    key = BuildKey(
        user_id="user-1",
        document_id="doc-1",
        document_version_id="version-1",
        build_fingerprint="f" * 64,
        source_child_id="child",
    )
    ref = ContextArtifactRef(uri="artifact://first.json", sha256="a" * 64, size_bytes=1)
    await SqlAlchemyContextBuildStore(release_sql).accept(key, claim, ref)
    async with release_sql.begin() as db:
        await db.execute(
            update(retrieval_rebuild_jobs).values(
                lease_expires_at=datetime.now(UTC) - timedelta(seconds=1)
            )
        )
    second, _ = await RebuildJobStore(release_sql).claim_next("worker-b", "new-v1")
    assert second.claim_generation > claim.claim_generation
    with pytest.raises(RuntimeError, match="lease"):
        await SqlAlchemyContextBuildStore(release_sql).accept(key, claim, ref)
    assert await SqlAlchemyContextBuildStore(release_sql).get_ready(key, second) == ref
    with pytest.raises(ValueError, match="durable ready"):
        await jobs.finish(second)


@pytest.mark.asyncio
async def test_arming_requires_external_authority_and_every_instance(release_sql):
    from tests.unit.ingestion.test_maintenance_receipt import signed_receipt

    store = ReleaseStore(release_sql)
    with pytest.raises(ReleaseNotReady, match="authority"):
        await store.install(generation="old-v1", config_hash="a" * 64, roles=("api",))
    inventory = [
        {"instance": "api-one", "role": "api"},
        {"instance": "api-two", "role": "api"},
    ]
    receipt, verifier = signed_receipt(
        from_generation="old-v1",
        to_generation="old-v1",
        config_hash="a" * 64,
        instances=inventory,
    )
    store.verifier = verifier
    await store.install(
        generation="old-v1", config_hash="a" * 64, roles=("api",), receipt=receipt
    )
    release_sql.configure(
        info={"context_config_hash": "a" * 64, "context_instance_id": "api-one"}
    )
    await store.attest("api", "a" * 64)
    with pytest.raises(ReleaseNotReady, match="roles"):
        await store.reopen()
    release_sql.configure(
        info={"context_config_hash": "a" * 64, "context_instance_id": "unknown"}
    )
    with pytest.raises(ReleaseNotReady, match="not selected"):
        await store.attest("api", "a" * 64)
    release_sql.configure(
        info={"context_config_hash": "a" * 64, "context_instance_id": "api-two"}
    )
    await store.attest("api", "a" * 64)
    await store.reopen()


@pytest.mark.asyncio
async def test_receipt_nonce_replay_and_incomplete_old_inventory_fail(release_sql):
    from tests.unit.ingestion.test_maintenance_receipt import signed_receipt
    from agentic_rag.persistence.context_release import consume_receipt

    receipt, verifier = signed_receipt()
    async with release_sql.begin() as db:
        with pytest.raises(ReleaseNotReady, match="missing or unknown"):
            await consume_receipt(
                db,
                verifier,
                receipt,
                operation="bad",
                source="old-v1",
                target="new-v1",
                config_hash="b" * 64,
                previous=[{"instance": "other-old", "role": "api"}],
                roles=("api",),
            )
    async with release_sql.begin() as db:
        await consume_receipt(
            db,
            verifier,
            receipt,
            operation="one",
            source="old-v1",
            target="new-v1",
            config_hash="b" * 64,
        )
    async with release_sql.begin() as db:
        with pytest.raises(ReleaseNotReady, match="nonce"):
            await consume_receipt(
                db,
                verifier,
                receipt,
                operation="two",
                source="old-v1",
                target="new-v1",
                config_hash="b" * 64,
            )
