"""Real SQL active-source rebuilds and restartable release boundaries."""

from datetime import UTC, datetime
import pytest
from sqlalchemy import insert, select, update
from agentic_rag.persistence.repositories import (
    documents,
    document_versions,
    parent_chunks,
)
from agentic_rag.persistence.context_registry import (
    RebuildJobStore,
    retrieval_builds,
    register_build,
)
from agentic_rag.persistence.context_release import (
    ReleaseStore,
    ReleaseNotReady,
    context_gate,
    release_journal,
)
from agentic_rag.ingestion.contextualization.release import ContextRelease
from agentic_rag.ingestion.indexer import IndexWriter
from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.models.contextual import ContextArtifactRef
from tests.unit.ingestion.test_context_service import setup, fence
from tests.unit.ingestion.test_manifest import (
    FakeEmbedding,
    RecordingParentStore,
    RecordingChildStore,
    _context,
)
from tests.unit.ingestion.test_context_release import release_sql as _release_sql

from tests.unit.ingestion.test_maintenance_receipt import signed_receipt
from agentic_rag.persistence.staging import _parent_values


@pytest.fixture(name="release_sql")
async def release_fixture(tmp_path):
    async for factory in _release_sql.__wrapped__(tmp_path):
        yield factory


async def seed(factory, tmp_path):
    artifacts, ast, ref, parent, _ = setup(tmp_path, ("原始正文500。",))
    context = _context().model_copy(update={"index_generation": "old-v1"})
    now = datetime.now(UTC)
    async with factory.begin() as db:
        await db.execute(
            insert(documents).values(
                id="document-1",
                user_id="user-1",
                source_type="text",
                filename="example.txt",
                mime_type="text/plain",
                content_hash="a" * 64,
                status="active",
                active_version_id="version-1",
                created_at=now,
                updated_at=now,
            )
        )
        await db.execute(
            insert(document_versions).values(
                id="version-1",
                document_id="document-1",
                version_no=1,
                parser_version="p",
                pipeline_version="ingestion-v1",
                embedding_version="text-embedding-v3",
                index_generation="old-v1",
                status="active",
                canonical_ast_path=ref.uri,
                canonical_ast_hash=ref.sha256,
                created_at=now,
            )
        )
        values = _parent_values(context, parent)
        values["status"] = "active"
        await db.execute(insert(parent_chunks).values(**values))
    return artifacts, ref, parent, context


class Children(RecordingChildStore):
    async def child_identities(self, context):
        return tuple(
            (child.chunk.id, child.chunk.content_hash) for child in self.staged
        )

    async def activate(self, context):
        self.events.append("activate")


@pytest.mark.asyncio
async def test_active_source_rebuild_only_writes_children_and_registry(
    release_sql, tmp_path
):
    artifacts, ref, parent, context = await seed(release_sql, tmp_path)
    jobs = RebuildJobStore(release_sql)
    await jobs.enqueue(
        owner="user-1",
        document="document-1",
        version="version-1",
        generation="new-v1",
        payload={"config_hash": "a" * 64},
    )
    claim, _ = await jobs.claim_next("worker", "new-v1")
    parents = RecordingParentStore([])
    children = Children([])
    writer = IndexWriter(
        embedding=FakeEmbedding([[[0.0] * 1024]]),
        parent_store=parents,
        child_store=children,
        artifacts=artifacts,
    )
    async with release_sql() as db:
        original = (await db.execute(select(parent_chunks))).mappings().one()
    await writer.stage_search_representation(
        [parent],
        factory=release_sql,
        claim=claim,
        context=context.model_copy(update={"index_generation": "new-v1"}),
        canonical_ast=ref,
        before_side_effect=fence,
    )
    await jobs.finish(claim)
    assert parents.events == []
    async with release_sql() as db:
        assert dict((await db.execute(select(parent_chunks))).mappings().one()) == dict(
            original
        )
        version = (await db.execute(select(document_versions))).mappings().one()
        assert version["status"] == "active" and version["index_generation"] == "old-v1"
        assert (
            await db.execute(select(retrieval_builds.c.state))
        ).scalar_one() == "ready"


class Aliases:
    def __init__(self):
        self.generation = "old-v1"

    async def prepare_release_membership(self, generation, members):
        assert all(context.index_generation == generation for context, _ in members)

    async def switch_active_alias(self, generation):
        self.generation = generation

    async def validate_search_target(self, generation):
        assert self.generation == generation


async def registered(factory, artifacts, ref, parent, context, generation):
    manifest = VersionManifest(
        canonical_ast_sha256=ref.sha256,
        parent_count=1,
        child_count=1,
        embedding_model="text-embedding-v3",
        embedding_dimensions=1024,
        index_generation=generation,
    )
    saved = artifacts.put_json(f"manifests/{generation}.json", manifest.payload())
    async with factory.begin() as db:
        await register_build(
            db,
            context.model_copy(update={"index_generation": generation}),
            ContextArtifactRef(**saved.__dict__),
            manifest,
            parents=[parent],
        )


async def prepared(factory, tmp_path):
    artifacts, ref, parent, context = await seed(factory, tmp_path)
    for generation in ("old-v1", "new-v1"):
        await registered(factory, artifacts, ref, parent, context, generation)
    from tests.unit.runtime.test_query_composition import SNAPSHOT

    source_snapshot = SNAPSHOT.model_copy(update={"index_generation": "old-v1"})
    inventory = [{"instance": "old-api", "role": "api"}]
    install_receipt, verifier = signed_receipt(
        from_generation="old-v1",
        to_generation="old-v1",
        config_hash=source_snapshot.snapshot_id,
        instances=inventory,
        previous_instances=inventory,
    )
    store = ReleaseStore(factory, verifier)
    await store.install(
        generation="old-v1",
        config_hash=source_snapshot.snapshot_id,
        roles=("api",),
        receipt=install_receipt,
    )
    from tests.unit.runtime.test_query_composition import SNAPSHOT

    target_snapshot = SNAPSHOT.model_copy(update={"index_generation": "new-v1"})
    receipt, verifier = signed_receipt(
        nonce="release-1", config_hash=target_snapshot.snapshot_id
    )
    aliases = Aliases()
    release = ContextRelease(factory, artifacts, aliases, verifier)
    await release.prepare(
        "journal-1",
        source="old-v1",
        target="new-v1",
        target_config_hash=target_snapshot.snapshot_id,
        target_snapshot=target_snapshot,
        receipt=receipt,
    )
    return release, aliases


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash", ["projected", "membership", "alias_switched", "configured", "verified"]
)
async def test_resume_every_external_release_boundary_keeps_gate_closed(
    release_sql, tmp_path, crash
):
    release, aliases = await prepared(release_sql, tmp_path)

    async def fail(stage):
        if stage == crash:
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        await release.resume("journal-1", after_boundary=fail)
    async with release_sql() as db:
        assert await db.scalar(select(context_gate.c.state)) == "frozen"
    with pytest.raises(ReleaseNotReady, match="roles"):
        await ContextRelease(release_sql, release.artifacts, aliases).resume(
            "journal-1"
        )
    async with release_sql() as db:
        config_hash = await db.scalar(select(context_gate.c.config_hash))
    release_sql.configure(
        info={"context_config_hash": config_hash, "context_instance_id": "new-api"}
    )
    await ReleaseStore(release_sql).attest("api", config_hash)
    await release.resume("journal-1")
    async with release_sql() as db:
        assert await db.scalar(select(context_gate.c.state)) == "open"
        assert await db.scalar(select(release_journal.c.stage)) == "complete"
    assert aliases.generation == "new-v1"


@pytest.mark.asyncio
async def test_new_source_prevents_incomplete_rollback_and_deletion_never_resurrects(
    release_sql, tmp_path
):
    release, _ = await prepared(release_sql, tmp_path)
    async with release_sql.begin() as db:
        await db.execute(update(documents).values(active_version_id="new-version"))
        await db.execute(
            insert(document_versions).values(
                id="new-version",
                document_id="document-1",
                version_no=2,
                parser_version="p",
                pipeline_version="ingestion-v1",
                embedding_version="text-embedding-v3",
                index_generation="new-v1",
                status="active",
                created_at=datetime.now(UTC),
            )
        )
    with pytest.raises(ReleaseNotReady, match="missing compatible"):
        await release.mapping("old-v1")
    async with release_sql.begin() as db:
        await db.execute(
            update(documents).values(
                status="deleted", active_version_id=None, deletion_status="pending"
            )
        )
    assert await release.mapping("old-v1") == {}
    with pytest.raises(ReleaseNotReady, match="corpus changed"):
        await release.resume("journal-1")
    async with release_sql() as db:
        assert await db.scalar(select(documents.c.status)) == "deleted"


@pytest.mark.asyncio
async def test_partial_switch_compensates_via_separate_recoverable_journal(
    release_sql, tmp_path
):
    from tests.unit.runtime.test_query_composition import SNAPSHOT

    release, aliases = await prepared(release_sql, tmp_path)

    async def crash(stage):
        if stage == "alias_switched":
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        await release.resume("journal-1", after_boundary=crash)
    source_snapshot = SNAPSHOT.model_copy(update={"index_generation": "old-v1"})
    receipt, verifier = signed_receipt(
        nonce="rollback-1",
        from_generation="old-v1",
        to_generation="old-v1",
        config_hash=source_snapshot.snapshot_id,
        previous_instances=[{"instance": "new-api", "role": "api"}],
        instances=[{"instance": "rollback-api", "role": "api"}],
    )
    release.store.verifier = verifier
    with pytest.raises(ReleaseNotReady, match="roles"):
        await release.rollback(
            "journal-1", target_snapshot=source_snapshot, receipt=receipt
        )
    release_sql.configure(
        info={
            "context_config_hash": source_snapshot.snapshot_id,
            "context_instance_id": "rollback-api",
        }
    )
    await ReleaseStore(release_sql).attest("api", source_snapshot.snapshot_id)
    await release.resume("journal-1:rollback")
    async with release_sql() as db:
        assert await db.scalar(select(context_gate.c.generation)) == "old-v1"
        assert await db.scalar(select(document_versions.c.index_generation)) == "old-v1"
        assert (
            await db.scalar(
                select(release_journal.c.stage).where(
                    release_journal.c.id == "journal-1"
                )
            )
            == "aborted"
        )
    assert aliases.generation == "old-v1"


@pytest.mark.asyncio
async def test_crash_after_reopen_finishes_journal_without_closing_again(
    release_sql, tmp_path
):
    release, _ = await prepared(release_sql, tmp_path)
    with pytest.raises(ReleaseNotReady, match="roles"):
        await release.resume("journal-1")
    async with release_sql() as db:
        config_hash = await db.scalar(select(context_gate.c.config_hash))
    release_sql.configure(
        info={"context_config_hash": config_hash, "context_instance_id": "new-api"}
    )
    await ReleaseStore(release_sql).attest("api", config_hash)

    async def crash(stage):
        if stage == "reopened":
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        await release.resume("journal-1", after_boundary=crash)
    await release.resume("journal-1")
    async with release_sql() as db:
        assert await db.scalar(select(context_gate.c.state)) == "open"
        assert await db.scalar(select(release_journal.c.stage)) == "complete"


@pytest.mark.asyncio
async def test_global_release_does_not_ignore_another_owner(release_sql, tmp_path):
    release, _ = await prepared(release_sql, tmp_path)
    now = datetime.now(UTC)
    async with release_sql.begin() as db:
        await db.execute(
            insert(documents).values(
                id="other-doc",
                user_id="other-owner",
                source_type="text",
                filename="other.txt",
                mime_type="text/plain",
                content_hash="b" * 64,
                status="active",
                active_version_id="other-v",
                created_at=now,
                updated_at=now,
            )
        )
        await db.execute(
            insert(document_versions).values(
                id="other-v",
                document_id="other-doc",
                version_no=1,
                parser_version="p",
                pipeline_version="ingestion-v1",
                embedding_version="text-embedding-v3",
                index_generation="old-v1",
                status="active",
                created_at=now,
            )
        )
    with pytest.raises(ReleaseNotReady, match="other-doc/other-v"):
        await release.mapping("new-v1")


@pytest.mark.asyncio
async def test_publication_commit_records_dual_build_and_delete_revokes_all(
    release_sql, tmp_path
):
    from agentic_rag.ingestion.publisher import PublicationTarget
    from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
    from agentic_rag.persistence.context_registry import (
        retrieval_rebuild_jobs,
        RegistrySourceAuthority,
    )
    from agentic_rag.persistence.repositories import SqlAlchemyDocumentRepository
    from agentic_rag.domain.models import UserScope
    from tests.unit.runtime.test_query_composition import SNAPSHOT

    selected = SNAPSHOT.model_copy(update={"index_generation": "new-v1"})
    artifacts, ref, parent, context = await seed(release_sql, tmp_path)
    context = context.model_copy(update={"index_generation": "new-v1"})
    raw_path = f"documents/{context.user_id}/{context.document_id}/{context.document_version_id}/chunks/{context.pipeline_version}/parents.json"
    artifacts.put_json(raw_path, {"parents": [parent.model_dump(mode="json")]})
    manifest = VersionManifest(
        canonical_ast_sha256=ref.sha256,
        parent_count=1,
        child_count=1,
        embedding_model="text-embedding-v3",
        embedding_dimensions=1024,
        index_generation="new-v1",
    )
    stored = artifacts.put_json("published.json", manifest.payload())
    receipt, verifier = signed_receipt(
        from_generation="new-v1",
        to_generation="new-v1",
        config_hash=selected.snapshot_id,
    )
    store = ReleaseStore(release_sql, verifier)
    await store.install(
        generation="new-v1",
        config_hash=selected.snapshot_id,
        roles=("api",),
        receipt=receipt,
    )
    release_sql.configure(
        info={
            "context_config_hash": selected.snapshot_id,
            "context_instance_id": "new-api",
            "context_artifacts": artifacts,
        }
    )
    await store.attest("api", selected.snapshot_id)
    await store.reopen()
    async with release_sql.begin() as db:
        await db.execute(update(context_gate).values(rollback_generation="old-v1"))
        await db.execute(update(document_versions).values(index_generation="new-v1"))
    target = PublicationTarget(
        context=context,
        active_version_id="version-1",
        manifest_uri=stored.uri,
        manifest_hash=stored.sha256,
        canonical_ast_uri=ref.uri,
        canonical_ast_sha256=ref.sha256,
        parent_count=1,
        child_count=1,
    )
    await SqlAlchemyPublicationRepository(release_sql).finalize(target)
    async with release_sql() as db:
        assert (
            await db.scalar(select(retrieval_rebuild_jobs.c.status)) == "blocked_config"
        )
        assert await db.scalar(select(retrieval_builds.c.state)) == "ready"
    async with release_sql.begin() as db:
        assert await SqlAlchemyDocumentRepository(db).soft_delete(
            "document-1", UserScope(user_id="user-1")
        )
    async with release_sql() as db:
        assert await db.scalar(select(retrieval_rebuild_jobs.c.status)) == "cancelled"
        assert await db.scalar(select(retrieval_builds.c.state)) == "revoked"
    with pytest.raises(ValueError, match="deleted"):
        await RegistrySourceAuthority(release_sql, artifacts).manifest(
            "document-1",
            "version-1",
            UserScope(user_id="user-1"),
            selected,
        )


@pytest.mark.asyncio
async def test_real_structural_rebuild_executor_uses_durable_claim_and_context_service(
    release_sql, tmp_path
):
    from agentic_rag.ingestion.contextualization.builds import RebuildExecutor
    from agentic_rag.ingestion.contextualization.service import ContextualizationService
    from agentic_rag.persistence.context_builds import (
        SqlAlchemyContextBuildStore,
        context_build_observations,
    )
    from agentic_rag.persistence.context_registry import retrieval_rebuild_jobs
    from agentic_rag.runtime.models import ContextBuildConfig
    from tests.unit.ingestion.test_chunker import _tokenizer

    artifacts, ref, parent, context = await seed(release_sql, tmp_path)
    artifacts.put_json(
        f"documents/{context.user_id}/{context.document_id}/{context.document_version_id}/chunks/{context.pipeline_version}/parents.json",
        {"parents": [parent.model_dump(mode="json")]},
    )
    tokenizer = _tokenizer()
    build = ContextBuildConfig(enabled=True)
    service = ContextualizationService(
        artifacts=artifacts,
        store=SqlAlchemyContextBuildStore(release_sql),
        model=None,
        count_tokens=tokenizer.count_tokens,
    )
    await RebuildJobStore(release_sql).enqueue(
        owner="user-1",
        document="document-1",
        version="version-1",
        generation="new-v1",
        payload={
            "config_hash": build.fingerprint("0" * 64),
            "canonical_ast_sha256": ref.sha256,
        },
    )
    executor = RebuildExecutor(
        factory=release_sql,
        artifacts=artifacts,
        writer=IndexWriter(
            embedding=FakeEmbedding([[[0.0] * 1024]]),
            parent_store=RecordingParentStore([]),
            child_store=Children([]),
            artifacts=artifacts,
        ),
        tokenizer=tokenizer,
        contextualizer=service,
        build=build,
    )
    assert await executor.run_once("worker", "new-v1")
    assert not await executor.run_once("restarted", "new-v1")
    async with release_sql() as db:
        assert await db.scalar(select(retrieval_rebuild_jobs.c.status)) == "completed"
        assert (
            await db.scalar(select(retrieval_builds.c.representation_version))
            == "context-v1"
        )
        events = (
            (await db.execute(select(context_build_observations.c.event)))
            .scalars()
            .all()
        )
        assert any(
            event["stage"] == "rendered"
            and event["before_search_hash"]
            and event["after_search_hash"]
            for event in events
        )
        assert all("原始正文" not in str(event) for event in events)
