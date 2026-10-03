"""Opt-in IK tests. The URL must point to a disposable, IK-enabled ES node."""
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from elasticsearch import AsyncElasticsearch
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.elasticsearch import ChildIndexMappingError, ElasticsearchChildIndexStore
from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
from agentic_rag.persistence.repositories import documents, document_versions, metadata, parent_chunks
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
from agentic_rag.retrieval.adapters.elasticsearch import ElasticsearchBm25Index
from agentic_rag.retrieval.models import SearchFilter
from scripts.migrate_ik_index import (
    MigrationError, execute_migration, inventory, preflight, rollback_migration,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def ik_es():
    url = os.getenv("AGENTIC_RAG_TEST_IK_ELASTICSEARCH_URL")
    if not url:
        pytest.skip("set AGENTIC_RAG_TEST_IK_ELASTICSEARCH_URL to a disposable IK node")
    async with AsyncElasticsearch(url) as es:
        await es.indices.analyze(analyzer="ik_smart", text="京东")
        yield es


async def test_ik_words_ranking_identifiers_and_user_filters(ik_es):
    generation = "ik-test-" + uuid4().hex[:12]
    index = ElasticsearchChildIndexStore.index_name(generation)
    store = ElasticsearchChildIndexStore(ik_es, lexical_analysis="ik")
    try:
        await store.ensure_index(generation)
        terms = ["京东", "回龙观", "王府井", "Redis", "Elasticsearch", "刘泽明", "星河智采", "2024.06"]
        for ordinal, term in enumerate([*terms, "北京东方", "京东"]):
            record = {"id": str(ordinal), "parent_id": "parent-" + str(ordinal), "document_id": "doc",
                      "document_version_id": "version", "user_id": "other" if ordinal == 9 else "u1",
                      "content": term, "contextualized_content": term,
                      "index_generation": generation, "is_active": True, "search_type": "document"}
            await ik_es.index(index=index, id=str(ordinal), document=record)
        await ik_es.indices.refresh(index=index)
        for term in terms[:3]:
            tokens = (await ik_es.indices.analyze(index=index, analyzer="ik_smart", text=term))["tokens"]
            assert term in [t["token"] for t in tokens]
        adapter = ElasticsearchBm25Index(ik_es, index=index, index_generation=generation, lexical_analysis="ik")
        for ordinal, term in enumerate(terms):
            hits = await adapter.search(term, SearchFilter(user_id="u1", index_generation=generation), 10)
            assert hits[0].child_id == str(ordinal)
            assert all(hit.user_id == "u1" and hit.lane == "bm25" for hit in hits)
        assert not await adapter.search("京东", SearchFilter(user_id="u1", index_generation=generation,
                                                              document_ids=("unrelated",)), 10)
    finally:
        await ik_es.indices.delete(index=index, ignore_unavailable=True)


@pytest.fixture
async def migration_fixture(ik_es, tmp_path):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    source, target = "ik-src-" + uuid4().hex[:12], "ik-dst-" + uuid4().hex[:12]
    source_index, target_index = [ElasticsearchChildIndexStore.index_name(g) for g in (source, target)]
    source_store = ElasticsearchChildIndexStore(ik_es)
    now = datetime.now(timezone.utc)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await conn.execute(insert(documents).values(id="doc", user_id="u1", source_type="upload",
                filename="fixture.txt", mime_type="text/plain", content_hash="a" * 64,
                status="active", active_version_id="version", created_at=now, updated_at=now))
            canonical = artifacts.put_json("documents/u1/doc/version/canonical/ast.json", {"text": "京东工作经历"})
            manifest = VersionManifest(canonical_ast_sha256=canonical.sha256, parent_count=1, child_count=1,
                embedding_model="text-embedding-v3", embedding_dimensions=1024, index_generation=source)
            ref = artifacts.put_json(
                f"documents/u1/doc/version/manifests/{source}/{manifest.manifest_hash}.json", manifest.payload())
            await conn.execute(insert(document_versions).values(id="version", document_id="doc", version_no=1,
                parser_version="docling-v1", pipeline_version="ingestion-v2", embedding_version="text-embedding-v3",
                index_generation=source, status="active", parent_count=1, child_count=1,
                canonical_ast_path=canonical.uri, canonical_ast_hash=canonical.sha256,
                manifest_path=ref.uri, manifest_hash=ref.sha256, created_at=now))
            await conn.execute(insert(parent_chunks).values(id="parent", user_id="u1", document_id="doc",
                document_version_id="version", ordinal=0, heading_path=["经历"], content_type="paragraph",
                content="京东工作经历", ast_locator="{}", content_hash="a" * 64, status="active"))
        await source_store.ensure_index(source)
        await ik_es.index(index=source_index, id="child", document={"id": "child", "parent_id": "parent",
            "user_id": "u1", "document_id": "doc", "document_version_id": "version", "is_active": True,
            "search_type": "document", "index_generation": source, "content": "京东工作经历",
            "contextualized_content": "经历\n京东工作经历", "embedding": [0.125] * 1024}, refresh=True)
        await source_store.ensure_active_alias(source)
        yield engine, artifacts, source, target, tmp_path
    finally:
        for index in (source_index, target_index):
            await ik_es.indices.delete(index=index, ignore_unavailable=True)
        await engine.dispose()


async def test_real_migration_preserves_vectors_parents_and_supports_rollback(ik_es, migration_fixture):
    engine, artifacts, source, target, root = migration_fixture
    plan = await preflight(engine, ik_es, artifacts, source=source, target=target)
    journal = LocalArtifactStore(root / "journal")
    await execute_migration(engine, ik_es, artifacts, journal, plan)
    assert plan["phase"] == "complete"
    assert await inventory(ik_es, ElasticsearchChildIndexStore.index_name(target), target) == plan["inventory"]
    async with engine.connect() as conn:
        row = (await conn.execute(select(document_versions))).mappings().one()
        assert row["index_generation"] == target
        assert artifacts.describe(row["manifest_path"]).sha256 == row["manifest_hash"]
        assert (await conn.execute(select(parent_chunks.c.content))).scalar_one() == "京东工作经历"
    assert set((await ik_es.indices.get_alias(name="agenticrag-children-active")).body) == {
        ElasticsearchChildIndexStore.index_name(target)}
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyPublicationRepository(sessions)
    publication = await repository.get_target("version")
    assert publication is not None
    await VersionPublisher(repository=repository, parent_store=SqlAlchemyParentStagingStore(sessions),
        child_store=ElasticsearchChildIndexStore(ik_es, lexical_analysis="ik"), artifacts=artifacts)._verify(publication)
    await rollback_migration(engine, ik_es, journal, plan)
    await rollback_migration(engine, ik_es, journal, plan)
    async with engine.connect() as conn:
        assert (await conn.execute(select(document_versions.c.index_generation))).scalar_one() == source
    assert await ik_es.indices.exists(index=ElasticsearchChildIndexStore.index_name(target))


async def test_preflight_rejects_missing_children_without_mutation(ik_es, migration_fixture):
    engine, artifacts, source, target, _ = migration_fixture
    async with engine.begin() as conn:
        await conn.execute(update(document_versions).values(child_count=2))
    with pytest.raises(MigrationError, match="inventory"):
        await preflight(engine, ik_es, artifacts, source=source, target=target)
    assert not await ik_es.indices.exists(index=ElasticsearchChildIndexStore.index_name(target))


async def test_rollback_refuses_new_target_writes(ik_es, migration_fixture):
    engine, artifacts, source, target, root = migration_fixture
    plan = await preflight(engine, ik_es, artifacts, source=source, target=target)
    journal = LocalArtifactStore(root / "journal")
    await execute_migration(engine, ik_es, artifacts, journal, plan)
    target_index = ElasticsearchChildIndexStore.index_name(target)
    await ik_es.update(index=target_index, id="child", doc={"content": "new content"}, refresh=True)
    with pytest.raises(MigrationError, match="new writes"):
        await rollback_migration(engine, ik_es, journal, plan)
    async with engine.connect() as conn:
        assert (await conn.execute(select(document_versions.c.index_generation))).scalar_one() == target


async def test_preflight_rejects_incompatible_source_mapping(ik_es, migration_fixture):
    engine, artifacts, source, target, _ = migration_fixture
    await ik_es.indices.put_mapping(index=ElasticsearchChildIndexStore.index_name(source),
                                    meta={"schema_version": 999})
    with pytest.raises(ChildIndexMappingError, match="incompatible"):
        await preflight(engine, ik_es, artifacts, source=source, target=target)
    assert not await ik_es.indices.exists(index=ElasticsearchChildIndexStore.index_name(target))


async def test_rollback_validates_source_mapping_before_sql_commit(ik_es, migration_fixture):
    engine, artifacts, source, target, root = migration_fixture
    plan = await preflight(engine, ik_es, artifacts, source=source, target=target)
    journal = LocalArtifactStore(root / "journal")
    await execute_migration(engine, ik_es, artifacts, journal, plan)
    await ik_es.indices.put_mapping(index=ElasticsearchChildIndexStore.index_name(source),
                                    meta={"schema_version": 999})
    with pytest.raises(ChildIndexMappingError, match="incompatible"):
        await rollback_migration(engine, ik_es, journal, plan)
    async with engine.connect() as conn:
        assert (await conn.execute(select(document_versions.c.index_generation))).scalar_one() == target
    assert set((await ik_es.indices.get_alias(name="agenticrag-children-active")).body) == {
        ElasticsearchChildIndexStore.index_name(target)}


@pytest.mark.parametrize("failure_window", ["write_blocked", "copied", "metadata_committed", "alias_switched"])
async def test_interrupted_migration_rolls_back_from_durable_journal(
        ik_es, migration_fixture, monkeypatch, failure_window):
    engine, artifacts, source, target, root = migration_fixture
    plan = await preflight(engine, ik_es, artifacts, source=source, target=target)
    journal = LocalArtifactStore(root / "journal")
    original_switch = ElasticsearchChildIndexStore.switch_active_alias
    original_save = journal.put_json

    async def fail_after_block(*args, **kwargs):
        raise RuntimeError("injected interruption")

    def save_then_fail(key, value):
        result = original_save(key, value)
        if value["phase"] == failure_window:
            raise RuntimeError("injected interruption")
        return result

    async def switch_then_fail(store, generation):
        await original_switch(store, generation)
        raise RuntimeError("injected interruption")

    with monkeypatch.context() as patch:
        if failure_window == "write_blocked":
            patch.setattr("scripts.migrate_ik_index.inventory", fail_after_block)
        elif failure_window == "alias_switched":
            patch.setattr(ElasticsearchChildIndexStore, "switch_active_alias", switch_then_fail)
        else:
            patch.setattr(journal, "put_json", save_then_fail)
        with pytest.raises(RuntimeError, match="injected"):
            await execute_migration(engine, ik_es, artifacts, journal, plan)
    # A new process sees only the durable journal, not the mutated in-memory plan.
    durable = journal.read_json(journal.describe("artifact://journal.json"))
    await rollback_migration(engine, ik_es, journal, durable)
    await rollback_migration(engine, ik_es, journal, durable)
    source_index = ElasticsearchChildIndexStore.index_name(source)
    async with engine.connect() as conn:
        row = (await conn.execute(select(document_versions))).mappings().one()
        assert all(row[key] == value for key, value in durable["versions"][0]["before"].items())
    assert set((await ik_es.indices.get_alias(name="agenticrag-children-active")).body) == {source_index}
    assert await inventory(ik_es, source_index, source) == durable["inventory"]
    settings = (await ik_es.indices.get_settings(index=source_index)).body
    assert settings[source_index]["settings"]["index"]["blocks"]["write"] == "false"
