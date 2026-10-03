"""Migration must preserve immutable content and atomically update its metadata."""
from datetime import datetime, timezone

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import create_async_engine

from agentic_rag.ingestion.manifest import VersionManifest
from agentic_rag.persistence.repositories import document_versions, metadata


def fixture_version():
    manifest = VersionManifest(canonical_ast_sha256="a" * 64, parent_count=2, child_count=3,
                               embedding_model="text-embedding-v3", embedding_dimensions=1024,
                               index_generation="index-v3")
    row = {"id": "version-1", "document_id": "doc-1", "user_id": "user-1", "status": "active",
           "index_generation": "index-v3", "manifest_path": "artifact://documents/user-1/doc-1/manifest.json",
           "manifest_hash": manifest.manifest_hash, "canonical_ast_hash": "a" * 64,
           "parent_count": 2, "child_count": 3}
    return row, manifest


def test_migration_rebinds_manifest_without_changing_content_or_counts():
    from scripts.migrate_ik_index import plan_version
    row, manifest = fixture_version()
    plan = plan_version(row, manifest, source="index-v3", target="index-v4")
    assert plan["before"]["manifest_hash"] == manifest.manifest_hash
    assert plan["after"]["index_generation"] == "index-v4"
    assert plan["after"]["manifest_path"] != row["manifest_path"]
    assert plan["after"]["manifest_path"] == (
        "artifact://documents/user-1/doc-1/version-1/manifests/index-v4/"
        + plan["after"]["manifest_hash"] + ".json"
    )
    assert plan["after"]["manifest_hash"] != row["manifest_hash"]
    assert plan["manifest"] == {**manifest.payload(), "index_generation": "index-v4"}


@pytest.mark.parametrize("changes", [{"manifest_hash": "0" * 64}, {"child_count": 4},
    {"canonical_ast_hash": "b" * 64}, {"status": "inactive"}, {"index_generation": "other"}])
def test_migration_rejects_unverified_or_nonactive_versions(changes):
    from scripts.migrate_ik_index import MigrationError, plan_version
    row, manifest = fixture_version()
    with pytest.raises(MigrationError):
        plan_version({**row, **changes}, manifest, source="index-v3", target="index-v4")


@pytest.mark.parametrize("target", ["index-v3", "*", "../index-v4", ""])
def test_migration_never_overwrites_source_or_accepts_broad_targets(target):
    from scripts.migrate_ik_index import MigrationError, plan_version
    row, manifest = fixture_version()
    with pytest.raises(MigrationError):
        plan_version(row, manifest, source="index-v3", target=target)


async def test_metadata_switch_and_rollback_are_atomic_and_idempotent():
    from scripts.migrate_ik_index import MigrationError, apply_metadata, plan_version
    row, manifest = fixture_version()
    plans = [plan_version({**row, "id": f"version-{i}"}, manifest, source="index-v3", target="index-v4")
             for i in (1, 2)]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            for i in (1, 2):
                await conn.execute(insert(document_versions).values(id=f"version-{i}", document_id="doc-1",
                    version_no=i, parser_version="docling-v1", pipeline_version="ingestion-v2",
                    embedding_version="text-embedding-v3", status="active", parent_count=2, child_count=3,
                    created_at=datetime.now(timezone.utc), **plans[i - 1]["before"]))
        async with engine.begin() as conn:
            await apply_metadata(conn, plans)
            await apply_metadata(conn, plans)
        async with engine.connect() as conn:
            assert set((await conn.execute(select(document_versions.c.index_generation))).scalars()) == {"index-v4"}
        async with engine.begin() as conn:
            await apply_metadata(conn, plans, rollback=True)
            await apply_metadata(conn, plans, rollback=True)
        broken = [plans[0], {**plans[1], "before": {**plans[1]["before"], "manifest_hash": "0" * 64}}]
        with pytest.raises(MigrationError):
            async with engine.begin() as conn:
                await apply_metadata(conn, broken)
        async with engine.connect() as conn:
            assert set((await conn.execute(select(document_versions.c.index_generation))).scalars()) == {"index-v3"}
    finally:
        await engine.dispose()
