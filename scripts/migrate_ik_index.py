"""Offline, journaled standard -> IK migration. Never delete either index.

Stop API and workers first. Default is read-only preflight. Apply requires a
verified backup; after an interrupted apply use rollback before restarting.
Only active versions are migrated. Unexpected/inactive ES records fail closed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_scan
from sqlalchemy import func, select, update

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.ingestion.manifest import VersionManifest  # noqa: E402
from agentic_rag.models.indexing import ACTIVE_CHILD_INDEX_ALIAS, validate_index_generation  # noqa: E402
from agentic_rag.persistence.artifacts import LocalArtifactStore  # noqa: E402
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore  # noqa: E402
from agentic_rag.persistence.mysql import create_mysql_engine  # noqa: E402
from agentic_rag.persistence.repositories import (  # noqa: E402
    agent_runs, documents, document_versions, ingestion_jobs, parent_chunks, task_outbox,
)


class MigrationError(RuntimeError):
    """A migration safety invariant was not established."""


def _generations(source, target):
    try:
        validate_index_generation(source)
        validate_index_generation(target)
    except ValueError as error:
        raise MigrationError("invalid exact index generation") from error
    if source == target:
        raise MigrationError("source and target must be different")


def plan_version(row, manifest, *, source, target):
    _generations(source, target)
    if (row["status"] != "active" or row["index_generation"] != source
            or manifest.index_generation != source or row["manifest_hash"] != manifest.manifest_hash
            or row["canonical_ast_hash"] != manifest.canonical_ast_sha256
            or row["parent_count"] != manifest.parent_count or row["child_count"] != manifest.child_count):
        raise MigrationError("version/manifest integrity mismatch")
    new = manifest.model_copy(update={"index_generation": target})
    before = {key: row[key] for key in ("index_generation", "manifest_path", "manifest_hash")}
    return {"id": row["id"], "document_id": row["document_id"], "user_id": row["user_id"],
            "before": before, "after": {"index_generation": target,
                "manifest_path": (f"artifact://documents/{row['user_id']}/{row['document_id']}/"
                                  f"{row['id']}/manifests/{target}/{new.manifest_hash}.json"),
                "manifest_hash": new.manifest_hash}, "manifest": new.payload()}


async def apply_metadata(conn, plans, *, rollback=False):
    """Caller owns a single transaction; CAS every row, including on rollback."""
    for plan in plans:
        before, after = (plan["after"], plan["before"]) if rollback else (plan["before"], plan["after"])
        row = (await conn.execute(select(document_versions).where(
            document_versions.c.id == plan["id"], document_versions.c.document_id == plan["document_id"],
            document_versions.c.status == "active").with_for_update())).mappings().one_or_none()
        if row is None:
            raise MigrationError("version changed while offline")
        current = {key: row[key] for key in before}
        if current == after:
            continue
        if current != before:
            raise MigrationError("version metadata changed while offline")
        result = await conn.execute(update(document_versions).where(
            document_versions.c.id == plan["id"],
            *(document_versions.c[key] == value for key, value in before.items())).values(**after))
        if result.rowcount != 1:
            raise MigrationError("version compare-and-swap failed")


async def require_quiescent(conn):
    for table, condition in (
        (agent_runs, agent_runs.c.status.in_(("queued", "running", "cancel_requested"))),
        (ingestion_jobs, ingestion_jobs.c.status.in_(("queued", "running"))),
        (task_outbox, task_outbox.c.status == "pending"),
        (documents, documents.c.deletion_status.in_(("pending", "fenced"))),
    ):
        if await conn.scalar(select(func.count()).select_from(table).where(condition)):
            raise MigrationError(f"{table.name} is not quiescent; drain work first")


async def inventory(es, index, generation):
    hashes, versions, identities = {}, Counter(), {}
    async for hit in async_scan(es, index=index, query={"query": {"match_all": {}}}, size=500):
        value = dict(hit["_source"])
        if (value.pop("index_generation") != generation or value["is_active"] is not True
                or value["search_type"] != "document" or value["id"] != hit["_id"]):
            raise MigrationError("unexpected index record; requires a separate migration review")
        version = value["document_version_id"]
        identity = [value["user_id"], value["document_id"]]
        if version in identities and identities[version] != identity:
            raise MigrationError("version crosses user/document scope")
        identities[version] = identity
        versions[version] += 1
        hashes[hit["_id"]] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"count": len(hashes), "versions": dict(versions), "identities": identities,
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


async def parent_inventory(conn, version_ids):
    rows = (await conn.execute(select(parent_chunks).where(
        parent_chunks.c.document_version_id.in_(version_ids)).order_by(parent_chunks.c.id))).mappings().all()
    counts = Counter(row["document_version_id"] for row in rows if row["status"] == "active")
    return {"counts": dict(counts), "sha256": hashlib.sha256(
        json.dumps([dict(row) for row in rows], sort_keys=True, default=str).encode()).hexdigest()}


async def preflight(engine, es, artifacts, *, source, target):
    _generations(source, target)
    source_index = ElasticsearchChildIndexStore.index_name(source)
    target_index = ElasticsearchChildIndexStore.index_name(target)
    alias = await es.indices.get_alias(name=ACTIVE_CHILD_INDEX_ALIAS)
    if set(alias.body) != {source_index}:
        raise MigrationError("active alias is not the exact source index")
    if await es.indices.exists(index=target_index):
        raise MigrationError("target already exists; do not overwrite it")
    # Establish that the retained source is a valid standard-profile rollback target.
    await ElasticsearchChildIndexStore(es).validate_index(source)
    async with engine.connect() as conn:
        await require_quiescent(conn)
        rows = (await conn.execute(select(document_versions, documents.c.user_id).join(
            documents, documents.c.active_version_id == document_versions.c.id).where(
            documents.c.status == "active", document_versions.c.status == "active",
            document_versions.c.index_generation == source).order_by(document_versions.c.id))).mappings().all()
        parents = await parent_inventory(conn, [row["id"] for row in rows])
    source_data = await inventory(es, source_index, source)
    if not rows or source_data["versions"] != {row["id"]: row["child_count"] for row in rows}:
        raise MigrationError("SQL/ES version inventory or counts disagree")
    plans = []
    for row in rows:
        if (source_data["identities"][row["id"]] != [row["user_id"], row["document_id"]]
                or parents["counts"].get(row["id"]) != row["parent_count"]):
            raise MigrationError("parent counts or child scope disagree")
        ref = artifacts.describe(row["manifest_path"])
        if ref.sha256 != row["manifest_hash"] or not artifacts.verify_hash(
                row["canonical_ast_path"], row["canonical_ast_hash"]):
            raise MigrationError("artifact integrity check failed")
        manifest = VersionManifest.model_validate(artifacts.read_json(ref))
        plans.append(plan_version(row, manifest, source=source, target=target))
    cluster = (await es.info()).body
    return {"format_version": 1, "source": source, "target": target,
            "cluster_uuid": cluster["cluster_uuid"], "phase": "prepared", "versions": plans,
            "inventory": source_data, "parents": parents}


async def execute_migration(engine, es, artifacts, journal, plan):
    """Offline cutover; journal is written before the first external mutation."""
    source_index = ElasticsearchChildIndexStore.index_name(plan["source"])
    target_index = ElasticsearchChildIndexStore.index_name(plan["target"])
    journal.put_json("journal.json", plan)
    await es.indices.add_block(index=source_index, block="write")
    if await inventory(es, source_index, plan["source"]) != plan["inventory"]:
        raise MigrationError("source changed after preflight; keep services stopped")
    target_store = ElasticsearchChildIndexStore(es, lexical_analysis="ik")
    await target_store.ensure_index(plan["target"])
    result = (await es.options(request_timeout=180).reindex(
        source={"index": source_index}, dest={"index": target_index, "op_type": "create"},
        script={"lang": "painless", "source": "ctx._source.index_generation = params.generation",
                "params": {"generation": plan["target"]}}, refresh=True, wait_for_completion=True)).body
    if (result.get("failures") or result.get("timed_out") or result.get("version_conflicts")
            or result.get("created") != plan["inventory"]["count"]):
        raise MigrationError("reindex was incomplete; keep services stopped")
    if await inventory(es, target_index, plan["target"]) != plan["inventory"]:
        raise MigrationError("target differs from source content/vectors")
    for item in plan["versions"]:
        uri = item["after"]["manifest_path"]
        try:
            existing = artifacts.describe(uri)
        except FileNotFoundError:
            existing = None
        if existing and existing.sha256 != item["after"]["manifest_hash"]:
            raise MigrationError("refusing to replace a different manifest")
        ref = artifacts.put_json(uri.removeprefix("artifact://"), item["manifest"])
        if ref.sha256 != item["after"]["manifest_hash"]:
            raise MigrationError("new manifest hash mismatch")
    plan["phase"] = "copied"
    journal.put_json("journal.json", plan)
    async with engine.begin() as conn:
        await require_quiescent(conn)
        if await parent_inventory(conn, [v["id"] for v in plan["versions"]]) != plan["parents"]:
            raise MigrationError("parents changed after preflight")
        await apply_metadata(conn, plan["versions"])
    plan["phase"] = "metadata_committed"
    journal.put_json("journal.json", plan)
    await target_store.switch_active_alias(plan["target"])
    await es.indices.put_settings(index=source_index, settings={"index.blocks.write": False})
    plan["phase"] = "complete"
    journal.put_json("journal.json", plan)


async def rollback_migration(engine, es, journal, plan):
    """Immediate rollback only: refuse after content or active versions change."""
    source_index = ElasticsearchChildIndexStore.index_name(plan["source"])
    target_index = ElasticsearchChildIndexStore.index_name(plan["target"])
    if (await es.info()).body["cluster_uuid"] != plan["cluster_uuid"]:
        raise MigrationError("journal belongs to another cluster")
    alias = set((await es.indices.get_alias(name=ACTIVE_CHILD_INDEX_ALIAS)).body)
    if alias not in ({source_index}, {target_index}):
        raise MigrationError("alias changed outside this migration")
    source_store = ElasticsearchChildIndexStore(es)
    # Validate before reverting SQL: an incompatible source cannot receive the alias.
    await source_store.validate_index(plan["source"])
    if await inventory(es, source_index, plan["source"]) != plan["inventory"]:
        raise MigrationError("source changed; restore needs separate review")
    if alias == {target_index} and await inventory(es, target_index, plan["target"]) != plan["inventory"]:
        raise MigrationError("target has new writes; do not discard them")
    async with engine.begin() as conn:
        await require_quiescent(conn)
        if await parent_inventory(conn, [v["id"] for v in plan["versions"]]) != plan["parents"]:
            raise MigrationError("parents changed; do not roll back")
        await apply_metadata(conn, plan["versions"], rollback=True)
    await source_store.switch_active_alias(plan["source"])
    await es.indices.put_settings(index=source_index, settings={"index.blocks.write": False})
    plan["phase"] = "rolled_back"
    journal.put_json("journal.json", plan)


async def main(args):
    settings = Settings()
    engine = create_mysql_engine(settings.mysql_dsn)
    es = AsyncElasticsearch(settings.elasticsearch_url)
    artifacts = LocalArtifactStore(settings.artifact_root)
    try:
        if args.apply or args.rollback:
            if not args.offline_confirmed or args.journal is None:
                raise MigrationError("mutations require --offline-confirmed and --journal")
        if args.rollback:
            journal = LocalArtifactStore(args.journal)
            plan = journal.read_json(journal.describe("artifact://journal.json"))
            if (args.source, args.target) != (plan["source"], plan["target"]):
                raise MigrationError("journal generations do not match arguments")
            await rollback_migration(engine, es, journal, plan)
        else:
            plan = await preflight(engine, es, artifacts, source=args.source, target=args.target)
            if args.apply:
                from scripts.restore_local import _verify_backup
                if args.backup is None:
                    raise MigrationError("apply requires --backup")
                backup = _verify_backup(args.backup)
                if backup["index_generation"] != args.source:
                    raise MigrationError("backup generation does not match source")
                if not all((args.backup / p).is_file() for p in ("mysql/dump.sql", "elasticsearch/export.json")):
                    raise MigrationError("backup must include MySQL and ES")
                if args.journal.exists():
                    raise MigrationError("journal path already exists; use rollback after interrupted apply")
                plan["backup"] = str(args.backup.resolve())
                await execute_migration(engine, es, artifacts, LocalArtifactStore(args.journal), plan)
        print(json.dumps({"phase": plan["phase"], "source": plan["source"], "target": plan["target"],
                          "versions": len(plan["versions"]), "children": plan["inventory"]["count"],
                          "parents": sum(plan["parents"]["counts"].values())}))
    finally:
        await es.close()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="index-v3")
    parser.add_argument("--target", default="index-v4")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--apply", action="store_true")
    actions.add_argument("--rollback", action="store_true")
    parser.add_argument("--offline-confirmed", action="store_true")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--backup", type=Path)
    asyncio.run(main(parser.parse_args()))
