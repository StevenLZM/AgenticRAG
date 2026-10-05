"""Protocol fixtures, never a measured retrieval-quality result."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from evals.gold_v2_models import CorpusSnapshot, canonical_hash


def freeze(value):
    value = deepcopy(value)
    value.pop("snapshot_id", None)
    return CorpusSnapshot.model_validate({**value, "snapshot_id": canonical_hash(value)})


def migration_fixture():
    row = {"document_id": "d", "document_version_id": "v", "filename": "d.txt",
           "content_hash": "a" * 64, "source_path": "/test/d.txt", "index_generation": "index-v3",
           "canonical_ast_hash": "b" * 64, "canonical_ast_path": "artifact://ast.json",
           "manifest_path": "artifact://v3.json", "manifest_hash": "c" * 64,
           "parent_count": 1, "child_count": 1, "es_child_count": 1, "searchable": True,
           "parser_version": "parser", "pipeline_version": "pipeline", "embedding_version": "embedding"}
    parent = {"id": "p", "document_id": "d", "document_version_id": "v", "user_id": "u",
              "content": "事实", "status": "active", "ordinal": 0}
    child = {"child_id": "c", "parent_id": "p", "document_id": "d", "document_version_id": "v",
             "user_id": "u", "content": "事实", "index_generation": "index-v3"}
    scope = {"user_id": "u", "es_url": "http://localhost:9200", "cluster_uuid": "cluster",
             "alias": "agenticrag-children-active", "concrete_indices": ["agenticrag-children-index-v3"],
             "index_metadata": {"agenticrag-children-index-v3": {"settings": {"index": {"uuid": "old"}}}}}
    old = freeze({"schema_version": 2, "captured_at": "old", "scope": scope, "status_counts": [],
                  "documents": [row], "parents": [parent], "children": [child]})
    payload = old.model_dump()
    payload["captured_at"] = "new"
    payload["documents"][0].update(index_generation="index-v4", manifest_path="artifact://v4.json", manifest_hash="d" * 64)
    payload["children"][0]["index_generation"] = "index-v4"
    payload["scope"]["concrete_indices"] = ["agenticrag-children-index-v4"]
    payload["scope"]["index_metadata"] = {"agenticrag-children-index-v4": {"settings": {"index": {"uuid": "new"}}}}
    payload["scope"]["alias_definitions"] = {"agenticrag-children-index-v4": {}}
    target = freeze(payload)
    inventory = {"count": 1, "versions": {"v": 1}, "identities": {"v": ["u", "d"]}, "sha256": "e" * 64}
    parents = {"counts": {"v": 1}, "sha256": hashlib.sha256(json.dumps([parent], sort_keys=True, default=str).encode()).hexdigest()}
    manifest = {"index_generation": "index-v3", "canonical_ast_sha256": "b" * 64, "child_count": 1}
    journal = {"phase": "complete", "source": "index-v3", "target": "index-v4", "cluster_uuid": "cluster",
               "inventory": inventory, "parents": parents, "versions": [{"id": "v", "document_id": "d", "user_id": "u",
                   "before": {k: row[k] for k in ("index_generation", "manifest_path", "manifest_hash")},
                   "after": {k: target.documents[0][k] for k in ("index_generation", "manifest_path", "manifest_hash")},
                   "manifest": {**manifest, "index_generation": "index-v4"}}]}
    alias = {"agenticrag-children-index-v4": {"aliases": {"agenticrag-children-active": {}}}}
    evidence = {"protocol": "ik-migration-evidence-v1", "journal_sha256": canonical_hash(journal),
                "source_inventory": deepcopy(inventory), "target_inventory": deepcopy(inventory),
                "parents_inventory": deepcopy(parents), "source_children": old.children,
                "source_index_metadata": old.scope["index_metadata"],
                "originals": {"v": "a" * 64}, "asts": {"v": "b" * 64},
                "manifests": {"v": {"before": manifest, "after": {**manifest, "index_generation": "index-v4"},
                                       "before_sha256": "c" * 64, "after_sha256": "d" * 64}},
                "alias_before": alias, "alias_after": deepcopy(alias), "content_fingerprint": "f" * 64}
    return old, target, journal, evidence


def test_generation_only_is_equivalent():
    from evals.migration_evidence import verify_migration_equivalence
    result = verify_migration_equivalence(*migration_fixture())
    assert result["verified"] is True
    assert result["normalization"] == "ik-rebind-normalization-v1"


@pytest.mark.parametrize("mutation", ["content", "parent", "user", "manifest", "original", "ast", "vector", "active", "add", "remove"])
def test_same_ids_do_not_hide_mutation(mutation):
    from evals.migration_evidence import verify_migration_equivalence
    old, target, journal, evidence = migration_fixture()
    raw = target.model_dump()
    if mutation == "content":
        raw["children"][0]["content"] = "改写"
    elif mutation == "parent":
        raw["parents"][0]["ordinal"] = 8
    elif mutation == "user":
        evidence["target_inventory"]["identities"]["v"][0] = "other"
    elif mutation == "manifest":
        evidence["manifests"]["v"]["after"]["embedding_dimensions"] = 9
    elif mutation in {"original", "ast"}:
        evidence[{"original": "originals", "ast": "asts"}[mutation]]["v"] = "0" * 64
    elif mutation in {"vector", "active"}:
        evidence["target_inventory"]["sha256"] = "0" * 64
    elif mutation == "add":
        raw["documents"].append({**raw["documents"][0], "document_id": "noise", "document_version_id": "noise"})
    else:
        raw["children"] = []
    with pytest.raises(ValueError):
        verify_migration_equivalence(old, freeze(raw), journal, evidence)


def test_both_indices_changed_must_match_journal():
    from evals.migration_evidence import verify_migration_equivalence
    old, target, journal, evidence = migration_fixture()
    for name in ("source_inventory", "target_inventory"):
        evidence[name]["sha256"] = "0" * 64
    assert evidence["source_inventory"] == evidence["target_inventory"]
    with pytest.raises(ValueError, match="journal"):
        verify_migration_equivalence(old, target, journal, evidence)


class ScanES:
    def __init__(self, fault):
        self.fault, self.cleared = fault, False

    async def search(self, **kwargs):
        assert kwargs["index"] == "physical-v4"
        return SimpleNamespace(body={"_scroll_id": "s", "timed_out": False, "_shards": {"failed": int(self.fault == "shard")},
                                     "hits": {"hits": [{"_id": "c", "_source": {"id": "c"}}]}})

    async def scroll(self, **kwargs):
        if self.fault == "interrupt":
            raise RuntimeError("scan interrupted")
        return SimpleNamespace(body={"_scroll_id": "s", "timed_out": False, "_shards": {"failed": 0},
                                     "hits": {"hits": [{"_id": "c", "_source": {"id": "c"}}] if self.fault == "duplicate" else []}})

    async def clear_scroll(self, **kwargs):
        self.cleared = True


@pytest.mark.parametrize("fault", ["shard", "duplicate", "interrupt"])
async def test_capture_alias_change_or_partial_scan_rejected(fault):
    from evals.migration_evidence import scan_children
    es = ScanES(fault)
    with pytest.raises((ValueError, RuntimeError)):
        await scan_children(es, "physical-v4")
    assert es.cleared


def test_alias_drift_rejected():
    from evals.migration_evidence import verify_migration_equivalence
    old, target, journal, evidence = migration_fixture()
    evidence["alias_after"] = {}
    with pytest.raises(ValueError, match="alias"):
        verify_migration_equivalence(old, target, journal, evidence)


def test_content_drift_checks_evidence_not_only_snapshot():
    from evals.migration_evidence import assert_corpus_unchanged
    before = {"content_fingerprint": "a", "observed_at": "yesterday"}
    assert_corpus_unchanged(before, {**before, "observed_at": "today"})
    with pytest.raises(ValueError, match="drift"):
        assert_corpus_unchanged(before, {"content_fingerprint": "b"})


@pytest.fixture
async def live_fixture(tmp_path):
    from datetime import datetime
    from sqlalchemy import insert
    from sqlalchemy.ext.asyncio import create_async_engine
    from agentic_rag.ingestion.manifest import VersionManifest
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from agentic_rag.persistence.repositories import metadata, documents, document_versions, parent_chunks
    from scripts.migrate_ik_index import plan_version
    root = tmp_path / "artifacts"
    artifacts = LocalArtifactStore(root)
    source = artifacts.put_bytes("documents/u/d/v/source/d.txt", "事实".encode())
    ast = artifacts.put_json("ast.json", {"text": "事实"})
    manifest = VersionManifest(canonical_ast_sha256=ast.sha256, child_count=1, parent_count=1,
                               embedding_model="text-embedding-v3", embedding_dimensions=1024, index_generation="index-v3")
    before = artifacts.put_json("v3.json", manifest.payload())
    plan = plan_version({"id": "v", "document_id": "d", "user_id": "u", "status": "active",
                         "index_generation": "index-v3", "manifest_path": before.uri, "manifest_hash": before.sha256,
                         "canonical_ast_hash": ast.sha256, "child_count": 1, "parent_count": 1}, manifest,
                        source="index-v3", target="index-v4")
    artifacts.put_json(plan["after"]["manifest_path"].removeprefix("artifact://"), plan["manifest"])
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    now = datetime(2026, 10, 1)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.execute(insert(documents).values(id="d", user_id="u", source_type="upload", filename="d.txt",
            mime_type="text/plain", content_hash=source.sha256, status="active", active_version_id="v", created_at=now, updated_at=now))
        await conn.execute(insert(document_versions).values(id="v", document_id="d", version_no=1, parser_version="parser",
            pipeline_version="pipeline", embedding_version="embedding", status="active", canonical_ast_path=ast.uri,
            canonical_ast_hash=ast.sha256, parent_count=1, child_count=1, created_at=now, **plan["after"]))
        await conn.execute(insert(parent_chunks).values(id="p", user_id="u", document_id="d", document_version_id="v",
            ordinal=0, heading_path=[], content_type="paragraph", content="事实", ast_locator="{}",
            content_hash=source.sha256, status="active"))
    old, _, _, _ = migration_fixture()
    payload = old.model_dump()
    payload["documents"][0].update(content_hash=source.sha256, canonical_ast_hash=ast.sha256,
        manifest_hash=before.sha256, source_path=str(root / "documents/u/d/v/source/d.txt"))

    class LiveES:
        def __init__(self):
            self.indices = self
            self.cleared = 0
            self.calls = []

        async def info(self):
            return SimpleNamespace(body={"cluster_uuid": "cluster"})

        async def get_alias(self, **kwargs):
            return SimpleNamespace(body={"agenticrag-children-index-v4": {"aliases": {"agenticrag-children-active": {}}}})

        async def get_settings(self, *, index, **kwargs):
            return SimpleNamespace(body={index: {"settings": {"index": {"uuid": "old" if index.endswith("v3") else "new"}}}})

        async def search(self, *, index, **kwargs):
            self.calls.append(index)
            value = {**old.children[0], "id": "c", "is_active": True, "search_type": "document", "embedding": [0.25],
                     "index_generation": "index-v3" if index.endswith("v3") else "index-v4"}
            value.pop("child_id")
            return SimpleNamespace(body={"_scroll_id": "s", "timed_out": False, "_shards": {"failed": 0},
                "hits": {"total": {"value": 1, "relation": "eq"}, "hits": [{"_id": "c", "_source": value}]}})

        async def scroll(self, **kwargs):
            return SimpleNamespace(body={"_scroll_id": "s", "timed_out": False, "_shards": {"failed": 0}, "hits": {"hits": []}})

        async def clear_scroll(self, **kwargs):
            self.cleared += 1

    try:
        yield freeze(payload), plan, engine, artifacts, LiveES()
    finally:
        await engine.dispose()


async def test_capture_reads_full_sql_source_and_vectors(live_fixture):
    from evals.migration_evidence import capture_corpus_evidence, capture_migration_evidence
    old, plan, engine, artifacts, es = live_fixture
    snapshot, evidence = await capture_corpus_evidence(old, engine=engine, es=es, artifacts=artifacts,
        index="agenticrag-children-index-v4", generation="index-v4")
    assert snapshot.children[0]["content"] == "事实"
    assert snapshot.scope["alias_definitions"] == {"agenticrag-children-index-v4": {}}
    assert evidence["child_hashes"].keys() == {"c"}
    assert evidence["originals"]["v"] == hashlib.sha256("事实".encode()).hexdigest()
    assert evidence["parents_inventory"]["counts"] == {"v": 1}
    journal = {"phase": "complete", "cluster_uuid": "cluster", "source": "index-v3", "target": "index-v4",
               "versions": [plan], "inventory": evidence["target_inventory"], "parents": evidence["parents_inventory"]}
    payload = old.model_dump()
    payload["parents"], payload["status_counts"] = snapshot.parents, snapshot.status_counts
    # Use exactly the old exporter schema, including document fields omitted by the small fixture.
    payload["documents"] = [{**snapshot.documents[0], **plan["before"]}]
    old = freeze(payload)
    target, proof = await capture_migration_evidence(old, journal, engine=engine, es=es, artifacts=artifacts)
    assert target.documents[0]["index_generation"] == "index-v4"
    assert proof["source_inventory"] == proof["target_inventory"] == journal["inventory"]
    assert es.cleared == len(es.calls)


async def test_capture_rejects_original_or_sql_drift(live_fixture):
    from evals.migration_evidence import capture_corpus_evidence
    old, _, engine, artifacts, es = live_fixture
    from pathlib import Path
    Path(old.documents[0]["source_path"]).write_text("篡改")
    with pytest.raises(ValueError, match="original"):
        await capture_corpus_evidence(old, engine=engine, es=es, artifacts=artifacts,
            index="agenticrag-children-index-v4", generation="index-v4")
