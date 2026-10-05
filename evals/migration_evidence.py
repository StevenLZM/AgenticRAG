"""Read-only equivalence proofs. Counts alone never authorize a gold rebind."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from agentic_rag.persistence.repositories import documents, document_versions, parent_chunks
from evals.gold_v2_models import CorpusSnapshot, canonical_hash

PROTOCOL = "ik-migration-evidence-v1"
NORMALIZATION = "ik-rebind-normalization-v1"
VERSION_FIELDS = ("index_generation", "manifest_path", "manifest_hash")


def _require(condition, reason):
    if not condition:
        raise ValueError("migration evidence: " + reason)


def _ordered(rows, key):
    return sorted(rows, key=lambda row: row[key])


def _without(value, keys):
    return {k: v for k, v in value.items() if k not in keys}


def parent_inventory(rows):
    """Match migration journal serialization, including inactive effective rows."""
    rows = _ordered(rows, "id")
    return {"counts": dict(Counter(r["document_version_id"] for r in rows if r["status"] == "active")),
            "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()}


def child_inventory(rows, generation):
    hashes, counts, identities = {}, Counter(), {}
    for item in rows:
        value = deepcopy(item)
        key = value.pop("_id")
        _require(key not in hashes and value["id"] == key, "duplicate/mismatched Child id")
        _require(value.pop("index_generation") == generation and value["is_active"] is True
                 and value["search_type"] == "document", "Child publication/generation")
        version, identity = value["document_version_id"], [value["user_id"], value["document_id"]]
        _require(version not in identities or identities[version] == identity, "Child scope")
        identities[version] = identity
        counts[version] += 1
        hashes[key] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"count": len(hashes), "versions": dict(counts), "identities": identities,
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}, hashes


async def scan_children(es, index: str) -> list[dict[str, Any]]:
    """Fixed physical index scan with strict partial-result detection and cleanup."""
    rows, seen, scroll = [], set(), None
    try:
        response = await es.search(index=index, query={"match_all": {}}, size=500, sort=["_doc"],
                                   scroll="2m", track_total_hits=True)
        expected = None
        while True:
            body = response.body
            scroll = body.get("_scroll_id", scroll)
            _require(not body.get("timed_out") and body.get("_shards", {}).get("failed") == 0, "partial scan")
            total = body["hits"].get("total")
            if expected is None and total is not None:
                _require(total["relation"] == "eq", "inexact scan total")
                expected = total["value"]
            hits = body["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                _require(hit["_id"] not in seen, "duplicate scan Child id")
                seen.add(hit["_id"])
                rows.append({**hit["_source"], "_id": hit["_id"]})
            _require(bool(scroll), "missing scroll")
            response = await es.scroll(scroll_id=scroll, scroll="2m")
        _require(expected is None or expected == len(rows), "incomplete scan")
        return rows
    finally:
        if scroll:
            await es.clear_scroll(scroll_id=scroll)


def assert_corpus_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    _require(bool(before.get("content_fingerprint")) and
             before["content_fingerprint"] == after.get("content_fingerprint"), "corpus drift")


def verify_migration_equivalence(old: CorpusSnapshot, target: CorpusSnapshot,
                                 journal: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    _require(journal["phase"] == "complete" and evidence["protocol"] == PROTOCOL, "incomplete journal/proof")
    _require(evidence["journal_sha256"] == canonical_hash(journal), "journal binding")
    source_index, target_index = ["agenticrag-children-" + journal[k] for k in ("source", "target")]
    _require(journal["source"] != journal["target"], "identical generations")
    _require(old.scope["concrete_indices"] == [source_index] and target.scope["concrete_indices"] == [target_index], "index scope")
    _require(old.scope["cluster_uuid"] == target.scope["cluster_uuid"] == journal["cluster_uuid"], "cluster scope")
    scope_exclusions = {"concrete_indices", "index_metadata", "alias_definitions"}
    _require(_without(old.scope, scope_exclusions) == _without(target.scope, scope_exclusions), "user/endpoint scope")
    alias = target.scope["alias"]
    expected_definition = old.scope.get("alias_definitions", {source_index: {}})[source_index]
    expected_alias = {target_index: {"aliases": {alias: expected_definition}}}
    _require(evidence["alias_before"] == evidence["alias_after"] == expected_alias, "alias drift")
    _require(target.scope.get("alias_definitions") == {target_index: expected_definition}, "alias snapshot")
    _require(old.scope["index_metadata"] == evidence["source_index_metadata"], "source UUID drift")
    for key in ("source_inventory", "target_inventory"):
        _require(evidence[key] == journal["inventory"], key + " differs from journal")
    _require(evidence["parents_inventory"] == journal["parents"], "Parent differs from journal")
    _require(_ordered(old.parents, "id") == _ordered(target.parents, "id"), "Parent content")
    _require(old.status_counts == target.status_counts, "publication status counts")
    _require(_ordered(evidence["source_children"], "child_id") == _ordered(old.children, "child_id"), "old Child projection")
    normalized_children = [{**row, "index_generation": journal["source"]} for row in target.children]
    _require(_ordered(old.children, "child_id") == _ordered(normalized_children, "child_id"), "Child content/locator")
    plans = {p["id"]: p for p in journal["versions"]}
    old_docs, new_docs = [{d["document_version_id"]: d for d in rows} for rows in (old.documents, target.documents)]
    _require(len(plans) == len(journal["versions"]) and set(plans) == set(old_docs) == set(new_docs), "document/version membership")
    _require(set(evidence["originals"]) == set(evidence["asts"]) == set(evidence["manifests"]) == set(plans), "missing artifact proof")
    for version, before in old_docs.items():
        after, plan = new_docs[version], plans[version]
        _require(plan["document_id"] == before["document_id"] and plan["user_id"] == old.scope["user_id"], "journal version scope")
        _require(_without(before, VERSION_FIELDS) == _without(after, VERSION_FIELDS), "document content/version: " + version)
        _require({k: before[k] for k in VERSION_FIELDS} == plan["before"] and
                 {k: after[k] for k in VERSION_FIELDS} == plan["after"], "manifest metadata: " + version)
        _require(evidence["originals"][version] == before["content_hash"] and
                 evidence["asts"][version] == before["canonical_ast_hash"], "original/AST: " + version)
        manifests = evidence["manifests"][version]
        _require(manifests["before_sha256"] == before["manifest_hash"] and
                 manifests["after_sha256"] == after["manifest_hash"], "manifest hash: " + version)
        _require(manifests["before"]["index_generation"] == journal["source"] and
                 manifests["after"]["index_generation"] == journal["target"], "manifest generation")
        _require(manifests["after"] == plan["manifest"] and
                 _without(manifests["before"], {"index_generation"}) ==
                 _without(manifests["after"], {"index_generation"}), "manifest content: " + version)
    return {"verified": True, "normalization": NORMALIZATION, "source_snapshot_id": old.snapshot_id,
            "target_snapshot_id": target.snapshot_id, "journal_sha256": evidence["journal_sha256"],
            "allowed_changes": ["index_generation", "concrete_indices/index_uuid", "manifest_path/hash", "manifest.index_generation"],
            "documents": len(old.documents), "parents": len(old.parents), "children": len(old.children)}


def _artifact(artifacts, uri, expected=None):
    ref = artifacts.describe(uri)
    _require(expected is None or ref.sha256 == expected, "artifact hash: " + uri)
    # Preserve all manifest fields; the permissive manifest model must not discard extras.
    return ref.sha256, artifacts.read_json(ref)


async def _capture_once(template, *, engine, es, artifacts, index, generation):
    user, alias = template.scope["user_id"], template.scope["alias"]
    cluster = (await es.info()).body
    alias_before = (await es.indices.get_alias(name=alias)).body
    _require(set(alias_before) == {index}, "target alias not exact physical index")
    metadata = (await es.indices.get_settings(index=index, filter_path="*.settings.index.uuid")).body
    async with engine.connect() as conn:
        counts = (await conn.execute(select(documents.c.user_id, documents.c.status, func.count()).group_by(
            documents.c.user_id, documents.c.status).order_by(documents.c.user_id, documents.c.status))).all()
        doc_rows = [dict(r) for r in (await conn.execute(select(documents).where(
            documents.c.user_id == user, documents.c.status == "active").order_by(documents.c.id))).mappings()]
        ids = [r["active_version_id"] for r in doc_rows]
        version_rows = [dict(r) for r in (await conn.execute(select(document_versions).where(
            document_versions.c.id.in_(ids)).order_by(document_versions.c.id))).mappings()]
        parent_rows = [dict(r) for r in (await conn.execute(select(parent_chunks).where(
            parent_chunks.c.document_version_id.in_(ids)).order_by(parent_chunks.c.id))).mappings()]
    old_docs = {d["document_version_id"]: d for d in template.documents}
    _require(set(ids) == set(old_docs) and len(ids) == len(old_docs), "document/version membership changed")
    versions = {v["id"]: v for v in version_rows}
    _require(set(versions) == set(ids), "missing SQL versions")
    children = await scan_children(es, index)
    inventory, hashes = child_inventory(children, generation)
    originals, asts, manifests, projected_docs = {}, {}, {}, []
    for doc in doc_rows:
        version = versions[doc["active_version_id"]]
        key, previous = version["id"], old_docs[version["id"]]
        _require(version["status"] == "active" and version["index_generation"] == generation
                 and doc["deletion_status"] not in {"pending", "fenced"}
                 and version["document_id"] == doc["id"] == previous["document_id"], "SQL publication/scope")
        path = Path(previous["source_path"])
        originals[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(originals[key] == doc["content_hash"], "original hash: " + key)
        asts[key], _ = _artifact(artifacts, version["canonical_ast_path"], version["canonical_ast_hash"])
        digest, manifest = _artifact(artifacts, version["manifest_path"], version["manifest_hash"])
        _require(manifest["canonical_ast_sha256"] == asts[key] and manifest["child_count"] == version["child_count"]
                 and manifest["parent_count"] == version["parent_count"] and manifest["index_generation"] == generation,
                 "manifest/SQL consistency: " + key)
        manifests[key] = {"after": manifest, "after_sha256": digest}
        projected_docs.append({"document_id": doc["id"], "filename": doc["filename"], "content_hash": doc["content_hash"],
            "document_version_id": key, **{k: version[k] for k in ("index_generation", "parent_count", "child_count",
                "canonical_ast_path", "canonical_ast_hash", "parser_version", "pipeline_version", "embedding_version",
                "manifest_path", "manifest_hash")}, "source_path": str(path.resolve()),
            "es_child_count": inventory["versions"].get(key, 0),
            "searchable": inventory["versions"].get(key, 0) == version["child_count"] > 0})
    _require(inventory["versions"] == {v["id"]: v["child_count"] for v in version_rows}, "SQL/ES version inventory")
    _require(inventory["identities"] == {v["id"]: [user, v["document_id"]] for v in version_rows}, "SQL/ES scope")
    parents = parent_inventory(parent_rows)
    _require(parents["counts"] == {v["id"]: v["parent_count"] for v in version_rows}, "SQL Parent counts")
    _require(all(p["user_id"] == user and p["document_id"] == versions[p["document_version_id"]]["document_id"]
                 for p in parent_rows), "SQL Parent scope")
    active_parents = [p for p in parent_rows if p["status"] == "active"]
    by_parent = {p["id"]: p for p in active_parents}
    for child in children:
        parent = by_parent.get(child["parent_id"])
        _require(parent is not None and parent["document_version_id"] == child["document_version_id"], "Child Parent version")
    fields = set(template.children[0]) - {"child_id"}
    _require(all(set(c) - {"child_id"} == fields for c in template.children), "inconsistent snapshot Child schema")
    projected_children = [{"child_id": c["_id"], **{k: c[k] for k in fields}} for c in children]
    alias_after = (await es.indices.get_alias(name=alias)).body
    _require(alias_before == alias_after, "alias drift during capture")
    scope = {**template.scope, "concrete_indices": [index], "cluster_uuid": cluster["cluster_uuid"], "index_metadata": metadata,
             "alias_definitions": {index: alias_before[index]["aliases"][alias]}}
    value = {"schema_version": 2, "captured_at": datetime.now(timezone.utc).isoformat(), "scope": scope,
             "status_counts": [{"user_id": u, "status": s, "count": n} for u, s, n in counts],
             "documents": _ordered(projected_docs, "document_id"), "parents": active_parents,
             "children": _ordered(projected_children, "child_id")}
    snapshot = CorpusSnapshot.model_validate({**value, "snapshot_id": canonical_hash(value)})
    proof = {"protocol": PROTOCOL, "target_inventory": inventory, "child_hashes": hashes, "parents_inventory": parents,
             "originals": originals, "asts": asts, "manifests": manifests, "alias_before": alias_before,
             "alias_after": alias_after, "effective_sql_sha256": canonical_hash([doc_rows, version_rows, parent_rows]),
             "snapshot_content_sha256": canonical_hash(_without(value, {"captured_at"}))}
    proof["content_fingerprint"] = canonical_hash(proof)
    proof["observed_at"] = value["captured_at"]
    return snapshot, proof


async def capture_corpus_evidence(template: CorpusSnapshot, *, engine, es, artifacts, index: str,
                                  generation: str) -> tuple[CorpusSnapshot, dict[str, Any]]:
    args = dict(engine=engine, es=es, artifacts=artifacts, index=index, generation=generation)
    snapshot, before = await _capture_once(template, **args)
    _, after = await _capture_once(template, **args)
    assert_corpus_unchanged(before, after)
    return snapshot, before


async def capture_migration_evidence(old: CorpusSnapshot, journal: dict[str, Any], *, engine, es,
                                     artifacts) -> tuple[CorpusSnapshot, dict[str, Any]]:
    _require(journal["phase"] == "complete", "incomplete journal")
    source_index, target_index = ["agenticrag-children-" + journal[k] for k in ("source", "target")]
    target, proof = await capture_corpus_evidence(old, engine=engine, es=es, artifacts=artifacts,
                                                 index=target_index, generation=journal["target"])
    source = await scan_children(es, source_index)
    inventory, hashes = child_inventory(source, journal["source"])
    inventory_after, _ = child_inventory(await scan_children(es, source_index), journal["source"])
    _require(inventory == inventory_after, "source drift during capture")
    fields = set(old.children[0]) - {"child_id"}
    proof.update(journal_sha256=canonical_hash(journal), source_inventory=inventory, source_child_hashes=hashes,
        source_children=[{"child_id": c["_id"], **{k: c[k] for k in fields}} for c in source],
        source_index_metadata=(await es.indices.get_settings(index=source_index, filter_path="*.settings.index.uuid")).body)
    for plan in journal["versions"]:
        digest, value = _artifact(artifacts, plan["before"]["manifest_path"], plan["before"]["manifest_hash"])
        proof["manifests"][plan["id"]].update(before=value, before_sha256=digest)
    verify_migration_equivalence(old, target, journal, proof)
    return target, proof
