"""Read-only full-corpus drift detection, including irrelevant distractors."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import httpx
from sqlalchemy import select

from agentic_rag.persistence.repositories import documents, document_versions, parent_chunks
from evals.gold_v2_models import CorpusSnapshot, canonical_hash


def experiment_fingerprint(gold_hash, corpus_id, config_id, judge_id, case_ids, **options):
    return canonical_hash({"gold": gold_hash, "corpus": corpus_id, "runtime": config_id,
                           "judge": judge_id, "case_ids": case_ids, **options})


def verify_alias_definition(snapshot, aliases):
    alias = snapshot.scope["alias"]
    actual = {index: row["aliases"][alias] for index, row in aliases.items()}
    # Existing frozen v2 snapshots used a plain, unrouted alias. Never assume
    # changed filters/routing are harmless merely because its index UUID stayed.
    expected = snapshot.scope.get("alias_definitions", {index: {} for index in snapshot.scope["concrete_indices"]})
    if actual != expected:
        raise ValueError("corpus drift: alias definition/filter/routing")
    return actual


def compare_live_snapshot(snapshot: CorpusSnapshot, live: dict, *, mode="strict"):
    if mode not in {"strict", "robustness"}:
        raise ValueError("invalid snapshot mode")
    differences = []
    for key, id_key in (("documents", "document_version_id"), ("parents", "id"), ("children", "child_id")):
        frozen = sorted(getattr(snapshot, key), key=lambda r: r[id_key])
        actual = sorted(live[key], key=lambda r: r[id_key])
        if canonical_hash(frozen) != canonical_hash(actual):
            differences.append(key)
    if snapshot.scope != live["scope"]:
        differences.append("scope/alias/index_uuid")
    if differences and mode == "strict":
        raise ValueError("corpus drift: " + ", ".join(differences))
    return {"status": "drift" if differences else "matched", "differences": differences,
            "missing_child_ids": sorted({c["child_id"] for c in snapshot.children} - {c["child_id"] for c in live["children"]}),
            "live_fingerprint": canonical_hash({k: live[k] for k in ("scope", "documents", "parents", "children")})}


def verify_originals(snapshot, cases):
    """Validate original hashes and original anchors, never retrieved text."""
    from scripts.build_formal_eval_gold import read_original
    from evals.formal_gold import check_anchor
    originals = {}
    for document in snapshot.documents:
        text, structure = read_original(document)
        originals[document["document_version_id"]] = (text, structure)
    for case in cases:
        for group in case.required_evidence_groups_all_of:
            for anchor in group.source_anchors:
                check_anchor(originals[group.document_version_id][0], anchor)
        if case.derivation and case.derivation.operation == "2026_limit-2025_limit":
            for year, value in case.derivation.inputs.items():
                pattern = r"每人每晚\s*" + re.escape(format(value, "g")) + r"\s*元"
                versions = {s.document_version_id for s in case.source_documents if year in s.filename}
                if not any(group.document_version_id in versions and any(re.search(pattern, a) for a in group.source_anchors)
                           for group in case.required_evidence_groups_all_of):
                    raise ValueError("gold yearly calculation input lacks original year/unit source binding")
        if case.derivation and case.derivation.sheet:
            data = originals[case.source_documents[0].document_version_id][1]["sheets"][case.derivation.sheet]
            # Validate numeric inputs against original cells, in addition to
            # the independent formula check performed by the strict loader.
            rows = [list(row) for row in data[5:9]]
            expected = case.derivation.inputs
            if case.derivation.operation == "B6+C6-D6":
                # Frozen row also carries E6 (cached formula output), while
                # source_cells identifies A6:D6 as the independent inputs.
                actual = rows[0][:len(expected)]
            else:
                actual = [r[:5] for r in rows]
            if actual != expected:
                raise ValueError("gold calculation input differs from original workbook")
    return {"original_documents_verified": len(originals), "original_cases_verified": len(cases)}


async def capture_live_snapshot(snapshot: CorpusSnapshot, engine, artifact_root: Path):
    user = snapshot.scope["user_id"]
    async with engine.connect() as conn:
        doc_rows = (await conn.execute(select(
            documents.c.id.label("document_id"), documents.c.filename, documents.c.content_hash,
            documents.c.active_version_id.label("document_version_id"),
            *(document_versions.c[name] for name in ("index_generation", "parent_count", "child_count", "canonical_ast_path",
                "canonical_ast_hash", "parser_version", "pipeline_version", "embedding_version", "manifest_path", "manifest_hash")),
        ).join(document_versions, documents.c.active_version_id == document_versions.c.id).where(
            documents.c.user_id == user, documents.c.status == "active", document_versions.c.status == "active"
        ))).mappings().all()
        parents = (await conn.execute(select(parent_chunks).where(parent_chunks.c.user_id == user,
                                                                 parent_chunks.c.status == "active"))).mappings().all()
    async with httpx.AsyncClient(base_url=snapshot.scope["es_url"], trust_env=False, timeout=60) as es:
        async def get(path):
            response = await es.get(path)
            response.raise_for_status()
            return response.json()
        cluster = await get("/")
        aliases = await get("/_alias/" + snapshot.scope["alias"])
        alias_definitions = verify_alias_definition(snapshot, aliases)
        indices = sorted(aliases)
        metadata = await get("/" + ",".join(indices) + "/_settings?filter_path=*.settings.index.uuid")
        body = {"size": 1000, "sort": ["_doc"], "_source": [k for k in snapshot.children[0] if k != "child_id"],
                "query": {"bool": {"filter": [{"term": {"user_id": user}}, {"term": {"is_active": True}}]}}}
        scroll_id, children = None, []
        try:
            response = await es.post("/" + snapshot.scope["alias"] + "/_search?scroll=2m", json=body)
            response.raise_for_status()
            result = response.json()
            while True:
                scroll_id = result.get("_scroll_id", scroll_id)
                hits = result["hits"]["hits"]
                if not hits:
                    break
                children.extend({"child_id": h["_id"], **h["_source"]} for h in hits)
                response = await es.post("/_search/scroll", json={"scroll_id": scroll_id, "scroll": "2m"})
                response.raise_for_status()
                result = response.json()
        finally:
            if scroll_id:
                response = await es.request("DELETE", "/_search/scroll", json={"scroll_id": [scroll_id]})
                response.raise_for_status()
        if aliases != await get("/_alias/" + snapshot.scope["alias"]):
            raise ValueError("corpus drift: alias changed during capture")
    rows = []
    counts = {}
    for child in children:
        counts[child["document_version_id"]] = counts.get(child["document_version_id"], 0) + 1
    for row in doc_rows:
        doc = dict(row)
        sources = list((artifact_root / "documents" / user / doc["document_id"] / doc["document_version_id"] / "source").glob("*"))
        if len(sources) != 1 or hashlib.sha256(sources[0].read_bytes()).hexdigest() != doc["content_hash"]:
            raise ValueError("corpus drift: original source hash")
        doc["source_path"] = str(sources[0].resolve())
        doc["es_child_count"] = counts.get(doc["document_version_id"], 0)
        doc["searchable"] = doc["es_child_count"] == doc["child_count"] and doc["child_count"] > 0
        rows.append(doc)
    scope = {**snapshot.scope, "cluster_uuid": cluster["cluster_uuid"], "concrete_indices": indices, "index_metadata": metadata}
    return {"scope": scope, "alias_definitions": alias_definitions,
            "documents": rows, "parents": [dict(p) for p in parents], "children": children}
