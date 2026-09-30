"""Read-only SQL/ES corpus inventory; never upload, delete or reindex documents."""
import asyncio
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select, func

from agentic_rag.config import Settings
from agentic_rag.persistence.mysql import create_mysql_engine
from agentic_rag.persistence.repositories import documents, document_versions, parent_chunks
from evals.report import atomic_write_text

ROOT = Path("var/artifacts/evals/formal-corpus-gold-v2-20260930")


async def main(root=ROOT):
    settings = Settings()
    engine = create_mysql_engine(settings.mysql_dsn)
    try:
        async with engine.connect() as conn:
            users = (await conn.execute(select(documents.c.user_id, documents.c.status, func.count()).group_by(
                documents.c.user_id, documents.c.status))).all()
            rows = (await conn.execute(select(
                documents.c.id.label("document_id"), documents.c.filename, documents.c.content_hash,
                documents.c.active_version_id.label("document_version_id"),
                document_versions.c.index_generation, document_versions.c.parent_count,
                document_versions.c.child_count, document_versions.c.canonical_ast_path,
                document_versions.c.canonical_ast_hash, document_versions.c.parser_version,
                document_versions.c.pipeline_version, document_versions.c.embedding_version,
                document_versions.c.manifest_path, document_versions.c.manifest_hash,
            ).join(document_versions, documents.c.active_version_id == document_versions.c.id).where(
                documents.c.user_id == settings.default_user_id, documents.c.status == "active",
                document_versions.c.status == "active").order_by(documents.c.id))).mappings().all()
            versions = [r["document_version_id"] for r in rows]
            parents = (await conn.execute(select(parent_chunks).where(
                parent_chunks.c.user_id == settings.default_user_id,
                parent_chunks.c.document_version_id.in_(versions), parent_chunks.c.status == "active",
            ).order_by(parent_chunks.c.id))).mappings().all()
    finally:
        await engine.dispose()
    async with httpx.AsyncClient(base_url=settings.elasticsearch_url, trust_env=False, timeout=60) as es:
        async def get(path):
            response = await es.get(path)
            response.raise_for_status()
            return response.json()
        cluster = await get("/")
        alias = await get("/_alias/agenticrag-children-active")
        indices = sorted(alias)
        metadata = await get("/" + ",".join(indices) + "/_settings?filter_path=*.settings.index.uuid")
        # Scroll only the fixed concrete indices, not a mutable alias. Cleanup is mandatory.
        body = {"size": 1000, "sort": ["_doc"], "_source": [
            "document_id", "document_version_id", "parent_id", "index_generation", "content",
            "contextualized_content", "heading_path", "heading_ast_locators", "ast_locator",
            "content_type", "token_count", "pipeline_version", "embedding_version", "page_from", "page_to",
            "user_id", "parent_ordinal"],
            "query": {"bool": {"filter": [{"term": {"user_id": settings.default_user_id}},
                {"term": {"is_active": True}}]}}}
        response = await es.post("/" + ",".join(indices) + "/_search?scroll=2m", json=body)
        response.raise_for_status()
        result = response.json()
        children, scroll_id = [], result.get("_scroll_id")
        try:
            while result["hits"]["hits"]:
                children.extend({"child_id": h["_id"], **h["_source"]} for h in result["hits"]["hits"])
                response = await es.post("/_search/scroll", json={"scroll": "2m", "scroll_id": scroll_id})
                response.raise_for_status()
                result = response.json()
                scroll_id = result.get("_scroll_id", scroll_id)
        finally:
            if scroll_id:
                await es.request("DELETE", "/_search/scroll", json={"scroll_id": [scroll_id]})
        alias_after = await get("/_alias/agenticrag-children-active")
        if alias != alias_after:
            raise RuntimeError("alias changed during snapshot")
    docs = [dict(r) for r in rows]
    for doc in docs:
        paths = list((settings.artifact_root / "documents" / settings.default_user_id /
                      doc["document_id"] / doc["document_version_id"] / "source").glob("*"))
        if len(paths) != 1:
            raise ValueError("source missing or ambiguous: " + doc["document_id"])
        path = paths[0]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != doc["content_hash"]:
            raise ValueError("original hash mismatch: " + doc["document_id"])
        doc["source_path"] = str(path.resolve())
        doc["es_child_count"] = sum(c["document_version_id"] == doc["document_version_id"] for c in children)
        doc["searchable"] = doc["es_child_count"] == doc["child_count"] and doc["child_count"] > 0
    payload = {"schema_version": 2, "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"user_id": settings.default_user_id, "es_url": settings.elasticsearch_url,
                  "alias": "agenticrag-children-active", "concrete_indices": indices,
                  "cluster_uuid": cluster["cluster_uuid"], "index_metadata": metadata},
        "status_counts": [{"user_id": u, "status": s, "count": n} for u, s, n in users],
        "documents": docs, "parents": [dict(p) for p in parents], "children": children}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    payload["snapshot_id"] = hashlib.sha256(encoded.encode()).hexdigest()
    target = root / "snapshot.json"
    if target.exists():
        raise ValueError("snapshot exists; use a new directory, never overwrite frozen corpus")
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"path": str(target), "documents": len(docs), "searchable": sum(d["searchable"] for d in docs),
                      "parents": len(parents), "children": len(children), "status_counts": payload["status_counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    asyncio.run(main(parser.parse_args().output_dir))
