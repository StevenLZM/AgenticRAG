"""Upload a frozen synthetic batch through the ordinary API; resume by job ID."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import httpx

from evals.bulk_corpus import ROOT, freeze_manifest, upload_order
from evals.report import atomic_write_text
from evals.upload import upload_document


async def upload_batch(client, root, documents, *, concurrency=2):
    for start in range(0, len(documents), concurrency):
        batch = documents[start:start + concurrency]
        results = await asyncio.gather(*[
            upload_document(client, root / "documents" / doc["filename"], doc["sha256"],
                            root / "uploads" / (doc["document_key"] + ".json"),
                            timeout_seconds=900, poll_interval=2)
            for doc in batch
        ])
        statuses = Counter(json.loads(p.read_text())["status"] for p in (root / "uploads").glob("*.json"))
        print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "processed": start + len(batch),
                          "selected": len(documents), "statuses": statuses}, ensure_ascii=False), flush=True)
        if any(row["status"] != "completed" for row in results):
            raise RuntimeError("upload not completed; inspect ledger, do not duplicate POST")


async def verify(client, root, docs, settings, runtime):
    """Read actual jobs, scoped SQL versions/parents and active ES child counts."""
    from sqlalchemy import select, func
    from agentic_rag.persistence.mysql import create_mysql_engine
    from agentic_rag.persistence.repositories import documents, document_versions, parent_chunks

    receipts = [json.loads((root / "uploads" / (doc["document_key"] + ".json")).read_text()) for doc in docs]
    for doc, receipt in zip(docs, receipts, strict=True):
        if receipt["sha256"] != doc["sha256"] or receipt["base_url"].rstrip("/") != str(client.base_url).rstrip("/"):
            raise ValueError("receipt binding mismatch")
        response = await client.get("/v1/ingestion-jobs/" + receipt["job_id"])
        response.raise_for_status()
        actual = response.json()
        if actual["status"] != "completed" or any(actual[k] != receipt[k] for k in ("job_id", "document_id", "document_version_id")):
            raise ValueError("job not completed or identity mismatch")
    ids = [r["document_id"] for r in receipts]
    versions = [r["document_version_id"] for r in receipts]
    if len(set(ids)) != len(docs) or len(set(versions)) != len(docs):
        raise ValueError("duplicate document/version identity")
    engine = create_mysql_engine(settings.mysql_dsn)
    try:
        async with engine.connect() as conn:
            rows = (await conn.execute(select(
                documents.c.id, documents.c.filename, documents.c.content_hash, documents.c.active_version_id,
                document_versions.c.parent_count, document_versions.c.child_count,
                document_versions.c.index_generation,
            ).join(document_versions, documents.c.active_version_id == document_versions.c.id).where(
                documents.c.id.in_(ids), documents.c.user_id == settings.default_user_id,
                documents.c.status == "active", document_versions.c.status == "active",
            ))).mappings().all()
            parents = dict((await conn.execute(select(parent_chunks.c.document_version_id, func.count()).where(
                parent_chunks.c.document_version_id.in_(versions), parent_chunks.c.user_id == settings.default_user_id,
                parent_chunks.c.status == "active",
            ).group_by(parent_chunks.c.document_version_id))).all())
    finally:
        await engine.dispose()
    by_id = {row["id"]: row for row in rows}
    for doc, receipt in zip(docs, receipts, strict=True):
        row = by_id.get(receipt["document_id"])
        if (row is None or row["active_version_id"] != receipt["document_version_id"]
                or row["content_hash"] != doc["sha256"] or row["filename"] != doc["filename"]
                or row["index_generation"] != runtime["index_generation"]
                or row["parent_count"] < 1 or row["child_count"] < 1
                or parents.get(row["active_version_id"]) != row["parent_count"]):
            raise ValueError("SQL publication mismatch: " + doc["document_key"])
    query = {"size": 0, "track_total_hits": True, "query": {"bool": {"filter": [
        {"term": {"user_id": settings.default_user_id}}, {"term": {"is_active": True}},
        {"term": {"index_generation": runtime["index_generation"]}},
        {"terms": {"document_version_id": versions}},
    ]}}, "aggs": {"versions": {"terms": {"field": "document_version_id", "size": len(versions)}}}}
    async with httpx.AsyncClient(trust_env=False, timeout=60) as es:
        response = await es.post(settings.elasticsearch_url.rstrip("/") + "/agenticrag-children-active/_search", json=query)
        response.raise_for_status()
        data = response.json()
    counts = {b["key"]: b["doc_count"] for b in data["aggregations"]["versions"]["buckets"]}
    if counts != {row["active_version_id"]: row["child_count"] for row in rows}:
        raise ValueError("ES active child counts differ from published SQL versions")
    report = {"batch_id": root.name, "verified_at": datetime.now(timezone.utc).isoformat(),
              "api_url": str(client.base_url), "index_generation": runtime["index_generation"],
              "document_count": len(rows), "parent_count": sum(parents.values()), "child_count": sum(counts.values()),
              "formats": dict(Counter(d["format"] for d in docs)), "all_selected_verified": True,
              "documents": [{"document_key": d["document_key"], **r} for d, r in zip(docs, receipts, strict=True)]}
    atomic_write_text(root / f"verification-{len(docs)}.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, ensure_ascii=False), flush=True)


async def run(args):
    from agentic_rag.config import Settings
    root = args.root.resolve()
    source = json.loads((root / "source.json").read_text())
    manifest = freeze_manifest(root, source)
    docs = upload_order(manifest["documents"])
    if args.limit:
        docs = docs[:args.limit]
    settings = Settings()
    if args.base_url.rstrip("/") != "http://localhost:8000":
        raise ValueError("this batch is explicitly approved only for http://localhost:8000")
    async with httpx.AsyncClient(base_url=args.base_url, timeout=60, trust_env=False) as client:
        response = await client.get("/v1/runtime/summary")
        response.raise_for_status()
        runtime = response.json()
        if runtime["index_generation"] != settings.index_generation:
            raise ValueError("API and local configuration generation mismatch")
        binding = {"api_url": args.base_url, "index_generation": runtime["index_generation"],
                   "user_id": settings.default_user_id,
                   "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
        binding_path = root / "upload-binding.json"
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise ValueError("upload binding changed")
        if not binding_path.exists():
            atomic_write_text(binding_path, json.dumps(binding, indent=2) + "\n")
        with (root / ".upload.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.stage == "upload":
                await upload_batch(client, root, docs, concurrency=args.concurrency)
            await verify(client, root, docs, settings, runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--stage", choices=("upload", "verify"), default="verify")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 1000:
        parser.error("limit must be between 1 and 1000")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
