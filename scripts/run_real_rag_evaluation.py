"""Resume isolated real ingestion and source-gold mapping (query stages follow)."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from agentic_rag.config import Settings
from agentic_rag.persistence.repositories import parent_chunks
from evals.gold_mapping import map_source_facts
from evals.real_stack import isolated_environment
from evals.report import atomic_write_text
from evals.upload import upload_document


async def main(args):
    root = args.run_dir.resolve()
    allocation = json.loads((root / "stack-allocation.json").read_text())
    environment = isolated_environment(Settings(), allocation, root=Path.cwd())
    manifest_bytes = (root / "corpus/manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != allocation["manifest_sha256"]:
        raise ValueError("frozen manifest changed")
    manifest = json.loads(manifest_bytes)
    async with httpx.AsyncClient(timeout=60) as probe:
        response = await probe.get(allocation["elasticsearch_url"])
        response.raise_for_status()
        if response.json()["cluster_uuid"] != allocation["elasticsearch_cluster_uuid"]:
            raise ValueError("isolated ES identity changed")
    if args.stage == "evaluate":
        return await evaluate_cases(root, allocation, environment, args.case_id)
    if args.stage == "upload":
        async with httpx.AsyncClient(base_url=allocation["api_url"], timeout=60) as client:
            for doc in manifest["documents"]:
                result = await upload_document(client, root / "corpus" / doc["filename"], doc["sha256"],
                                               root / "uploads" / (doc["document_key"] + ".json"))
                print(json.dumps({"document_key": doc["document_key"], "status": result["status"]}), flush=True)
                if result["status"] != "completed":
                    return 1
        return 0
    engine = create_async_engine(environment["AGENTIC_RAG_MYSQL_DSN"])
    facts = {}
    try:
        async with engine.connect() as connection:
            for doc in manifest["documents"]:
                upload = json.loads((root / "uploads" / (doc["document_key"] + ".json")).read_text())
                if (upload["status"] != "completed" or upload["sha256"] != doc["sha256"]
                        or upload["base_url"].rstrip("/") != allocation["api_url"].rstrip("/")):
                    raise ValueError("upload is not complete or belongs to another source/endpoint")
                rows = (await connection.execute(select(parent_chunks).where(
                    parent_chunks.c.user_id == allocation["user_id"],
                    parent_chunks.c.document_id == upload["document_id"],
                    parent_chunks.c.document_version_id == upload["document_version_id"],
                ))).mappings().all()
                result = map_source_facts(doc["facts"], rows, user_id=allocation["user_id"],
                                          document_version_id=upload["document_version_id"])
                if set(result) & set(facts):
                    raise ValueError("source fact ids must be globally unique")
                facts.update(result)
    finally:
        await engine.dispose()
    mapping = {"manifest_sha256": allocation["manifest_sha256"], "allocation_id": allocation["allocation_id"],
               "index_generation": allocation["index_generation"], "facts": facts,
               "cases": [{**case, "gold_parent_ids": sorted({facts[f]["parent_id"] for f in case["source_fact_ids"]})}
                         for case in manifest["cases"]]}
    output = root / "gold-mapping.json"
    encoded = json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError("existing gold mapping differs; do not overwrite silently")
    if not output.exists():
        atomic_write_text(output, encoded)
    print(json.dumps({"output": str(output), "facts": len(facts), "cases": len(mapping["cases"])}))
    return 0


async def evaluate_cases(root, allocation, environment, case_ids, *, output_dir=None, gold_path=None,
                         expected_snapshot=None):
    from evals.clients import HttpQueryClient
    from evals.collector import RuntimeCollector
    from evals.judge import JudgeConfig, RagasJudge
    from evals.models import EvaluationCase
    from evals.ragas_adapter import RagasAdapter
    from evals.run import EvalRunner

    output_dir = output_dir or root / "evaluation"
    gold = json.loads((gold_path or root / "gold-mapping.json").read_text())
    if gold["manifest_sha256"] != allocation["manifest_sha256"] or gold["allocation_id"] != allocation["allocation_id"]:
        raise ValueError("gold mapping belongs to another allocation")
    async with httpx.AsyncClient(timeout=30) as probe:
        response = await probe.get(allocation["api_url"] + "/v1/runtime/summary")
        response.raise_for_status()
        runtime = response.json()
    if runtime["index_generation"] != allocation["index_generation"]:
        raise ValueError("API index generation mismatch")
    if expected_snapshot and runtime["runtime_config_snapshot_id"] != expected_snapshot:
        raise ValueError("API snapshot changed after preflight")
    if case_ids and not set(case_ids).issubset({c["case_id"] for c in gold["cases"]}):
        raise ValueError("unknown case id")
    cases = [EvaluationCase(
        case_id=c["case_id"], user_id=allocation["user_id"], question=c["question"],
        reference_answer=c["reference_answer"], reference_parent_ids=c["gold_parent_ids"],
        answerable=c["answerable"], expected_route=c["expected_route"], tags=c["tags"],
        runtime_config_snapshot_id=runtime["runtime_config_snapshot_id"],
    ) for c in gold["cases"] if not case_ids or c["case_id"] in case_ids]
    judge = RagasJudge(JudgeConfig.from_settings(Settings()))
    engine = create_async_engine(environment["AGENTIC_RAG_MYSQL_DSN"])
    collector = RuntimeCollector(engine, Path(environment["AGENTIC_RAG_QUERY_CHECKPOINT_PATH"]))
    client = HttpQueryClient(allocation["api_url"], collector=collector,
                             ledger_dir=output_dir / "queries", timeout_seconds=360)
    try:
        summary = await EvalRunner(client, output_dir=output_dir, ragas_adapter=RagasAdapter(judge),
                                   evaluation_mode="api", client_provenance=client.provenance).run(cases)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if summary["ragas_status"] == "available" else 1
    finally:
        await client.aclose()
        await judge.aclose()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("upload", "map", "evaluate"), required=True)
    parser.add_argument("--case-id", action="append", help="select frozen case(s); omit for all 24")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
