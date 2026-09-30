"""Verify a completed frozen-corpus run against live scoped server state.

Local judge rows are trusted experiment records, not cryptographically signed
provider receipts. Their aggregation is checked; no judge calls are repeated.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from agentic_rag.config import Settings
from agentic_rag.persistence.repositories import agent_runs, ingestion_jobs, parent_chunks
from evals.clients import _citation_coverage_from_answer
from evals.collector import RuntimeCollector
from evals.corpus import SOURCE_ROOT, load_source
from evals.gold_mapping import map_source_facts
from evals.models import EvaluationCase
from evals.real_stack import isolated_environment
from evals.report import build_summary
from evals.run import EvalCaseResult, _case_fingerprint, _deterministic_metrics
from evals.verification import verify_projection


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def load_frozen_inputs(root, *, docs_dir=None, gold_path=None):
    """Check pre-query gold and uploaded bytes before any network access."""
    manifest = _load(root / "corpus/manifest.json")
    source = load_source()
    if (manifest["cases"] != source["cases"]
            or manifest["source_sha256"] != _digest((SOURCE_ROOT / "source.json").read_bytes())):
        raise ValueError("frozen source gold changed")
    locks = _load(SOURCE_ROOT / "assets.sha256.json")
    if len(manifest["documents"]) != len(source["documents"]):
        raise ValueError("frozen documents changed")
    for doc, original in zip(manifest["documents"], source["documents"], strict=True):
        if any(doc.get(k) != v for k, v in original.items()):
            raise ValueError("frozen document metadata changed")
        if doc["sha256"] != locks[doc["filename"]] or _digest(
                ((docs_dir or root / "corpus") / doc["filename"]).read_bytes()) != doc["sha256"]:
            raise ValueError("frozen upload bytes changed")
    allocation = _load(root / "stack-allocation.json")
    if _digest((root / "corpus/manifest.json").read_bytes()) != allocation["manifest_sha256"]:
        raise ValueError("frozen manifest binding changed")
    gold = _load(gold_path or root / "gold-mapping.json")
    if (gold["manifest_sha256"] != allocation["manifest_sha256"]
            or gold["allocation_id"] != allocation["allocation_id"]
            or gold["index_generation"] != allocation["index_generation"]):
        raise ValueError("gold mapping allocation mismatch")
    expected = [{**case, "gold_parent_ids": sorted({gold["facts"][fact]["parent_id"]
                 for fact in case["source_fact_ids"]})} for case in manifest["cases"]]
    if gold["cases"] != expected:
        raise ValueError("mapped gold differs from frozen source")
    return allocation, manifest, gold


def check_summary(rows, summary):
    """Recompute aggregates; a positive JSON counter is not an attestation."""
    rebuilt = build_summary(rows, evaluation_mode="api", client_provenance="real_query_api")
    for key, value in rebuilt.items():
        if key == "real_query_count":
            continue
        if summary.get(key) != value:
            raise ValueError(f"report aggregate mismatch: {key}")
    if (summary.get("requested_cases") != len(rows)
            or summary.get("real_query_count") != len(rows)
            or summary.get("query_failure_count") != 0 or summary.get("failures") != []
            or not rows or rebuilt["ragas_status"] != "available"):
        raise ValueError("report is incomplete")


def check_scored_row(case, row, observed):
    if (row.case_fingerprint != _case_fingerprint(case)
            or row.runtime_config_snapshot_id != case.runtime_config_snapshot_id
            or row.evaluation_mode != "api" or row.client_provenance != "real_query_api"):
        raise ValueError("scored case binding mismatch")
    verify_projection(observed, row)
    expected = _deterministic_metrics(case, ranked_parent_ids=observed["ranked_parent_ids"] or [],
                                      response=observed, events=[])
    if expected != row.deterministic_metrics:
        raise ValueError("scored deterministic metrics differ from runtime")
    answer = observed["answer"]
    origin = "terminal_status" if answer.get("status") and not answer.get("segments") else "model"
    if (row.audited != (answer.get("audited") is True)
            or row.citation_coverage != _citation_coverage_from_answer(answer)
            or row.answer_origin != origin
            or row.evidence_parent_ids != tuple(answer.get("evidence_parent_ids", []))):
        raise ValueError("scored answer safety fields differ from runtime")
    names = {"faithfulness", "answer_relevancy", "context_precision"} if case.answerable else {"correct_refusal"}
    if (set(row.ragas_metrics) != names | {"status"} or row.ragas_metrics["status"] != "available"
            or not row.judge_fingerprint or not row.judge_metadata):
        raise ValueError("missing real judge result metadata")


def project_operational(timing, answer):
    # Worker does not currently populate SQL termination_reason; the public
    # answer was already checked against the scoped checkpoint by collector.
    outcome = answer.get("status") or "completed"
    if timing["termination_reason"] not in {None, outcome}:
        raise ValueError("SQL termination differs from verified checkpoint")
    started, finished = timing["started_at"], timing["finished_at"]
    elapsed = (finished - started).total_seconds() if started and finished else None
    if elapsed is not None and elapsed < 0:
        raise ValueError("invalid run timestamps")
    return {"terminal_outcome": outcome, "elapsed_seconds": elapsed,
            "tokens": None, "tokens_status": "unavailable: provider usage not persisted per run"}


async def verify_saved_evaluation(run_dir: Path, *, output_dir=None, docs_dir=None, gold_path=None) -> dict:
    root = Path(run_dir).resolve()
    allocation, manifest, gold = load_frozen_inputs(root, docs_dir=docs_dir, gold_path=gold_path)
    output = output_dir or root / "evaluation"
    summary = _load(output / "summary.json")
    rows = [EvalCaseResult.model_validate_json(line) for line in
            (output / "results.jsonl").read_text().splitlines() if line.strip()]
    ids = [case["case_id"] for case in gold["cases"]]
    if [row.case_id for row in rows] != ids or len(set(ids)) != len(ids):
        raise ValueError("full frozen case set is missing, reordered or duplicated")
    check_summary(rows, summary)
    if (output / "failures.jsonl").read_text().strip():
        raise ValueError("unresolved evaluation failures")
    verification = summary.get("runtime_verification", {})
    receipts = verification.get("receipts", [])
    if (verification.get("method") != "scoped-mysql-checkpoint-v1"
            or [r.get("case_id") for r in receipts] != ids
            or len({r.get("run_id") for r in receipts}) != len(ids)):
        raise ValueError("runtime receipt coverage mismatch")
    settings = Settings()
    env = isolated_environment(settings, allocation, root=Path.cwd())
    engine = create_async_engine(env["AGENTIC_RAG_MYSQL_DSN"])
    collector = RuntimeCollector(engine, Path(env["AGENTIC_RAG_QUERY_CHECKPOINT_PATH"]))
    operational = []
    try:
        # Rebind source facts using only the ingestion version, never retrieved ranking.
        facts = {}
        async with engine.connect() as connection:
            for doc in manifest["documents"]:
                upload = _load(root / "uploads" / (doc["document_key"] + ".json"))
                if (upload["status"] != "completed" or upload["sha256"] != doc["sha256"]
                        or upload["base_url"].rstrip("/") != allocation["api_url"].rstrip("/")):
                    raise ValueError("upload ledger mismatch")
                job = (await connection.execute(select(ingestion_jobs.c.status).where(
                    ingestion_jobs.c.id == upload["job_id"], ingestion_jobs.c.user_id == allocation["user_id"],
                    ingestion_jobs.c.document_id == upload["document_id"],
                    ingestion_jobs.c.document_version_id == upload["document_version_id"],
                ))).scalar_one_or_none()
                if job != "completed":
                    raise ValueError("scoped ingestion job is not completed")
                parents = (await connection.execute(select(parent_chunks).where(
                    parent_chunks.c.user_id == allocation["user_id"],
                    parent_chunks.c.document_id == upload["document_id"],
                    parent_chunks.c.document_version_id == upload["document_version_id"],
                ))).mappings().all()
                facts.update(map_source_facts(doc["facts"], parents, user_id=allocation["user_id"],
                                             document_version_id=upload["document_version_id"]))
            if facts != gold["facts"]:
                raise ValueError("live source mapping differs from frozen gold")
        for spec, row, receipt in zip(gold["cases"], rows, receipts, strict=True):
            case = EvaluationCase(case_id=spec["case_id"], user_id=allocation["user_id"],
                question=spec["question"], answerable=spec["answerable"],
                reference_answer=spec["reference_answer"], reference_parent_ids=spec["gold_parent_ids"],
                expected_route=spec["expected_route"], tags=spec["tags"],
                runtime_config_snapshot_id=summary["runtime_config_snapshot_id"])
            ledger = _load(output / "queries" / (case.case_id + ".json"))
            if (ledger["case_sha256"] != _digest(case.model_dump_json().encode())
                    or ledger["base_url"] != allocation["api_url"].rstrip("/")
                    or ledger["run_id"] != receipt["run_id"] or ledger["status"] != "collected"
                    or receipt["snapshot_id"] != case.runtime_config_snapshot_id
                    or receipt["artifact"] != f"runtime-evidence/{case.case_id}.json"):
                raise ValueError("case runtime ledger binding mismatch")
            observed = await collector.collect(run_id=ledger["run_id"], user_id=case.user_id,
                snapshot_id=case.runtime_config_snapshot_id, question=case.question)
            check_scored_row(case, row, observed)
            data = (output / receipt["artifact"]).read_bytes()
            if _digest(data) != receipt["sha256"] or json.loads(data) != observed:
                raise ValueError("runtime artifact differs from live evidence")
            async with engine.connect() as connection:
                timing = (await connection.execute(select(agent_runs.c.started_at, agent_runs.c.finished_at,
                    agent_runs.c.termination_reason).where(agent_runs.c.id == ledger["run_id"],
                    agent_runs.c.user_id == case.user_id,
                    agent_runs.c.runtime_config_snapshot_id == case.runtime_config_snapshot_id,
                ))).mappings().one()
            operational.append({"case_id": case.case_id, "run_id": ledger["run_id"],
                                **project_operational(timing, observed["answer"])})
    finally:
        await engine.dispose()
    return {**summary, "completion_verified": True, "operational_cases": operational,
            "outcome_counts": dict(Counter(item["terminal_outcome"] for item in operational))}
