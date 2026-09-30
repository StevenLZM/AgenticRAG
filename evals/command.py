"""Evaluate an already ingested frozen corpus; never upload or build an index."""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from agentic_rag.config import Settings
from agentic_rag.persistence.repositories import parent_chunks
from evals.gold_mapping import map_source_facts
from evals.real_stack import isolated_environment
from evals.report import atomic_write_text, build_summary
from evals.run import EvalCaseResult
from evals.saved_report import load_frozen_inputs, verify_saved_evaluation
from scripts.run_real_rag_evaluation import evaluate_cases

DEFAULT_SOURCE = Path("var/artifacts/evals/real-corpus-v1-20260916")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_inventory(docs_dir, manifest):
    """All document files must be covered; auxiliary manifest/QA assets aren't inputs."""
    expected = {doc["filename"] for doc in manifest["documents"]}
    actual = {p.relative_to(docs_dir).as_posix() for p in docs_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".xlsx", ".docx", ".md", ".csv", ".pptx", ".html"}
              and "qa" not in p.relative_to(docs_dir).parts}
    if actual != expected:
        raise ValueError(f"document inventory mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


@contextmanager
def experiment(output, binding, *, resume):
    if resume:
        if not output.is_dir():
            raise ValueError("resume output does not exist")
    else:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("output exists; use --resume explicitly") from exc
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("experiment is already running") from exc
        try:
            path = output / "experiment.json"
            if resume:
                if not path.is_file() or read_json(path) != binding:
                    raise ValueError("experiment binding changed; start a new output directory")
            else:
                atomic_write_text(path, json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


async def preflight(root, docs_dir, gold_path):
    allocation, manifest, gold = load_frozen_inputs(root, docs_dir=docs_dir, gold_path=gold_path)
    check_inventory(docs_dir, manifest)
    environment = isolated_environment(Settings(), allocation, root=Path.cwd())
    uploads = [read_json(root / "uploads" / (d["document_key"] + ".json")) for d in manifest["documents"]]
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(allocation["api_url"] + "/v1/runtime/summary")
        response.raise_for_status()
        runtime = response.json()
        if runtime["index_generation"] != allocation["index_generation"]:
            raise ValueError("API index generation mismatch")
        response = await client.get(allocation["elasticsearch_url"])
        response.raise_for_status()
        if response.json()["cluster_uuid"] != allocation["elasticsearch_cluster_uuid"]:
            raise ValueError("ES cluster identity changed")
        query = {"size": 10000, "track_total_hits": True, "query": {"bool": {"filter": [
            {"term": {"user_id": allocation["user_id"]}}, {"term": {"is_active": True}}]}},
            "_source": ["document_id", "document_version_id", "parent_id", "index_generation"]}
        response = await client.get(allocation["elasticsearch_url"] + "/agenticrag-children-active/_search",
                                    params={"source": json.dumps(query), "source_content_type": "application/json"})
        response.raise_for_status()
        hits = response.json()["hits"]
        children = [hit["_source"] for hit in hits["hits"]]
        versions = {(u["document_id"], u["document_version_id"]) for u in uploads}
        if (not children or hits["total"]["value"] != len(children)
                or {(c["document_id"], c["document_version_id"]) for c in children} != versions
                or any(c["index_generation"] != allocation["index_generation"] for c in children)
                or not {f["parent_id"] for f in gold["facts"].values()} <= {c["parent_id"] for c in children}):
            raise ValueError("existing ES index does not cover exactly the frozen document versions/gold")
    engine = create_async_engine(environment["AGENTIC_RAG_MYSQL_DSN"])
    try:
        facts = {}
        async with engine.connect() as connection:
            for doc, upload in zip(manifest["documents"], uploads, strict=True):
                if (upload["status"] != "completed" or upload["sha256"] != doc["sha256"]
                        or upload["base_url"].rstrip("/") != allocation["api_url"].rstrip("/")):
                    raise ValueError("upload ledger mismatch")
                parents = (await connection.execute(select(parent_chunks).where(
                    parent_chunks.c.user_id == allocation["user_id"],
                    parent_chunks.c.document_id == upload["document_id"],
                    parent_chunks.c.document_version_id == upload["document_version_id"],
                ))).mappings().all()
                facts.update(map_source_facts(doc["facts"], parents, user_id=allocation["user_id"],
                                             document_version_id=upload["document_version_id"]))
        if facts != gold["facts"]:
            raise ValueError("live source mapping differs from frozen gold")
    finally:
        await engine.dispose()
    return allocation, environment, runtime, len(manifest["documents"]), len(gold["cases"])


def partial_summary(output, requested):
    """Diagnostic aggregation only: saved rows do not attest to live evidence."""
    rows, failures, errors = [], [], []
    for name in ("results.jsonl", "failures.jsonl"):
        path = output / name
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeError):
            errors.append(f"{name}: unreadable")
            continue
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                if name == "results.jsonl":
                    row = EvalCaseResult.model_validate_json(line)
                    if any(previous.case_id == row.case_id for previous in rows):
                        raise ValueError("duplicate case")
                    rows.append(row)
                else:
                    failure = json.loads(line)
                    if not isinstance(failure, dict) or not isinstance(failure.get("case_id"), str):
                        raise ValueError("invalid failure")
                    failures.append(failure)
            except (ValueError, TypeError):
                errors.append(f"{name}:{number}: invalid record")
    try:
        summary = build_summary(rows, evaluation_mode="api", client_provenance="real_query_api")
    except ValueError:
        errors.append("saved rows cannot be aggregated (e.g. mixed snapshots)")
        summary = {"completed_cases": len(rows)}
    return {**summary, "requested_cases": requested, "real_query_count": 0,
            "query_failure_count": len(failures), "failures": failures, "artifact_errors": errors,
            "runtime_verification_status": "unverified", "ragas_status": "unverified"}


def render_report(summary, *, verified, error=None):
    lines = ["# RAG 真实测评报告", "", "状态：" + ("全量完成并验证" if verified else "未完成 / 未通过完整验证"), "",
             f"完成用例：{summary.get('completed_cases', 0)} / {summary.get('requested_cases', '?')}",
             f"已验证真实查询：{summary.get('real_query_count', 0)}", "",
             "复用已有文档和索引；真实 Query API 与真实 Ragas。此报告不代表生产发布验收。", "",
             "| 指标 | 分数 | 样本数 |", "| --- | ---: | ---: |"]
    for name, value in summary.get("metrics", {}).items():
        lines.append(f"| {name} | {value} | {summary.get('metric_sample_counts', {}).get(name, 0)} |")
    ragas = summary.get("ragas", {})
    for name, value in ragas.get("metrics", {}).items():
        lines.append(f"| {name} | {value} | {ragas.get('metric_sample_counts', {}).get(name, 0)} |")
    lines.extend(["", "## Ragas 与失败明细", "", "```json",
                  json.dumps({"ragas": summary.get("ragas", {}), "failures": summary.get("failures", []),
                              "outcomes": summary.get("outcome_counts", {}), "error": error,
                              "artifact_errors": summary.get("artifact_errors", [])}, ensure_ascii=False, indent=2), "```", "",
                  "未通过核验时，完成数与分数仅为本地已保存记录，不代表可信成绩；真实查询计数为 0。",
                  "单次检索的排名指标与多轮最终证据覆盖率分别统计；缺失评分不当作零分或成功。",
                  "逐题答案、路由、检索轮次与评分见 results.jsonl；运行证据见 runtime-evidence/。", ""])
    return "\n".join(lines)


async def run(args):
    # Resume resolves inputs from its immutable manifest; caller overrides still must match.
    saved = read_json(args.resume.resolve() / "experiment.json") if args.resume else {}
    root = (args.source_run or Path(saved.get("source_run", DEFAULT_SOURCE))).resolve()
    docs = (args.docs_dir or Path(saved.get("docs_dir", root / "corpus"))).resolve()
    gold = (args.gold or Path(saved.get("gold_path", root / "gold-mapping.json"))).resolve()
    print("[1/3] 只读检查文档、金标、API、ES 和 MySQL…", flush=True)
    allocation, environment, runtime, documents, cases = await preflight(root, docs, gold)
    binding = {"source_run": str(root), "docs_dir": str(docs), "gold_path": str(gold),
               "gold_sha256": hashlib.sha256(gold.read_bytes()).hexdigest(),
               "manifest_sha256": allocation["manifest_sha256"], "allocation_id": allocation["allocation_id"],
               "snapshot": runtime["runtime_config_snapshot_id"], "index_generation": allocation["index_generation"]}
    if args.resume and saved != binding:
        raise ValueError("experiment binding changed; start a new output directory")
    if args.check:
        print(json.dumps({"preflight": "passed", "documents": documents, "cases": cases, **binding}, ensure_ascii=False))
        return 0
    output = (args.resume or args.output_dir or Path("var/artifacts/evals") /
              ("evaluation-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])).resolve()
    # Never place result writes over an input directory or original evaluation.
    if (output.is_relative_to(root) or output.is_relative_to(docs)
            or root.is_relative_to(output) or docs.is_relative_to(output)):
        raise ValueError("output overlaps source inputs or original evaluation")
    with experiment(output, binding, resume=bool(args.resume)):
        print(f"[2/3] 真实测评 {cases} 题；输出 {output}", flush=True)
        atomic_write_text(output / "report.md", render_report({"requested_cases": cases}, verified=False))
        verified = False
        error = None
        summary = {"requested_cases": cases}
        try:
            code = await evaluate_cases(root, allocation, environment, None, output_dir=output,
                                        gold_path=gold, expected_snapshot=binding["snapshot"])
            summary = read_json(output / "summary.json")
            if code:
                raise ValueError("evaluation incomplete; inspect failures and judge status")
            print("[3/3] 核对实时运行证据，生成报告…", flush=True)
            summary = await verify_saved_evaluation(root, output_dir=output, docs_dir=docs, gold_path=gold)
            verified = True
        except Exception as exc:
            # Do not echo exception text: provider errors may contain URLs/credentials.
            error = type(exc).__name__ + ": evaluation/verification failed; inspect per-case artifacts"
            summary = partial_summary(output, cases)
        atomic_write_text(output / "report.json", json.dumps({**summary, "completion_verified": verified,
                                                             "error": error}, ensure_ascii=False, indent=2) + "\n")
        atomic_write_text(output / "report.md", render_report(summary, verified=verified, error=error))
        print(f"报告：{output / 'report.md'}\n续跑：./scripts/eval_rag.sh --resume {output}", flush=True)
        return 0 if verified else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, help="existing ingestion run containing allocation and upload ledgers")
    parser.add_argument("--docs-dir", type=Path, help="existing frozen documents directory (QA subdirectory excluded)")
    parser.add_argument("--gold", type=Path, help="existing mapped gold JSON; must match frozen corpus")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output-dir", type=Path, help="NEW experiment directory")
    group.add_argument("--resume", type=Path, help="resume a previously created experiment")
    parser.add_argument("--check", action="store_true", help="read-only preflight, no queries, judge calls or output files")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"测评未启动：{exc}", flush=True)
        return 2
    except Exception as exc:
        print(f"测评未启动：{type(exc).__name__}；请检查现有测评服务和运行环境。", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
