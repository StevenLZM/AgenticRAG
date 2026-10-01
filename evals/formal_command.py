"""Formal v2 evaluation on existing documents/index, with durable paid-call ledgers."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx

from agentic_rag.config import Settings
from agentic_rag.persistence.mysql import create_mysql_engine
from agentic_rag.runtime.models import EvaluationMetadata, RetrievalBudget
from agentic_rag.runtime.query_composition import build_query_snapshot
from evals.answer_fields import compare_fields, ExtractedFields, validate_extraction
from evals.answer_judge import FormalRagasJudge, evaluate_task_success, formal_judge_config, answer_requests, AnswerVerdict
from evals.collector import RuntimeCollector
from evals.command import experiment
from evals.corpus_snapshot import capture_live_snapshot, compare_live_snapshot, experiment_fingerprint, verify_originals
from evals.gold_v2_models import CorpusSnapshot, canonical_hash
from evals.gold_v2_validation import load_gold_v2, strict_json
from evals.gold_answer_spec import load_gold_answer_specs, validate_spec_for_case
from evals.grounded_facts import GroundedFact, validate_grounded_facts
from evals.grounded_judge import GroundedChecks, GroundedJudgeError, grounded_scores
from evals.report import atomic_write_text
from evals.retrieval_metrics_v2 import score_context, score_rounds


def write_json(path, value):
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return strict_json(path.read_text(encoding="utf-8"))


def validate_continuation(source, current_runtime_snapshot_id):
    if source.get("runtime_config_snapshot_id") != current_runtime_snapshot_id:
        raise ValueError("continue-from requires identical query deployment, session and retrieval configuration")


def import_query_ledger(source, target, case, snapshot_id):
    """Import a known Run identity only; paid results are re-collected live."""
    old = Path(source) / (case.case_id + ".json")
    if not old.exists():
        return False
    record = read_json(old)
    expected = canonical_hash({"case": case.model_dump(mode="json"), "snapshot": snapshot_id})
    if record.get("binding") != expected:
        raise ValueError("source query binding mismatch")
    if not record.get("run_id"):
        raise ValueError("uncertain source POST requires reconciliation")
    destination = Path(target) / old.name
    if destination.exists():
        if read_json(destination).get("run_id") != record["run_id"]:
            raise ValueError("conflicting source and target Run")
    else:
        write_json(destination, {**record, "source_ledger": str(old.resolve()), "source_sha256": hashlib.sha256(old.read_bytes()).hexdigest()})
    return True


def actual_contexts(trace):
    from agentic_rag.safety.context import DataEnvelope
    contexts = [DataEnvelope(source_label="document:" + item["document_id"], evidence_id=item["evidence_id"],
                content=item["content"], heading_path=tuple(item.get("heading_path", []))).render()
                for item in trace.get("context_items") or []]
    if "\n".join(contexts) != (trace.get("rendered_context") or ""):
        raise ValueError("actual ordered context envelopes differ from saved rendered_context")
    return contexts


def select_cases(cases, *, split="dev", smoke=None, limit=None, case_ids=None):
    selected = [c for c in cases if split == "all" or c.split == split]
    if case_ids:
        selected = [c for c in selected if c.case_id in set(case_ids)]
        if {c.case_id for c in selected} != set(case_ids):
            raise ValueError("case IDs missing or outside requested split")
    if smoke:
        groups = defaultdict(list)
        for case in sorted(selected, key=lambda c: hashlib.sha256(c.case_id.encode()).hexdigest()):
            groups[case.category].append(case)
        selected = []
        # Include the known unmapped regression if it belongs to this split.
        for group in groups.values():
            for case in list(group):
                if any(f.mapping_status == "unmapped" for f in case.required_evidence_groups_all_of):
                    selected.append(case)
                    group.remove(case)
        while len(selected) < smoke and any(groups.values()):
            for name in sorted(groups):
                if groups[name] and len(selected) < smoke:
                    selected.append(groups[name].pop(0))
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no cases selected")
    return selected


async def durable_operation(path: Path, binding: str, operation, *, require_cached=False, reuse_incomplete=False):
    if path.exists():
        record = read_json(path)
        if record.get("binding") != binding:
            raise ValueError("judge operation binding changed")
        if record.get("status") == "available":
            return record
        if reuse_incomplete and record.get("status") in {"running", "failed"}:
            return record
        raise ValueError("judge operation requires reconciliation; refusing another paid call")
    if require_cached:
        raise ValueError("cached case is missing a judge artifact")
    write_json(path, {"binding": binding, "status": "running", "started_at": datetime.now().isoformat()})
    try:
        result = await operation()
        record = {"binding": binding, "status": "available", "result": result}
    except Exception as error:
        record = {"binding": binding, "status": "failed", "result": None, "reason": type(error).__name__}
        if isinstance(error, GroundedJudgeError):
            record.update(reason=str(error), grading_evidence=error.grading_evidence)
    write_json(path, record)
    return record


class FormalHttpClient:
    def __init__(self, transport, collector, snapshot_id, evaluation, ledger, *, timeout=330, read_only=False):
        self.transport, self.collector = transport, collector
        self.snapshot_id, self.evaluation = snapshot_id, evaluation
        self.ledger, self.timeout = Path(ledger), timeout
        self.read_only = read_only

    async def query(self, case, *, require_existing=False):
        path = self.ledger / (case.case_id + ".json")
        binding = canonical_hash({"case": case.model_dump(mode="json"), "snapshot": self.snapshot_id})
        if path.exists():
            record = read_json(path)
            if record.get("binding") != binding:
                raise ValueError("query binding mismatch")
            if not record.get("run_id"):
                raise ValueError("uncertain POST requires reconciliation; refusing another submission")
        else:
            if require_existing:
                raise ValueError("cached evidence is missing query ledger; reconcile before proceeding")
            if self.read_only:
                raise ValueError("replay has no submitted Run; new queries are forbidden")
            record = {"binding": binding, "status": "submitting", "thread_id": "eval-v2-" + uuid4().hex}
            write_json(path, record)
            response = await self.transport.post("/v1/query-runs", json={"query": case.question,
                "thread_id": record["thread_id"], "wait_seconds": 0, "evaluation": self.evaluation})
            response.raise_for_status()
            run_id = response.json().get("run_id")
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError("query response omitted run_id")
            record.update(run_id=run_id, status="submitted")
            write_json(path, record)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            response = await self.transport.get("/v1/query-runs/" + record["run_id"])
            response.raise_for_status()
            value = response.json()
            if value.get("run_id") != record["run_id"] or value.get("runtime_config_snapshot_id") != self.snapshot_id:
                raise ValueError("query run/snapshot mismatch")
            if value.get("status") in {"completed", "failed", "cancelled"}:
                record.update(status=value["status"])
                write_json(path, record)
                trace = await self.collector.collect_v2(run_id=record["run_id"], user_id=case.user_id,
                    snapshot_id=self.snapshot_id, question=case.question)
                # Compare the HTTP final answer with the scoped SQL/checkpoint.
                if value.get("answer") != trace.get("public_answer"):
                    from agentic_rag.query.public_answer import project_public_answer
                    public = project_public_answer(value.get("answer"))
                    if public is None or public.model_dump(mode="json") != trace.get("public_answer"):
                        raise ValueError("HTTP/checkpoint final answer mismatch")
                return trace
            await asyncio.sleep(.5)
        record.update(status="poll_timeout")
        write_json(path, record)
        raise TimeoutError("poll timeout; retain run_id")


async def score_case(case, trace, judge, output, binding, *, require_cached=False, pricing=None, answer_spec=None):
    from evals.agent_metrics_v2 import agent_metrics
    if answer_spec is not None:
        validate_spec_for_case(answer_spec, case)
        binding = canonical_hash({"case_binding": binding, "answer_spec": answer_spec.model_dump(mode="json")})
    actual = trace.get("context_items")
    rankings = score_rounds(case, trace["retrieval_rounds"])
    context = score_context(case, actual) if actual is not None else {"fact_recall": None, "complete_evidence": None}
    answer = trace["answer"]
    fields = compare_fields(ExtractedFields(fields=[]), [])
    verdict, ragas = None, {"metrics": {}, "metadata": judge.metadata_v2}
    grounded = {"status": "not_applicable"} if answer_spec is not None else None
    prefix = output / "judges" / case.case_id
    if answer and trace["outcome"] in {"completed", "cannot_answer"}:
        if answer_spec is not None and case.answerable:
            judge.operation_ref = case.case_id + ":grounded_extract"
            async def extract_grounded():
                return {"facts": [f.model_dump(mode="json") for f in await judge.extract_grounded_facts(case.question, answer)]}
            extraction = await durable_operation(prefix / "grounded_extract.json", binding, extract_grounded,
                require_cached=require_cached, reuse_incomplete=True)
            grounded = {"status": extraction["status"], "reason": extraction.get("reason")}
            if extraction["status"] == "available":
                facts = validate_grounded_facts(case.question, answer,
                    [GroundedFact.model_validate(f) for f in extraction["result"]["facts"]])
                judge.operation_ref = case.case_id + ":grounded_judge"
                operation = await durable_operation(prefix / "grounded_judge.json", binding,
                    lambda: judge.evaluate_grounded(case.question, answer, facts, answer_spec),
                    require_cached=require_cached, reuse_incomplete=True)
                grounded = {"status": operation["status"], "reason": operation.get("reason")}
                if operation["status"] == "available":
                    saved_grounded = operation["result"]
                    grounded = grounded_scores(facts, answer_spec,
                        GroundedChecks.model_validate(saved_grounded["evidence"]["checks"]))
                    if grounded != saved_grounded:
                        raise ValueError("cached_grounded_score_changed")
        if case.critical_fields:
            judge.operation_ref = case.case_id + ":fields"
            operation = await durable_operation(prefix / "fields.json", binding, lambda: judge.extract_fields(case, answer), require_cached=require_cached, reuse_incomplete=True)
            fields = operation.get("result") or {"status": operation["status"], "all_critical_fields_pass": None,
                "critical_field_accuracy": None, "contradiction_rate": None, "reason": operation.get("reason")}
            if fields.get("extracted") is not None:
                extracted = ExtractedFields.model_validate(fields["extracted"])
                protocol = fields.get("grounding_protocol")
                validate_extraction(extracted, answer, case.critical_fields, question=case.question,
                    require_grounding=protocol == "literal_field_sources_v1")
                fields = {**compare_fields(extracted, case.critical_fields), "extracted": extracted.model_dump(mode="json"),
                          **({"grounding_protocol": protocol} if protocol else {})}
        judge.operation_ref = case.case_id + ":verdict"
        operation = await durable_operation(prefix / "verdict.json", binding,
            lambda: judge.verdict(case, answer, answer_spec) if answer_spec is not None else judge.verdict(case, answer),
            require_cached=require_cached, reuse_incomplete=True)
        verdict = operation.get("result")
        if verdict is not None:
            verdict = AnswerVerdict.model_validate(verdict).model_dump(mode="json")
        metric_path = prefix / "ragas.json"
        saved = read_json(metric_path) if metric_path.exists() else {"binding": binding, "metrics": {}}
        if saved.get("binding") != binding:
            raise ValueError("Ragas metric binding mismatch")
        if require_cached and not set(answer_requests(case.question, answer, [], case.reference_answer, case.answerable)) <= saved["metrics"].keys():
            raise ValueError("cached case is missing expected Ragas artifacts")
        def persist(name, result):
            judge.operation_ref = case.case_id + ":" + name
            saved["metrics"][name] = result
            write_json(metric_path, saved)
        # Existing failed/uncertain metrics are explicitly carried forward;
        # never automatically retry after unknown provider billing outcomes.
        ragas = await judge.evaluate_v2(question=case.question, answer=answer,
            contexts=actual_contexts(trace),
            reference_answer=case.reference_answer, answerable=case.answerable,
            completed=saved["metrics"], on_metric=persist)
    success = evaluate_task_success(case, trace["outcome"], verdict, fields,
        grounded=grounded if answer_spec is not None and case.answerable else None)
    if not answer:
        success = {"status": "evaluated", "value": 0.0, "reason": "no_actual_answer"}
    failure_class = ("system_failure" if not answer or trace["outcome"] not in {"completed", "cannot_answer"}
        else "gold_issue" if success.get("reason") == "gold_insufficient_for_extra_claims"
        else "judge_error" if success["value"] is None else "answer_incorrect" if success["value"] == 0 else None)
    from evals.costs_v2 import price_requests
    provider = trace.get("provider_requests", {})
    query_cost = price_requests(provider.get("requests", []), pricing)
    if provider.get("status") != "available":
        query_cost.update(value=None, status="unknown")
    return {"schema_version": 2, "case_id": case.case_id, "binding": binding,
            "run_id": trace["run_id"], "outcome": trace["outcome"], "route": trace["route"],
            "answer_origin": trace["answer_origin"], "answer": trace["answer"],
            "terminal_status": trace.get("terminal_status"),
            "context": context, "retrieval": rankings, "fields": fields, "verdict": verdict,
            "ragas": ragas, "task_success": success, "total_seconds": trace.get("total_seconds"),
            "grounded": grounded, "failure_class": failure_class,
            "answer_spec_sha256": canonical_hash(answer_spec.model_dump(mode="json")) if answer_spec is not None else None,
            "eligible_for_tuning": False,
            "ttft_seconds": trace.get("ttft_seconds"), "provider_usage": trace.get("provider_usage"),
            "provider_requests": provider, "query_cost": query_cost,
            "scoring_evidence_sha256": canonical_hash({str(p.relative_to(prefix)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(prefix.glob("*.json"))}),
            "research_rounds": trace["research_rounds"], "retrieval_calls": trace["retrieval_calls"],
            "agent_metrics": agent_metrics(trace),
            "review_status": case.review_status, "judging_provenance": "real_ragas_and_independent_llm"}


def implementation_hash():
    paths = sorted([*Path("evals").glob("*.py"), *Path("src/agentic_rag").rglob("*.py"), Path("pyproject.toml")])
    return canonical_hash({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


async def run_formal(args):
    from evals.report_v2 import write_formal_report
    from evals.costs_v2 import Pricing
    previous = read_json(args.resume / "experiment.json") if args.resume else {}
    continuation = getattr(args, "continue_from", None) or (Path(previous["continue_from"]) if previous.get("continue_from") else None)
    replay_path = getattr(args, "replay_from", None) or (Path(previous["replay_from"]) if previous.get("replay_from") else None) or continuation
    source = read_json(replay_path / "experiment.json") if replay_path else {}
    defaults = previous or source
    if replay_path and source.get("schema_version") != 2:
        raise ValueError("replay requires a formal v2 source experiment")
    if not defaults and (args.dataset is None or args.corpus_snapshot is None):
        raise ValueError("formal v2 requires both --dataset and --corpus-snapshot")
    if any(v is not None and v <= 0 for v in (args.limit, args.max_cases, args.max_wall_seconds, args.query_timeout, args.max_judge_requests)):
        raise ValueError("evaluation budgets must be positive")
    dataset = (args.dataset or Path(defaults.get("dataset", ""))).resolve()
    corpus_path = (args.corpus_snapshot or Path(defaults.get("corpus_snapshot", ""))).resolve()
    snapshot = CorpusSnapshot.model_validate(strict_json(corpus_path.read_text()))
    cases = load_gold_v2(dataset, snapshot)
    split = args.split or defaults.get("split", "dev")
    smoke = args.smoke if args.smoke is not None else defaults.get("smoke")
    limit = args.limit if args.limit is not None else defaults.get("limit")
    mode = args.snapshot_mode or defaults.get("snapshot_mode", "strict")
    case_filter = args.case_id or previous.get("case_ids_filter") or (source.get("case_ids") if replay_path else None)
    selected = select_cases(cases, split=split, smoke=smoke, limit=limit, case_ids=case_filter)
    spec_path = getattr(args, "answer_spec", None) or (Path(defaults["answer_spec_path"]) if defaults.get("answer_spec_path") else None)
    specs = load_gold_answer_specs(spec_path, cases, snapshot) if spec_path else {}
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest() if spec_path else None
    if len(selected) > args.max_cases:
        raise ValueError("requested cases exceed --max-cases budget")
    api_url = (args.api_url or defaults.get("api_url", "http://127.0.0.1:8000")).rstrip("/")
    settings = Settings()
    if settings.default_user_id != snapshot.scope["user_id"] or settings.elasticsearch_url.rstrip("/").replace("localhost", "127.0.0.1") != snapshot.scope["es_url"].rstrip("/").replace("localhost", "127.0.0.1"):
        raise ValueError("configured formal user/ES does not match frozen corpus")
    base = build_query_snapshot(settings)
    budget_dict = strict_json(args.retrieval_budget.read_text()) if args.retrieval_budget else defaults.get("retrieval_budget")
    budget = RetrievalBudget.model_validate(budget_dict) if budget_dict else None
    session = defaults.get("session_id", uuid4().hex)
    gold_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    evaluation = EvaluationMetadata(session_id=session, dataset_sha256=gold_hash,
                                     corpus_snapshot_id=snapshot.snapshot_id, retrieval_budget=budget)
    runtime = base.model_copy(update={"evaluation": evaluation})
    if continuation:
        validate_continuation(source, runtime.snapshot_id)
    query_snapshot_id = source["runtime_config_snapshot_id"] if replay_path else runtime.snapshot_id
    if replay_path and (source["gold_sha256"] != gold_hash or source["corpus_snapshot_id"] != snapshot.snapshot_id
                       or source["retrieval_budget"] != budget_dict or source["api_url"].rstrip("/") != api_url
                       or not {c.case_id for c in selected} <= set(source["case_ids"])):
        raise ValueError("replay source gold/corpus/budget/API/cases changed")
    pricing_path = getattr(args, "pricing", None) or (Path(defaults["pricing_path"]) if defaults.get("pricing_path") else None)
    pricing = Pricing.model_validate(read_json(pricing_path)) if pricing_path else None
    engine = create_mysql_engine(settings.mysql_dsn)
    judge = None
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=30, trust_env=False) as transport:
            print("[1/3] 校验全部原文、金标、SQL/ES快照和正式API…", flush=True)
            response = await transport.get("/v1/runtime/summary")
            response.raise_for_status()
            summary = response.json()
            if summary["runtime_config_snapshot_id"] != base.snapshot_id:
                raise ValueError("API/local runtime configuration drift; restart or use matching configuration")
            if not summary.get("evaluation_requests_enabled"):
                raise ValueError("formal API evaluation requests disabled; enable AGENTIC_RAG_ALLOW_EVALUATION_REQUESTS")
            if any(summary["dependencies"].get(k) != "available" for k in ("mysql", "elasticsearch", "redis", "checkpoints")):
                raise ValueError("formal dependencies unavailable")
            originals = await asyncio.to_thread(verify_originals, snapshot, cases)
            live = await capture_live_snapshot(snapshot, engine, settings.artifact_root)
            preflight = {**compare_live_snapshot(snapshot, live, mode=mode), **originals}
            judge = FormalRagasJudge(formal_judge_config(settings))
            judge.max_requests = args.max_judge_requests
            fingerprint = experiment_fingerprint(gold_hash, snapshot.snapshot_id, base.snapshot_id, judge.fingerprint,
                [c.case_id for c in selected], implementation_hash=implementation_hash(), budget=budget_dict,
                live_fingerprint=preflight["live_fingerprint"], mode=mode, source=source or None,
                answer_spec_sha256=spec_hash,
                pricing_fingerprint=pricing.fingerprint if pricing else None)
            binding = {"schema_version": 2, "dataset": str(dataset), "corpus_snapshot": str(corpus_path),
                "api_url": api_url, "split": split, "smoke": smoke, "limit": limit, "snapshot_mode": mode,
                "case_ids_filter": case_filter, "case_ids": [c.case_id for c in selected],
                "session_id": session, "gold_sha256": gold_hash, "corpus_snapshot_id": snapshot.snapshot_id,
                "base_runtime_snapshot_id": source.get("base_runtime_snapshot_id", base.snapshot_id), "runtime_config_snapshot_id": query_snapshot_id,
                "provider_config_fingerprint": None if replay_path and not continuation else base.provider_config_fingerprint,
                "replay_from": str(replay_path.resolve()) if replay_path and not continuation else None,
                "continue_from": str(continuation.resolve()) if continuation else None,
                "source_experiment_sha256": canonical_hash(source) if replay_path else None,
                "pricing_path": str(pricing_path.resolve()) if pricing_path else None,
                "pricing_fingerprint": pricing.fingerprint if pricing else None,
                "answer_spec_path": str(spec_path.resolve()) if spec_path else None, "answer_spec_sha256": spec_hash,
                "implementation_hash": implementation_hash(),
                "judge_fingerprint": judge.fingerprint, "retrieval_budget": budget_dict, "fingerprint": fingerprint}
            if args.check:
                print(json.dumps({"preflight": preflight, "selected_cases": len(selected), "binding": binding,
                    "judge": judge.metadata_v2, "real_query_count": 0}, ensure_ascii=False), flush=True)
                return 0
            output = (args.resume or args.output_dir or Path("var/artifacts/evals") /
                ("formal-v2-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])).resolve()
            if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (dataset.parent, corpus_path.parent)):
                raise ValueError("output overlaps frozen inputs")
            if spec_path and spec_path.resolve().is_relative_to(output):
                raise ValueError("output overlaps frozen scoring spec")
            if replay_path and (output.is_relative_to(replay_path.resolve()) or replay_path.resolve().is_relative_to(output)):
                raise ValueError("replay output overlaps source experiment")
            with ExitStack() as locks:
                if replay_path:
                    locks.enter_context(experiment(replay_path, source, resume=True))
                locks.enter_context(experiment(output, binding, resume=bool(args.resume)))
                output.chmod(0o700)
                judge.configure_requests(output, fingerprint)
                write_json(output / "preflight.json", preflight)
                collector = RuntimeCollector(engine, settings.query_checkpoint_path, settings.artifact_root)
                client = FormalHttpClient(transport, collector, query_snapshot_id,
                    evaluation.model_dump(mode="json"), output / "queries", timeout=args.query_timeout, read_only=bool(replay_path and not continuation))
                rows, failures = [], {}
                started = time.monotonic()
                print(f"[2/3] 真实正式API查询+独立Judge，共{len(selected)}题；{output}", flush=True)
                for index, case in enumerate(selected, 1):
                    if time.monotonic() - started > args.max_wall_seconds:
                        failures[case.case_id] = {"status": "pending", "reason": "evaluation_wall_budget"}
                        continue
                    result_path, trace_path = output / "cases" / (case.case_id + ".json"), output / "traces" / (case.case_id + ".json")
                    per_case_binding = canonical_hash({"experiment": fingerprint, "case": case.model_dump(mode="json")})
                    try:
                        if replay_path and not import_query_ledger(replay_path / "queries", output / "queries", case, query_snapshot_id):
                            if any((replay_path / folder / (case.case_id + ".json")).exists() for folder in ("cases", "traces")) or (replay_path / "judges" / case.case_id).exists():
                                raise ValueError("source cached evidence is missing query ledger")
                            if not continuation:
                                failures[case.case_id] = {"status": "pending", "reason": "source_run_not_submitted"}
                                continue
                        # Even a completed case must reconcile with the scoped
                        # API/SQL/checkpoint and the independently saved judges.
                        trace = await client.query(case, require_existing=(result_path.exists() or trace_path.exists()
                            or (output / "judges" / case.case_id).exists()))
                        if trace_path.exists():
                            saved_trace = read_json(trace_path)
                            if saved_trace != {"binding": per_case_binding, "trace": trace}:
                                raise ValueError("cached runtime evidence changed")
                        else:
                            if result_path.exists():
                                raise ValueError("cached case is missing trace evidence")
                            write_json(trace_path, {"binding": per_case_binding, "trace": trace})
                        row = await score_case(case, trace, judge, output, per_case_binding, require_cached=result_path.exists(),
                            pricing=pricing, answer_spec=specs.get(case.case_id))
                        if result_path.exists():
                            if read_json(result_path) != row:
                                raise ValueError("cached result differs from verified trace/judge artifacts")
                        else:
                            write_json(result_path, row)
                        rows.append(row)
                    except Exception as error:
                        path = output / "queries" / (case.case_id + ".json")
                        ledger = read_json(path) if path.exists() else {}
                        failures[case.case_id] = {"status": "failed", "reason": type(error).__name__,
                                                 "run_id": ledger.get("run_id"), "query_status": ledger.get("status")}
                    write_json(output / "failures.json", failures)
                    write_formal_report(output, selected, rows, failures, binding, preflight, pricing=pricing)
                    print(f"[{index}/{len(selected)}] {case.case_id}: {rows[-1]['outcome'] if rows and rows[-1]['case_id'] == case.case_id else 'failed'}", flush=True)
                print("[3/3] 复核全语料快照，生成评分覆盖率和最终报告…", flush=True)
                after = compare_live_snapshot(snapshot, await capture_live_snapshot(snapshot, engine, settings.artifact_root), mode=mode)
                if after["live_fingerprint"] != preflight["live_fingerprint"]:
                    raise ValueError("corpus drift during evaluation; scores cannot be verified")
                write_json(output / "failures.json", failures)
                final = write_formal_report(output, selected, rows, failures, binding, preflight, verified=True, pricing=pricing)
                print(f"报告：{output / 'report.md'}\n续跑：./scripts/eval_rag.sh --resume {output}", flush=True)
                return 0 if not failures and final["scoring_complete"] else 1
    finally:
        if judge:
            await judge.aclose()
        await engine.dispose()
