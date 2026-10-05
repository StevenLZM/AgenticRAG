"""Real QueryGraph paired evaluation against the frozen original 20 multi-hop cases."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import importlib.metadata
from pathlib import Path
from uuid import uuid4

from evals.answer_judge import FormalRagasJudge, formal_judge_config
from evals.command import experiment
from evals.formal_command import score_case
from evals.gold_rebinding import _json, _write, file_sha256, load_rebound_bundle
from evals.gold_v2_models import canonical_hash
from evals.ik_evaluation_command import PhaseBudget, _preflight, implementation_fingerprint, resources
from evals.lexical_environment import assert_lexical_environment_unchanged
from evals.migration_evidence import assert_corpus_unchanged
from evals.paired_graph import GraphSearchClient, grade_diagnostics, query_once, select_graph_cases
from evals.paired_graph_runtime import arm_snapshot, compose, execute_graph, protect_sql_reads
from evals.paired_retrieval_identity import validate_identity
from evals.report import atomic_write_text
from evals.report_v2 import build_formal_summary

ARMS = ("ik", "ik_max_word")


def operation_binding(binding, case_id, arm, operation, result_sha256=None):
    return canonical_hash({"experiment": binding, "case_id": case_id, "arm": arm,
                           "operation": operation, "result_sha256": result_sha256})


async def run_pairs(*, cases, root, binding, query, score, progress=None):
    records = []
    for ordinal, case in enumerate(cases):
        for arm in (ARMS if ordinal % 2 == 0 else ARMS[::-1]):
            directory = root / arm / "cases" / case.case_id
            result = await query_once(directory / "query.json", operation_binding(binding, case.case_id, arm, "query"),
                lambda: query(case, arm, directory))
            scored = None
            if result["status"] == "completed":
                scored = await query_once(directory / "score.json", operation_binding(binding, case.case_id, arm, "score", result["result_sha256"]),
                    lambda: score(case, arm, result["result"], directory))
            records.append({"case_id": case.case_id, "arm": arm, "query": result, "score": scored})
            if progress is not None:
                progress(records)
            if result["status"] == "failed":
                # An exception outside the graph's recorded outcome indicates a
                # harness/projection failure. Do not spend the remaining budget.
                return records
    return records


def report(cases, records, identity, verification):
    arms, pairs = {}, []
    for arm in ARMS:
        subset = {r["case_id"]: r for r in records if r["arm"] == arm}
        rows = [r["score"]["result"] for r in subset.values() if r["score"] and r["score"]["status"] == "completed"]
        failures = {c.case_id: {"status": "pending" if c.case_id not in subset else "failed",
            "reason": "not_attempted" if c.case_id not in subset else (
                "query_" + subset[c.case_id]["query"]["status"] if subset[c.case_id]["query"]["status"] != "completed"
                else "score_" + (subset[c.case_id]["score"] or {}).get("status", "missing"))}
            for c in cases if c.case_id not in {r["case_id"] for r in rows}}
        arms[arm] = build_formal_summary(cases, rows, failures, identity)
        diagnostics = []
        for case in cases:
            r = subset.get(case.case_id)
            if r and r["query"]["status"] == "completed":
                trace = r["query"]["result"]
                diagnostics.append({"case_id": case.case_id, **grade_diagnostics(case, trace["grade_history"], trace["context_items"] or [])})
        arms[arm]["grader_diagnostics"] = {"observed_queries": len(diagnostics),
            "gap_recovery_count": sum(d["gap_recovery"] for d in diagnostics),
            "false_sufficient_decisions": sum(d["false_sufficient_count"] for d in diagnostics), "cases": diagnostics}
        arms[arm]["query_journal_statuses"] = dict(Counter(r["query"]["status"] for r in subset.values()))
    for case in cases:
        pair = {"case_id": case.case_id, "question": case.question}
        for arm in ARMS:
            r = next((r for r in records if r["arm"] == arm and r["case_id"] == case.case_id), None)
            trace = r["query"].get("result", {}) if r else {}
            row = (r["score"] or {}).get("result", {}) if r else {}
            pair[arm] = {"outcome": trace.get("outcome"), "answer": trace.get("answer"),
                "task_success": row.get("task_success"), "research_actions": trace.get("research_rounds"),
                "retrieval_calls": trace.get("retrieval_calls"), "total_seconds": trace.get("total_seconds"),
                "grade_diagnostics": grade_diagnostics(case, trace.get("grade_history", []), trace.get("context_items") or []) if trace else None}
        pairs.append(pair)
    return {"protocol": "real-querygraph-ik-paired-v1", "requested_queries": len(cases) * 2,
        "recorded_queries": len(records), "client_provenance": "real_query_graph", "arms": arms, "pairs": pairs,
        "verification": verification, "production_release_accepted": False,
        "limitations": ["not_http_outbox_worker_timing", "human_gold_and_judge_calibration_pending",
                        "third_party_tools_fail_closed", "pricing_not_pinned_cost_unknown", "single_stochastic_run_per_arm"]}


def save_report(root, value):
    _write(root / "report.json", value)
    lines = ["# 真实 QueryGraph：20题双臂对照", "", "固定原20道多跳题；不是 HTTP/Outbox/Worker 压测。失败及未知保留在总分母。", "",
        "| 指标 | IK smart 查询 | IK max_word 查询 |", "|---|---:|---:|"]
    def metric(arm, key):
        m = value["arms"][arm]["metrics"][key]
        return f"{m['value']:.3f} ({m['evaluated']}/{m['total']})" if m["value"] is not None else f"unknown (0/{m['total']})"
    for key in ("task_success", "answer_correctness", "grounded_factual_f1", "critical_field_accuracy", "final_context_complete_evidence"):
        lines.append(f"| {key} | {metric('ik', key)} | {metric('ik_max_word', key)} |")
    for key in ("gap_recovery_count", "false_sufficient_decisions"):
        lines.append(f"| {key} | {value['arms']['ik']['grader_diagnostics'][key]} | {value['arms']['ik_max_word']['grader_diagnostics'][key]} |")
    lines += ["", "详细答案、检索各阶段、每次 grader 的 gap、Todo/研究动作和真实用量见 report.json 及逐题目录。",
              "价格未固定，因此费用为 unknown；缺失用量不记作0。此小样本不构成上线验收。", ""]
    atomic_write_text(root / "report.md", "\n".join(lines))
    (root / "report.md").chmod(0o600)


def seal(root):
    paths = sorted(p for p in root.rglob("*.json") if p.name not in {"seal.json", "audit-resume.json"})
    _write(root / "seal.json", {"files": {str(p.relative_to(root)): file_sha256(p) for p in paths}})


def verify_seal(root):
    saved = _json(root / "seal.json")
    for name, digest in saved["files"].items():
        path = root / name
        if path.is_symlink() or root.resolve() not in path.resolve().parents or file_sha256(path) != digest:
            raise ValueError("completed experiment artifact hash mismatch")


async def run(args):
    from agentic_rag.config import Settings
    from agentic_rag.runtime.query_composition import build_query_snapshot, close_query_dependencies
    root = args.output_dir.resolve()
    if args.output_dir.is_symlink() or root == args.bundle.resolve() or root in args.bundle.resolve().parents:
        raise ValueError("unsafe output path")
    source = _json(args.source_experiment / "experiment.json")
    validate_identity(source)
    snapshot, all_cases, specs, audit = load_rebound_bundle(args.bundle)
    cases = select_graph_cases(source, all_cases)
    implementation = implementation_fingerprint()
    versions = {name: importlib.metadata.version(name) for name in ("ragas", "langgraph", "openai", "sentence-transformers", "torch")}
    settings = Settings(_env_file=args.settings_file, artifact_root=args.artifact_root, elasticsearch_url=args.es_url)
    if settings.lexical_analysis != "ik":
        raise ValueError("IK production adapter required")
    base = build_query_snapshot(settings)
    judge_config = formal_judge_config(settings)
    fingerprint_judge = FormalRagasJudge(judge_config)
    judge_identity = fingerprint_judge.fingerprint
    await fingerprint_judge.aclose()
    identity = {"protocol": "real-querygraph-ik-paired-v1", "source_sha256": source["sha256"],
        "bundle_sha256": canonical_hash(audit), "implementation": implementation, "dependencies": versions,
        "base_snapshot": base.model_dump(mode="json"), "judge_fingerprint": judge_identity,
        "node_config_sha256": file_sha256(args.node_evidence_config),
        "cases": [{"case_id": c.case_id, "question": c.question, "sha256": canonical_hash(c.model_dump(mode="json"))} for c in cases],
        "boundaries": {"memory": "disabled", "conversation_history": "empty", "external_tools": "fail_closed",
            "events_artifacts_tool_store": "experiment_local", "query_provider_limit_each": 100,
            "judge_provider_limit_per_arm": 500, "judge_case_deadline_seconds": 600,
            "query_limit": 40, "order": "AB_BA", "pricing": "unknown"}}
    continuation_source = args.continue_from
    if args.resume and continuation_source is None and (root / "experiment.json").is_file():
        prior = _json(root / "experiment.json").get("continuation")
        continuation_source = Path(prior["source_directory"]) if prior else None
    recovered = None
    if continuation_source is not None:
        from evals.paired_graph_recovery import prepare_continuation
        continuation, recovered = prepare_continuation(continuation_source, identity, cases)
        identity["continuation"] = continuation
    binding = canonical_hash(identity)
    dataset_binding = identity.get("continuation", {}).get("runtime_dataset_sha256", binding)
    snapshots = {arm: arm_snapshot(base, arm, dataset_binding, snapshot.snapshot_id) for arm in ARMS}
    with experiment(root, identity, resume=args.resume):
        root.chmod(0o700)
        (root / "experiment.json").chmod(0o600)
        if args.resume and (root / "seal.json").exists():
            verify_seal(root)
            _write(root / "audit-resume.json", {"new_queries": 0, "new_searches": 0, "new_model_requests": 0,
                "verified_cached_artifacts": True, "live_state_reverified": False})
            print("Verified completed cached experiment; zero network/model/search calls.", flush=True)
            return 0
        async with resources(args) as (_, engine, es, originals):
            protect_sql_reads(engine.sync_engine)
            baseline = _json(args.bundle / "migration-evidence.json")
            pre_budget = PhaseBudget("preflight", 250, 300)
            before, lexical_before = await _preflight(snapshot, baseline, args, engine=engine, es=es, artifacts=originals,
                implementation=implementation, budget=pre_budget)
            if continuation_source is not None:
                from evals.paired_graph_recovery import assert_source_environment
                assert_source_environment(continuation_source, lexical_before)
                imported = root / "ik" / "cases" / cases[0].case_id / "query.json"
                if not imported.exists():
                    _write(imported, {"binding": operation_binding(binding, cases[0].case_id, "ik", "query"),
                        "status": "completed", "result": recovered, "result_sha256": canonical_hash(recovered)})
                    _write(root / "continuation-proof.json", identity["continuation"])
            phase_path = root / "phases" / (uuid4().hex + ".json")
            phase = {"status": "running", "preflight": pre_budget.summary(), "environment_sha256": lexical_before["environment_sha256"]}
            _write(phase_path, phase)
            previous = list((root / "phases").glob("*.json"))
            if any(_json(p)["environment_sha256"] != phase["environment_sha256"] for p in previous):
                raise ValueError("environment changed since earlier phase")
            _write(root / "lexical-pre.json", lexical_before) if not (root / "lexical-pre.json").exists() else None
            transport = GraphSearchClient(es, index=snapshot.scope["concrete_indices"][0], user_id=args.user_id,
                generation=base.index_generation, children=snapshot.children)
            deps = None
            judges = {}
            async def query(case, arm, directory):
                nonlocal deps
                if deps is None:
                    deps, events, budget = await compose(settings, engine, transport, root)
                    query.events, query.budget = events, budget
                return await execute_graph(deps=deps, events=query.events, budget=query.budget,
                    transport=transport, snapshot=snapshots[arm], arm=arm, question=case.question, user_id=case.user_id, directory=directory)
            async def score(case, arm, trace, directory):
                if arm not in judges:
                    judge = FormalRagasJudge(judge_config)
                    judge.configure_requests(root / arm, binding)
                    judge.max_requests = 500
                    judges[arm] = judge
                async with asyncio.timeout(600):
                    score_binding = operation_binding(binding, case.case_id, arm, "judge", canonical_hash(trace))
                    return await score_case(case, trace, judges[arm], root / arm, score_binding, answer_spec=specs[case.case_id])
            def progress(records):
                last = records[-1]
                trace = last["query"].get("result", {})
                score_result = (last["score"] or {}).get("result", {})
                print(f"{len(records)}/40 {last['case_id']} {last['arm']} outcome={trace.get('outcome', last['query']['status'])} "
                      f"success={score_result.get('task_success', {}).get('value')} actions={trace.get('research_rounds')} "
                      f"seconds={trace.get('total_seconds')}", flush=True)
                save_report(root, report(cases, records, identity, {"status": "in_progress"}))
            try:
                records = await run_pairs(cases=cases, root=root, binding=binding, query=query, score=score, progress=progress)
            finally:
                if deps is not None:
                    await close_query_dependencies(deps)
                for judge in judges.values():
                    await judge.aclose()
            post_budget = PhaseBudget("postflight", 250, 300)
            try:
                after, lexical_after = await _preflight(snapshot, baseline, args, engine=engine, es=es, artifacts=originals,
                    implementation=implementation, budget=post_budget)
                assert_corpus_unchanged(before, after)
                assert_lexical_environment_unchanged(lexical_before, lexical_after)
                if implementation != implementation_fingerprint():
                    raise ValueError("implementation changed during execution")
                phase.update(status="verified", postflight=post_budget.summary())
            except Exception as error:
                phase.update(status="unverified", error_type=type(error).__name__)
            _write(phase_path, phase)
            verification = {"intervals": [_json(p) for p in sorted((root / "phases").glob("*.json"))]}
            verification["all_intervals_verified"] = all(p["status"] == "verified" for p in verification["intervals"])
            save_report(root, report(cases, records, identity, verification))
            if len(records) == 40 and verification["all_intervals_verified"]:
                seal(root)
            return 0 if (root / "seal.json").exists() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bundle", "source-experiment", "settings-file", "artifact-root", "node-evidence-config", "output-dir"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--es-url", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", type=Path,
        help="explicitly approved offline import of the single completed projection-failure query; never replay it")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except Exception as error:
        print("QueryGraph experiment stopped: " + type(error).__name__, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
