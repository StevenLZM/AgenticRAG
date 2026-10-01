"""Answer-first reports with explicit denominators, failures, and macro groups."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

from evals.answer_judge import answer_requests

from evals.report import atomic_write_text

ANSWER_METRICS = ("task_success", "answer_correctness", "factual_correctness_precision", "factual_correctness_recall",
    "grounded_factual_precision", "grounded_factual_recall", "grounded_factual_f1",
    "factual_correctness_f1", "critical_field_accuracy", "all_critical_fields_pass", "contradiction_rate",
    "faithfulness", "answer_relevancy", "context_precision", "context_recall", "correct_refusal",
    "final_context_fact_recall", "final_context_complete_evidence")


def _values(row):
    values = {name: item.get("value") if item.get("status") == "available" else None
              for name, item in row.get("ragas", {}).get("metrics", {}).items()}
    values["task_success"] = row.get("task_success", {}).get("value")
    grounded = row.get("grounded") or {}
    values.update({"grounded_factual_" + key: grounded.get(key) if grounded.get("status") == "available" else None
                   for key in ("precision", "recall", "f1")})
    values.update({name: row.get("fields", {}).get(name) for name in (
        "critical_field_accuracy", "all_critical_fields_pass", "contradiction_rate")})
    values["final_context_fact_recall"] = row.get("context", {}).get("fact_recall")
    values["final_context_complete_evidence"] = row.get("context", {}).get("complete_evidence")
    values.update({f"final_context.{name}": value for name, value in row.get("context", {}).get("metrics", {}).items()})
    for stage, metrics in row.get("retrieval", {}).get("union", {}).items():
        for name in ("fact_recall", "complete_evidence"):
            values[f"{stage}.union_{name}"] = metrics.get(name)
    # Macro across independent retrieval calls, not a concatenated fake rank.
    stages = defaultdict(lambda: defaultdict(list))
    for batch in row.get("retrieval", {}).get("rounds", []):
        for stage, result in batch.items():
            for key, score in result.get("metrics", {}).items():
                stages[stage][key]
                if score is not None:
                    stages[stage][key].append(score)
    for stage, metrics in stages.items():
        for key, scores in metrics.items():
            values[f"{stage}.per_retrieval_{key}"] = sum(scores) / len(scores) if scores else None
    return values


def _average(rows, keys, total):
    data = [_values(r) for r in rows]
    return {key: {"value": (sum(v) / len(v)) if (v := [d[key] for d in data if d.get(key) is not None]) else None,
                  "evaluated": len(v), "total": total, "coverage": len(v) / total if total else None,
                  "missing": total - len(v)} for key in sorted(keys)}


def _company(case):
    match = re.match(r"(S\d+)-", case.case_id)
    return match[1] if match else "historical-documents"


def _cluster_ci(cases, rows, seed=20260930):
    by_case = {r["case_id"]: r for r in rows}
    clusters = defaultdict(list)
    for case in cases:
        value = by_case.get(case.case_id, {}).get("task_success", {}).get("value")
        clusters[_company(case)].append(float(value == 1))
    means = [sum(v) / len(v) for v in clusters.values()]
    if len(means) < 2:
        return {"status": "insufficient_company_clusters", "lower": None, "upper": None}
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(1000))
    return {"status": "available", "lower": samples[24], "upper": samples[974],
            "clusters": len(means), "method": "company_macro_bootstrap_95pct_conservative", "seed": seed}


def build_formal_summary(cases, rows, failures, binding):
    ids = {c.case_id for c in cases}
    if len({r["case_id"] for r in rows}) != len(rows) or any(r["case_id"] not in ids for r in rows):
        raise ValueError("duplicate/foreign report result")
    keys = set(ANSWER_METRICS) | {name for row in rows for name in _values(row)}
    metrics = _average(rows, keys, len(cases))
    scored_ids = {r["case_id"] for r in rows}
    failed = {k: v for k, v in failures.items() if k not in scored_ids and v.get("status") != "pending"}
    result = {"schema_version": 2, "requested_cases": len(cases), "observed_cases": len(rows),
        "failed_cases": len(failed), "pending_cases": len(ids - scored_ids - failed.keys()),
        "failure_reasons": dict(Counter(f.get("reason", "unknown") for f in failed.values())),
        "outcomes": dict(Counter(r["outcome"] for r in rows)), "metrics": metrics,
        "conservative_task_success_rate": sum(r.get("task_success", {}).get("value") == 1 for r in rows) / len(cases),
        "scoring_complete": len(rows) == len(cases) and all(scoring_complete(c, next(r for r in rows if r["case_id"] == c.case_id)) for c in cases),
        "experiment": binding, "production_release_accepted": False,
        "eligible_for_tuning": False,
        "failure_classes": dict(Counter(r["failure_class"] for r in rows if r.get("failure_class"))),
        "release_blockers": ["human_gold_review_pending", "judge_calibration_pending", "held_out_test_acceptance_pending"],
        "company_cluster_ci": _cluster_ci(cases, rows),
        "grading_policy": "unknown_is_null; failures_remain_in_total; conservative_success_counts_unknown_as_not_confirmed"}
    groups = {}
    for dimension in ("category", "company", "format", "route"):
        selected = defaultdict(list)
        by_id = {r["case_id"]: r for r in rows}
        for case in cases:
            key = (case.category if dimension == "category" else _company(case) if dimension == "company"
                   else "+".join(sorted({Path(s.filename).suffix.lower().lstrip(".") for s in case.source_documents}))
                   if dimension == "format" else by_id.get(case.case_id, {}).get("route") or "unobserved")
            selected[key].append(case.case_id)
        groups[dimension] = {key: _average([r for r in rows if r["case_id"] in set(members)], ANSWER_METRICS, len(members))
                             for key, members in sorted(selected.items())}
    result["groups"] = groups
    result["category_macro"] = {name: sum(v) / len(v) if (v := [g[name]["value"] for g in groups["category"].values()
                                                             if g[name]["value"] is not None]) else None for name in ANSWER_METRICS}
    result["category_macro_coverage"] = {name: {"evaluated_categories": sum(g[name]["value"] is not None for g in groups["category"].values()),
                                               "total_categories": len(groups["category"])} for name in ANSWER_METRICS}
    by_id = {r["case_id"]: r for r in rows}
    categories = defaultdict(list)
    for case in cases:
        categories[case.category].append(float(by_id.get(case.case_id, {}).get("task_success", {}).get("value") == 1))
    result["conservative_category_task_success"] = sum(sum(v) / len(v) for v in categories.values()) / len(categories)
    durations = sorted(r["total_seconds"] for r in rows if r.get("total_seconds") is not None)
    result["latency_seconds"] = {"evaluated": len(durations), "total": len(cases),
        "p50": durations[max(0, math.ceil(len(durations) * .5) - 1)] if durations else None,
        "p95": durations[max(0, math.ceil(len(durations) * .95) - 1)] if durations else None}
    result["ttft_seconds"] = {"value": None, "status": "unknown_not_durably_observed"}
    query_costs = [r.get("query_cost", {}) for r in rows]
    known_costs = [c["value"] for c in query_costs if c.get("value") is not None]
    result["cost"] = {"value": sum(known_costs) if len(known_costs) == len(cases) else None,
        "known_subtotal": sum(c.get("known_subtotal", 0) for c in query_costs), "evaluated": len(known_costs),
        "total": len(cases), "currency": next((c.get("currency") for c in query_costs if c.get("currency")), None),
        "basis": "fixed_tariff_estimate_from_provider_usage", "scope": "query_provider_api_only",
        "status": "available" if len(known_costs) == len(cases) else "unknown_or_incomplete"}
    stage_counts = defaultdict(Counter)
    for row in rows:
        for stage, counts in row.get("retrieval", {}).get("stage_counts", {}).items():
            stage_counts[stage].update(counts)
    result["retrieval_stage_execution"] = {stage: {**counts, "total_calls": sum(counts.values())} for stage, counts in stage_counts.items()}
    result["agent_diagnostics"] = {name: {"value": sum(values) / len(values) if (values := [r.get("agent_metrics", {}).get(name)
        for r in rows if r.get("agent_metrics", {}).get(name) is not None]) else None, "evaluated": len(values), "total": len(cases)}
        for name in ("research_rounds", "retrieval_calls", "action_count", "dependency_violations", "repeated_retrieval_rate", "subagent_chunk_overlap", "budget_exhausted")}
    result["scoring_failures"] = {r["case_id"]: {name: value for name, value in r.get("ragas", {}).get("metrics", {}).items()
        if value.get("status") != "available"} for r in rows if any(v.get("status") != "available" for v in r.get("ragas", {}).get("metrics", {}).values())}
    for row in rows:
        if (row.get("grounded") or {}).get("status") not in {None, "available", "not_applicable"}:
            result["scoring_failures"].setdefault(row["case_id"], {})["grounded_factual"] = row["grounded"]
        fields = row.get("fields") or {}
        if fields.get("status") in {"failed", "running"}:
            result["scoring_failures"].setdefault(row["case_id"], {})["critical_fields"] = {
                "status": fields["status"], "reason": fields.get("reason")}
    return result


def scoring_complete(case, row):
    value = row.get("task_success", {}).get("value")
    if value not in (0, 1):
        return False
    if not row.get("answer") or row["outcome"] not in {"completed", "cannot_answer"}:
        # Runtime failures are evaluated as task failures, never as a positive
        # empty-metrics success. Their inapplicable LLM metrics remain null.
        return value == 0
    expected = answer_requests(case.question, row["answer"], [], case.reference_answer, case.answerable)
    metrics = row.get("ragas", {}).get("metrics", {})
    for name in expected:
        item = metrics.get(name, {})
        number = item.get("value")
        if (item.get("status") != "available" or type(number) not in (int, float) or not math.isfinite(number)
                or not (-1 <= number <= 1 if name == "answer_relevancy" else 0 <= number <= 1)):
            return False
    if row.get("verdict") is None:
        return False
    if row.get("answer_spec_sha256") and case.answerable:
        grounded = row.get("grounded") or {}
        if grounded.get("status") != "available" or any(type(grounded.get(k)) not in (int, float)
                or not math.isfinite(grounded[k]) or not 0 <= grounded[k] <= 1 for k in ("precision", "recall", "f1")):
            return False
    if case.critical_fields and row.get("fields", {}).get("all_critical_fields_pass") not in (0, 1):
        return False
    return True


def write_formal_report(output, cases, rows, failures, binding, preflight, *, verified=False, pricing=None):
    summary = build_formal_summary(cases, rows, failures, binding)
    from evals.costs_v2 import price_requests
    from evals.judge_requests import reconcile_judge_requests
    requests = reconcile_judge_requests(output, binding.get("fingerprint"))
    summary["judge_cost"] = price_requests(requests, pricing)
    if not requests and not any(r.get("answer") and r["outcome"] in {"completed", "cannot_answer"} for r in rows):
        summary["judge_cost"].update(value=0.0, status="available")
    summary["evaluation_cost"] = {"value": summary["cost"]["value"] + summary["judge_cost"]["value"]
        if summary["cost"]["value"] is not None and summary["judge_cost"]["value"] is not None else None,
        "currency": pricing.currency if pricing else None, "basis": "fixed_tariff_estimate_not_invoice"}
    summary.update(snapshot_verified=verified, preflight=preflight,
                   real_query_count=len(rows) if verified else 0)
    atomic_write_text(output / "report.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write_text(output / "results.jsonl", "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
    lines = ["# 正式 Agentic RAG 真实评测 v2", "",
        "状态：" + ("快照核验通过" if verified else "运行中 / 尚未完成末尾快照核验"), "",
        f"请求 {len(cases)} 题；已观察 {len(rows)}；失败 {summary['failed_cases']}；待执行 {summary['pending_cases']}。",
        "真实正式Query API + 独立LLM/Ragas；不生成文档、不上传、不重建索引。", "",
        "grounded_factual 为原文绑定的项目自定义主事实指标；原Ragas指标仅作未校准诊断，不混称同一算法。",
        "裁判抽错记unknown，系统运行失败保留为任务失败；评分草案未人工审核，不可用于自动选参。", "",
        "## 最终答案与金标比较", "", "| 指标 | 得分 | 有效 n / 总 n |", "| --- | ---: | ---: |"]
    for name in ANSWER_METRICS:
        item = summary["metrics"][name]
        lines.append(f"| {name} | {item['value'] if item['value'] is not None else 'unknown / 不适用'} | {item['evaluated']} / {item['total']} |")
    lines.extend(["", f"全量保守整题通过率：{summary['conservative_task_success_rate']:.4f}；未知/失败样本仍在分母。",
        f"全类别保守宏平均：{summary['conservative_category_task_success']:.4f}；各指标的类别覆盖率见report.json。",
        "人工金标审核、Judge校准、独立test验收尚未完成，不能据此宣称生产发布通过。", "",
        "## 检索原因定位", "", "Child各阶段与Parent分别评分；跨Child内层AND/外层OR；多跳必须覆盖全部事实。",
        "per_retrieval指标是逐次检索的宏平均；union仅是多轮覆盖，没有虚构全局MRR/RRF排名。",
        "graded NDCG缺人工qrels时为null，未审核候选不当作无关。", "",
        "| 阶段指标 | 得分 | 有效 n / 总 n |", "| --- | ---: | ---: |"])
    for name, item in summary["metrics"].items():
        if "." in name:
            lines.append(f"| {name} | {item['value']} | {item['evaluated']} / {item['total']} |")
    lines.extend(["", "## 可靠性与恢复", "", "```json", json.dumps({"outcomes": summary["outcomes"],
        "failures": failures, "failure_classes": summary["failure_classes"], "scoring_failures": summary["scoring_failures"], "latency_seconds": summary["latency_seconds"],
        "query_cost": summary["cost"], "judge_cost": summary["judge_cost"], "retrieval_stage_execution": summary["retrieval_stage_execution"]}, ensure_ascii=False, indent=2), "```", "",
        "费用按冻结单价与provider实测usage估算，查询与裁判分开；不代表账单，不含本地GPU/ES/MySQL成本。缺失usage或未报价模型保持unknown。",
        "完整分组/公司bootstrap见report.json。原文事实→各轮Child/Parent候选→实际裁剪上下文→最终回答，见本地cases/、traces/、judges/。",
        "报告正文不导出个人原文或回答；完整本地轨迹目录权限700。评分失败/未知响应不自动重复收费调用。", "",
        f"续跑：`./scripts/eval_rag.sh --resume {output}`", ""])
    atomic_write_text(output / "report.md", "\n".join(lines))
    return summary
