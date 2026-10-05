"""Pure retrieval diagnostics: denominators, missing evidence and paired deltas."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from statistics import mean

from evals.gold_v2_models import canonical_hash
from evals.paired_retrieval_identity import comparison_arms, validate_identity
from evals.retrieval_metrics_v2 import score_retrieval_stage

KS = (1, 3, 5, 10, 20, 30, 40)
METRICS = ["mrr@40"] + [f"{name}@{k}" for name in ("chunk_recall", "fact_recall", "complete_evidence", "binary_ndcg") for k in KS]
STAGES = ("router", "dense", "embedding", "rrf", "rerank", "parent", "final_context", "generation", "judge", "ragas", "answer_correctness")


def _aggregate(values):
    result = {}
    for name in METRICS:
        known = [row[name] for row in values if row.get(name) is not None]
        total, n = len(values), len(known)
        result[name] = {"mean": mean(known) if known else None, "n": n, "denominator": total,
                        "lower_bound": sum(known) / total if total else None,
                        "upper_bound": (sum(known) + total - n) / total if total else None}
    return result


def build_paired_report(*, cases, records, identity, verification):
    validate_identity(identity)
    arm_order = comparison_arms(identity["common"]["config"])
    expected = [{"case_id": c.case_id, "question": c.question, "sha256": canonical_hash(c.model_dump(mode="json"))} for c in cases]
    if identity["common"]["cases"] != expected:
        raise ValueError("report case identity mismatch")
    cases_by_id, rows = {c.case_id: c for c in cases}, {}
    for record in records:
        key = (record["case_id"], record["arm"])
        if key in rows or key[0] not in cases_by_id or key[1] not in arm_order:
            raise ValueError("duplicate/foreign report record")
        if record["identity_sha256"] != identity["sha256"] or record["status"] not in {"success", "failed", "uncertain"}:
            raise ValueError("report record identity/status mismatch")
        rows[key] = record
    verified = all(verification.get(k) is True for k in ("pre_corpus", "post_corpus", "pre_lexical", "post_lexical"))
    scored, pairs, groups = {}, [], {"category": defaultdict(list), "format": defaultdict(list)}
    unmapped = 0
    for case in cases:
        unmapped += int(any(not g.child_evidence_sets_any_of for g in case.required_evidence_groups_all_of))
        suffixes = {Path(s.filename).suffix.lower().lstrip(".") or "unknown" for s in case.source_documents}
        groups["category"][case.category].append(case.case_id)
        groups["format"][next(iter(suffixes)) if len(suffixes) == 1 else "mixed"].append(case.case_id)
        for arm in arm_order:
            record = rows.get((case.case_id, arm))
            values = dict.fromkeys(METRICS)
            if verified and record and record["status"] == "success" and record.get("interval_verification") == "verified":
                hits = record["hits"]
                ids = [h["child_id"] for h in hits]
                if len(ids) > 40 or len(set(ids)) != len(ids) or [h["rank"] for h in hits] != list(range(1, len(ids) + 1)):
                    raise ValueError("malformed actual ranking")
                metrics = score_retrieval_stage(case, ids, ks=KS)["metrics"]
                values = {k: metrics["mrr" if k == "mrr@40" else k] for k in METRICS}
            scored[(case.case_id, arm)] = values
        valid = verified and all(rows.get((case.case_id, a), {}).get("status") == "success"
            and rows[(case.case_id, a)].get("interval_verification") == "verified" for a in arm_order)
        deltas, changes = None, None
        if valid:
            deltas, changes = {}, {}
            for key in METRICS:
                a, b = [scored[(case.case_id, arm)][key] for arm in arm_order]
                difference = None if a is None or b is None else b - a
                deltas[key] = difference
                changes[key] = (None if difference is None else "equal" if math.isclose(difference, 0, abs_tol=1e-12)
                                else "improve" if difference > 0 else "regress")
        pairs.append({"case_id": case.case_id, "category": case.category, "valid_pair": valid, "deltas": deltas,
                      "changes": changes, "interpretation": "evidence_recall" if case.answerable else "refusal_evidence_recall_only",
                      "evidence_ids": {a: [h["child_id"] for h in rows.get((case.case_id, a), {}).get("hits", [])]
                                       for a in arm_order},
                      "interval_verification": {a: rows.get((case.case_id, a), {}).get("interval_verification", "unknown")
                                                for a in arm_order},
                      "metrics": {a: scored[(case.case_id, a)] for a in arm_order}})
    arms = {}
    for arm in arm_order:
        counts = Counter(rows.get((c.case_id, arm), {}).get("status", "pending") for c in cases)
        arms[arm] = {"metrics": _aggregate([scored[(c.case_id, arm)] for c in cases]), "status_counts": dict(counts),
                     "elapsed_seconds": sum(r.get("elapsed_seconds") or 0 for r in records if r["arm"] == arm),
                     "graded_ndcg": {"status": "not_available", "value": None}}
    grouped, macros = {}, {}
    for axis, buckets in groups.items():
        grouped[axis], macros[axis] = {}, {}
        for name, ids in sorted(buckets.items()):
            grouped[axis][name] = {"count": len(ids), "arms": {a: _aggregate([scored[(cid, a)] for cid in ids]) for a in arms}}
        for arm in arms:
            macros[axis][arm] = {}
            for metric in METRICS:
                means = [g["arms"][arm][metric]["mean"] for g in grouped[axis].values()]
                valid_means = [v for v in means if v is not None]
                macros[axis][arm][metric] = {"mean": mean(valid_means) if valid_means else None,
                                            "valid_groups": len(valid_means), "total_groups": len(means)}
    return {"protocol": "ik-bm25-report-v1", "identity_sha256": identity["sha256"], "verification": verification,
            "requested_cases": len(cases), "requested_requests": 2 * len(cases), "actual_attempts": len(records),
            "successful_requests": sum(r["status"] == "success" for r in records),
            "valid_pairs": sum(p["valid_pair"] for p in pairs), "qrel_unmapped_cases": unmapped,
            "completion_verified": verified and all(p["valid_pair"] for p in pairs),
            "arm_order": list(arm_order), "arms": arms, "pairs": pairs, "groups": grouped, "macro": macros,
            "stages": dict.fromkeys(STAGES, "not_executed"), "eligible_for_tuning": False, "production_release_accepted": False,
            "limitations": ["template_gold_not_natural_language_representative", "native_tied_ranks_may_fluctuate",
                "local_latency_not_production_p95", "unjudged_not_proven_irrelevant", "bounds_are_not_missing_as_wrong",
                "binary_ndcg_is_annotated_evidence_diagnostic", "human_gold_and_judge_calibration_pending"]}


def render_paired_report(report):
    arm_order = report.get("arm_order", ["standard", "ik"])
    labels = {"standard": "standard", "ik": "IK (ik_smart)", "ik_max_word": "IK (ik_max_word)"}
    left, right = [labels[a] for a in arm_order]
    lines = [f"# {left} / {right} BM25 配对诊断", "", f"严格核验完成：{report['completion_verified']}",
        f"题目：{report['requested_cases']}；实际尝试：{report['actual_attempts']}/{report['requested_requests']}；"
        f"有效配对：{report['valid_pairs']}；unmapped 题：{report['qrel_unmapped_cases']}", "",
        f"配对差值方向：{right} − {left}。", "",
        f"| 指标 | {left} 条件均值 (n/N) | {right} 条件均值 (n/N) |", "| --- | --- | --- |"]
    for name in ("chunk_recall@40", "fact_recall@40", "complete_evidence@40", "mrr@40", "binary_ndcg@40"):
        cells = []
        for arm in arm_order:
            value = report["arms"][arm]["metrics"][name]
            cells.append(f"{value['mean']} ({value['n']}/{value['denominator']})")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines.extend(["", "失败、uncertain、pending 与 unmapped 保留在覆盖率和界限的分母；未标注不等于已证明无关。",
        "MRR@40 为首次完整覆盖某事实充分集合的倒数排名。graded NDCG：N/A。", "",
        "Dense、RRF、rerank、Parent、最终上下文、回答、Ragas 和 Answer Correctness：not_executed。",
        "这不是完整 RAG 或生产上线成绩；eligible_for_tuning=false；production_release_accepted=false。", "",
        "模板题不代表自然中文问法；并列排名可能波动；单次本地延时不是生产 P95。",
        "逐题差值、分类与格式宏平均、失败状态和保守上下界见同目录 report.json。"])
    return "\n".join(lines) + "\n"
