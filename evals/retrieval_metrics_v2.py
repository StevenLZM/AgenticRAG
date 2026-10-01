"""Child/Parent rankings and sufficient-evidence coverage; never fake fusion."""
from __future__ import annotations

import math
from collections import defaultdict, Counter

from evals.formal_gold import normalize
from evals.gold_v2_models import GoldCaseV2

DEFAULT_KS = (1, 3, 5, 10, 20, 30, 50)


def evidence_sets(case: GoldCaseV2, level: str):
    if level not in {"child", "parent"}:
        raise ValueError("level must be child or parent")
    groups = {}
    for fact in case.required_evidence_groups_all_of:
        key = fact.equivalent_fact_id or fact.fact_id
        alternatives = fact.child_evidence_sets_any_of if level == "child" else [[p] for p in fact.parent_ids_any_of]
        groups.setdefault(key, []).extend(set(s) for s in alternatives)
    return groups


def covered_facts(case, ids, level="child"):
    groups = evidence_sets(case, level)
    present = set(ids)
    return {key for key, alternatives in groups.items() if any(s <= present for s in alternatives)}


def _ndcg(ranked, qrels, k, *, graded):
    def gain(rel):
        return 2 ** rel - 1 if graded else float(rel > 0)
    seen, dcg = set(), 0.0
    for rank, item in enumerate(ranked[:k], 1):
        if item not in seen:
            dcg += gain(qrels.get(item, 0)) / math.log2(rank + 1)
            seen.add(item)
    ideal = sum(gain(r) / math.log2(i + 2) for i, r in enumerate(sorted(qrels.values(), reverse=True)[:k]))
    return dcg / ideal if ideal else None


def score_retrieval_stage(case: GoldCaseV2, ranked_ids: list[str], ks=DEFAULT_KS, level="child") -> dict:
    if any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("K must be a positive integer")
    groups = evidence_sets(case, level)
    unmapped = sum(not alternatives for alternatives in groups.values())
    eligible = len(groups) - unmapped
    qrels = {chunk: 1 for alternatives in groups.values() for s in alternatives for chunk in s}
    graded = {q.chunk_id: q.relevance for q in case.graded_qrels if q.level == level}
    metrics = {}
    first = next((i for i in range(1, len(ranked_ids) + 1) if covered_facts(case, ranked_ids[:i], level)), None)
    metrics["mrr"] = None if unmapped else 1 / first if first else 0.0
    for k in ks:
        present = set(ranked_ids[:k])
        covered = len(covered_facts(case, present, level))
        metrics[f"chunk_recall@{k}"] = None if unmapped else len(present & qrels.keys()) / len(qrels) if qrels else None
        metrics[f"fact_recall@{k}"] = None if unmapped else covered / len(groups)
        metrics[f"complete_evidence@{k}"] = None if unmapped else float(covered == len(groups))
        metrics[f"binary_ndcg@{k}"] = None if unmapped else _ndcg(ranked_ids, qrels, k, graded=False)
        # A missing judgment is not an irrelevant result. Do not score an
        # unjudged prefix or derive the ideal from the observed hit subset.
        metrics[f"graded_ndcg@{k}"] = (_ndcg(ranked_ids, graded, k, graded=True)
            if graded and all(i in graded for i in ranked_ids[:k]) else None)
    return {"status": "qrel_unmapped" if unmapped else "evaluated", "level": level,
            "ranked_count": len(ranked_ids), "eligible_facts": eligible, "total_facts": len(groups),
            "qrel_unmapped": unmapped, "graded_qrels_status": "human_judged" if graded else "unjudged",
            "metrics": metrics}


def score_context(case: GoldCaseV2, items: list[dict]) -> dict:
    """Check the *actual cropped* context, not hydrated Parent IDs alone."""
    required, covered = set(), set()
    for group in case.required_evidence_groups_all_of:
        key = group.equivalent_fact_id or group.fact_id
        required.add(key)
        fragments = [normalize(" ".join(item.get("heading_path", [])) + " " + item["content"])
                     for item in items if item.get("parent_id") in group.parent_ids_any_of]
        if any(all(normalize(anchor) in fragment for anchor in group.source_anchors) for fragment in fragments):
            covered.add(key)
    coverage, ranked = set(), []
    prefixes = []
    for ordinal, item in enumerate(items):
        fragment = normalize(" ".join(item.get("heading_path", [])) + " " + item["content"])
        facts = {g.equivalent_fact_id or g.fact_id for g in case.required_evidence_groups_all_of
                 if item.get("parent_id") in g.parent_ids_any_of and all(normalize(a) in fragment for a in g.source_anchors)}
        coverage.update(facts)
        prefixes.append(set(coverage))
        ranked.append(item["parent_id"] if facts else f"__cropped_without_fact_{ordinal}")
    qrels = {p: 1 for g in case.required_evidence_groups_all_of for p in g.parent_ids_any_of}
    first = next((i for i, facts in enumerate(prefixes, 1) if facts), None)
    metrics = {"mrr": 1 / first if first else 0.0}
    for k in DEFAULT_KS:
        facts = prefixes[min(k, len(prefixes)) - 1] if prefixes else set()
        metrics.update({f"fact_recall@{k}": len(facts) / len(required), f"complete_evidence@{k}": float(facts == required),
                        f"binary_ndcg@{k}": _ndcg(ranked, qrels, k, graded=False), f"graded_ndcg@{k}": None})
    return {"fact_recall": len(covered) / len(required), "complete_evidence": float(covered == required),
            "covered_fact_ids": sorted(covered), "required_fact_ids": sorted(required), "metrics": metrics}


def score_rounds(case: GoldCaseV2, rounds: list[dict], ks=DEFAULT_KS) -> dict:
    result, unions, gains = [], {}, {}
    statuses = defaultdict(Counter)
    for batch in rounds:
        scored = {}
        rankings = {**batch.get("stages", {}), "parent": batch.get("parent_ids", [])}
        for name, ids in rankings.items():
            level = "parent" if name == "parent" else "child"
            scored[name] = score_retrieval_stage(case, ids, ks, level)
            status = batch.get("stage_status", {}).get(name, "available")
            statuses[name][status] += 1
            scored[name]["execution_status"] = status
            if status in {"failed", "fallback", "unknown"}:
                scored[name]["status"] = status
                scored[name]["metrics"] = {key: None for key in scored[name]["metrics"]}
            if status in {"failed", "unknown"}:
                continue
            before = covered_facts(case, unions.get(name, set()), level)
            unions.setdefault(name, set()).update(ids)
            after = covered_facts(case, unions[name], level)
            gains.setdefault(name, []).append(len(after - before))
        result.append(scored)
    union = {}
    for name, ids in unions.items():
        groups = evidence_sets(case, "parent" if name == "parent" else "child")
        unmapped = sum(not alternatives for alternatives in groups.values())
        count = len(covered_facts(case, ids, "parent" if name == "parent" else "child"))
        union[name] = {"fact_recall": None if unmapped else count / len(groups),
                       "complete_evidence": None if unmapped else float(count == len(groups)),
                       "qrel_unmapped": unmapped, "unique_chunks": len(ids)}
    for name in statuses.keys() - unions.keys():
        union[name] = {"fact_recall": None, "complete_evidence": None, "status": "all_calls_failed"}
    return {"rounds": result, "union": union, "new_fact_gain": gains, "stage_counts": dict(statuses), "ranking_scope": "per-retrieval"}
