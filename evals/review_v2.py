"""Blind human-review handoff and calibration audit, never automatic approval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

from evals.gold_v2_models import CorpusSnapshot, canonical_hash
from evals.gold_v2_validation import load_gold_v2, strict_json
from evals.report import atomic_write_text


def review_sample(cases):
    groups = defaultdict(list)
    for case in sorted(cases, key=lambda c: canonical_hash(c.case_id)):
        groups[case.category].append(case)
    selected = {c.case_id: c for group in groups.values() for c in group[:20]}
    selected.update({c.case_id: c for c in cases if c.category in {"multi_hop", "unanswerable"}
                     or not c.case_id.startswith("S")})
    return [selected[key] for key in sorted(selected)]


def _gold_binding(case):
    return canonical_hash(case.model_dump(mode="json"))


def gold_review_queue(cases):
    return [{"case_id": c.case_id, "case_sha256": _gold_binding(c), "gold": c.model_dump(mode="json"),
             "reviewer": None, "reviewed_at": None, "decision": "pending", "reason": None,
             "instructions": "Human: check original answer, units/year/entity, source anchors and sufficient Child AND/OR sets. Reject errors; do not edit frozen gold in place."}
            for c in review_sample(cases)]


def _has_human_review(row):
    if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("reviewer", "reviewed_at", "reason")):
        return False
    try:
        return datetime.fromisoformat(row["reviewed_at"]).utcoffset() is not None
    except ValueError:
        return False


def _unique(records):
    result = {r["case_id"]: r for r in records}
    if len(result) != len(records):
        raise ValueError("duplicate human review case")
    return result


def audit_gold(cases, records):
    expected = {c.case_id: c for c in review_sample(cases)}
    approved, rejected = [], []
    for key, row in _unique(records).items():
        if key not in expected or row.get("case_sha256") != _gold_binding(expected[key]):
            raise ValueError("gold review binding mismatch")
        if row.get("gold") != expected[key].model_dump(mode="json"):
            raise ValueError("gold review content binding mismatch")
        if row.get("decision") not in {"pending", "approve", "reject"}:
            raise ValueError("invalid human review decision")
        if _has_human_review(row):
            if row["decision"] == "approve":
                approved.append(key)
            elif row["decision"] == "reject":
                rejected.append(key)
    return {"required_sample": len(expected), "approved": len(approved), "rejected": rejected,
            "pending": sorted(expected.keys() - set(approved) - set(rejected)),
            "sample_review_complete": len(approved) == len(expected),
            "scope": "20_per_category_all_multihop_unanswerable_historical_not_full_gold_review",
            "production_release_accepted": False}


def calibration_queue(cases, results, experiment_fingerprint):
    by_id = {c.case_id: c for c in cases}
    queue = []
    for row in results:
        case = by_id[row["case_id"]]
        if case.split == "test":
            raise ValueError("held-out test must not calibrate the judge")
        answer = row.get("answer") or ""
        if not answer:
            continue
        queue.append({"case_id": case.case_id, "case_sha256": _gold_binding(case),
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(), "experiment_fingerprint": experiment_fingerprint,
            "question": case.question, "reference_answer": case.reference_answer, "answer": answer,
            "critical_fields": [f.model_dump(mode="json") for f in case.critical_fields],
            "source_facts": [f.model_dump(mode="json") for f in case.required_evidence_groups_all_of],
            "reviewer": None, "reviewed_at": None, "human_fully_correct": None, "reason": None})
    return queue


def audit_calibration(cases, results, records, experiment_fingerprint):
    expected = _unique(calibration_queue(cases, results, experiment_fingerprint))
    by_id, by_case = _unique(results), {c.case_id: c for c in cases}
    labels, categories = [], Counter()
    for key, row in _unique(records).items():
        if key not in expected or any(row.get(k) != value for k, value in expected[key].items()
            if k not in {"reviewer", "reviewed_at", "human_fully_correct", "reason"}):
            raise ValueError("calibration binding mismatch")
        verdict = by_id[key].get("verdict")
        label = row.get("human_fully_correct")
        if label is not None and type(label) is not bool:
            raise ValueError("human label must be a boolean or null")
        if type(label) is bool and _has_human_review(row) and verdict is not None:
            predicted = verdict["fully_correct"] and not verdict["contradictions"] and not verdict["missing_facts"]
            labels.append((label, predicted))
            categories[by_case[key].category] += 1
    n = len(labels)
    agreement = sum(h == p for h, p in labels) / n if n else None
    human_rate = sum(h for h, _ in labels) / n if n else 0
    predicted_rate = sum(p for _, p in labels) / n if n else 0
    chance = human_rate * predicted_rate + (1 - human_rate) * (1 - predicted_rate)
    complete = n >= 100 and len({h for h, _ in labels}) == 2 and all(categories[c] >= 20 for c in
        ("single_hop", "multi_hop", "table_calculation", "unanswerable", "section_comprehension"))
    return {"evaluated": n, "exported": len(expected), "agreement": agreement,
            "cohen_kappa": (agreement - chance) / (1 - chance) if n and chance < 1 else None,
            "false_positive": sum(not h and p for h, p in labels), "false_negative": sum(h and not p for h, p in labels),
            "category_counts": dict(categories), "calibration_sample_complete": complete,
            "acceptance_threshold_approved": False, "production_release_accepted": False,
            "limitations": "Observational human/Judge agreement, not approved release thresholds; no invented labels or automatic model tuning."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gold-export", "gold-audit", "calibration-export", "calibration-audit"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-snapshot", type=Path, required=True)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("review output already exists; preserve previous reviews")
    cases = load_gold_v2(args.dataset, CorpusSnapshot.model_validate(strict_json(args.corpus_snapshot.read_text())))
    if args.action.startswith("calibration"):
        if not args.experiment:
            parser.error("calibration requires --experiment")
        experiment = strict_json((args.experiment / "experiment.json").read_text())
        if experiment["gold_sha256"] != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
            raise ValueError("experiment/gold mismatch")
        results = [strict_json(line) for line in (args.experiment / "results.jsonl").read_text().splitlines() if line.strip()]
        value = calibration_queue(cases, results, experiment["fingerprint"])
    else:
        value = gold_review_queue(cases)
    if args.action.endswith("audit"):
        if not args.reviews:
            parser.error("audit requires --reviews")
        records = [strict_json(line) for line in args.reviews.read_text().splitlines() if line.strip()]
        value = (audit_gold(cases, records) if args.action == "gold-audit" else
                 audit_calibration(cases, results, records, experiment["fingerprint"]))
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    else:
        text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value)
    atomic_write_text(args.output, text)
    args.output.chmod(0o600)
    print(f"Saved {args.action} to {args.output}; no gold, index or release settings changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
