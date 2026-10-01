"""Progressive candidate experiments; never use held-out test for tuning."""
from __future__ import annotations

from itertools import product
import argparse
import json
import math
from pathlib import Path

from agentic_rag.runtime.models import RetrievalBudget


def progressive_budgets(stage, base):
    base = RetrievalBudget.model_validate(base).model_dump()
    if stage == "recall":
        updates = [{"dense_k": dense, "bm25_k": bm25} for dense, bm25 in product((20, 40, 80), repeat=2)]
    elif stage == "rrf":
        updates = [{"rrf_k": k} for k in (30, 50, 80)]
    elif stage == "rerank":
        updates = [{"rerank_k": k} for k in (10, 20, 30) if k <= base["rrf_k"]]
    elif stage == "parent":
        updates = [{"parent_k": k} for k in (6, 10, 15) if k <= base["rerank_k"]]
    else:
        raise ValueError("unknown tuning stage")
    return [RetrievalBudget.model_validate({**base, **update}) for update in updates]


def _comparable(reports, expected_split):
    if not reports:
        raise ValueError("no reports to compare")
    keys = ("gold_sha256", "corpus_snapshot_id", "base_runtime_snapshot_id", "judge_fingerprint", "case_ids",
            "implementation_hash", "provider_config_fingerprint", "pricing_fingerprint")
    baseline = reports[0]["experiment"]
    for report in reports:
        binding = report["experiment"]
        if binding.get("replay_from"):
            raise ValueError("replayed historical outputs cannot select new deployment parameters")
        if binding["split"] == "test":
            raise ValueError("held-out test cannot select tuning parameters")
        if binding["split"] != expected_split or any(binding.get(k) != baseline.get(k) for k in keys):
            raise ValueError("incomparable corpus/model/gold/judge/cases/split")


def pareto_frontier(reports, *, split="dev"):
    _comparable(reports, split)
    eligible, excluded = [], []
    for report in reports:
        if (report.get("eligible_for_tuning") is not True or not report.get("snapshot_verified") or not report.get("scoring_complete")
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (
                    report.get("cost", {}).get("value"), report.get("latency_seconds", {}).get("p95"),
                    report.get("metrics", {}).get("answer_correctness", {}).get("value"),
                    report.get("conservative_task_success_rate")))):
            excluded.append(report.get("name", report["experiment"]["fingerprint"]))
        else:
            eligible.append(report)
    def values(r):
        return r["conservative_task_success_rate"], r["metrics"]["answer_correctness"]["value"], r["latency_seconds"]["p95"], r["cost"]["value"]
    frontier = []
    for report in eligible:
        q, a, latency, cost = values(report)
        if not any((oq >= q and oa >= a and ol <= latency and oc <= cost)
                   and (oq > q or oa > a or ol < latency or oc < cost)
                   for other in eligible if other is not report for oq, oa, ol, oc in [values(other)]):
            frontier.append(report)
    return {"frontier": frontier, "ineligible": excluded, "production_parameters_changed": False}


def select_validation(reports):
    candidates = pareto_frontier(reports, split="validation")["frontier"]
    if not candidates:
        raise ValueError("no fully scored cost-aware validation candidate")
    return sorted(candidates, key=lambda r: (-r["conservative_task_success_rate"],
        -r["metrics"]["answer_correctness"]["value"], r["latency_seconds"]["p95"], r["cost"]["value"]))[0]


def main(argv=None):
    """Generate explicit budgets or compare real reports; never run/release implicitly."""
    from evals.gold_v2_validation import strict_json
    from evals.report import atomic_write_text
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write candidate budgets; does not execute queries")
    plan.add_argument("--stage", choices=("recall", "rrf", "rerank", "parent"), required=True)
    plan.add_argument("--base", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    compare = commands.add_parser("compare", help="compare completed real reports without publishing settings")
    compare.add_argument("--split", choices=("dev", "validation"), required=True)
    compare.add_argument("--reports", nargs="+", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        budgets = progressive_budgets(args.stage, strict_json(args.base.read_text()))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for index, budget in enumerate(budgets, 1):
            atomic_write_text(args.output_dir / f"budget-{index:02d}.json", budget.model_dump_json(indent=2) + "\n")
        print(f"Created {len(budgets)} budgets; pass each explicitly to eval_rag.sh --retrieval-budget. No queries submitted.")
    else:
        reports = [strict_json(p.read_text()) for p in args.reports]
        result = pareto_frontier(reports, split=args.split)
        selected = select_validation(reports) if args.split == "validation" and result["frontier"] else None
        value = {"split": args.split, "reports": [str(p.resolve()) for p in args.reports],
                 "frontier": [r["experiment"] for r in result["frontier"]], "ineligible": result["ineligible"],
                 "selected": selected["experiment"] if selected else None, "production_parameters_changed": False,
                 "selection_policy": "task_success_then_answer_correctness_then_p95_then_query_cost"}
        if args.output.exists():
            raise ValueError("comparison output already exists")
        atomic_write_text(args.output, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"Saved comparison to {args.output}; no production settings changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
