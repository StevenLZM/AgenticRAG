"""Fail closed on the non-negotiable final acceptance quality gates."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_GATES = (
    "user_leak_count",
    "citation_coverage",
    "unaudited_answer_count",
    "recovery_drill_passed",
    "backup_restore_passed",
    "evaluation_mode",
    "client_provenance",
    "real_query_count",
)


def _matches_verified_report(summary, verified):
    additions = {"completion_verified", "operational_cases", "outcome_counts"}
    return dict(summary) == {key: value for key, value in verified.items() if key not in additions}


def verify_acceptance(summary: Mapping[str, Any], *, verified_summary: Mapping[str, Any] | None = None) -> int:
    """Return zero only when every hard gate has the exact required value."""
    if verified_summary is None or verified_summary.get("completion_verified") is not True:
        return 1
    if not _matches_verified_report(summary, verified_summary):
        return 1
    try:
        return int(not (
            type(summary["user_leak_count"]) is int
            and summary["user_leak_count"] == 0
            and type(summary["citation_coverage"]) in {int, float}
            and not isinstance(summary["citation_coverage"], bool)
            and summary["citation_coverage"] == 1.0
            and type(summary["unaudited_answer_count"]) is int
            and summary["unaudited_answer_count"] == 0
            and type(summary["recovery_drill_passed"]) is bool
            and summary["recovery_drill_passed"]
            and type(summary["backup_restore_passed"]) is bool
            and summary["backup_restore_passed"]
            and summary.get("evaluation_mode") == "api"
            and isinstance(summary.get("client_provenance"), str)
            and summary["client_provenance"] not in {"", "fixture"}
            and type(summary.get("real_query_count")) is int
            and summary["real_query_count"] > 0
        ))
    except (KeyError, TypeError):
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="isolated run whose live SQL/checkpoints verify the report")
    parser.add_argument("--quality-only", action="store_true",
                        help="verify evaluation completion, not release quality thresholds or drills")
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read JSON acceptance report: {error}")
    if not isinstance(raw, Mapping):
        parser.error("acceptance report must be a JSON object")
    from evals.verification import verify_saved_evaluation
    try:
        verified = asyncio.run(verify_saved_evaluation(args.run_dir))
    except Exception as error:
        print(f"VERIFICATION FAILED: {type(error).__name__}")
        return 1
    if not _matches_verified_report(raw, verified):
        print("VERIFICATION FAILED: report differs from verified run")
        return 1
    if args.quality_only:
        print(json.dumps({"evaluation_complete": True, "release_accepted": False,
                          "completed_cases": verified["completed_cases"],
                          "ragas": verified["ragas"], "outcome_counts": verified["outcome_counts"]},
                         ensure_ascii=False))
        return 0
    result = verify_acceptance(raw, verified_summary=verified)
    if result:
        missing = sorted(gate for gate in REQUIRED_GATES if gate not in raw)
        print("ACCEPTANCE FAILED" + (f"; missing: {', '.join(missing)}" if missing else ""))
    else:
        print("ACCEPTANCE PASSED")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
