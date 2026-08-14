"""Fail closed on the non-negotiable final acceptance quality gates."""

from __future__ import annotations

import argparse
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


def verify_acceptance(summary: Mapping[str, Any]) -> int:
    """Return zero only when every hard gate has the exact required value."""
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
            and summary.get("evaluation_mode") in {"graph", "api"}
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
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read JSON acceptance report: {error}")
    if not isinstance(raw, Mapping):
        parser.error("acceptance report must be a JSON object")
    result = verify_acceptance(raw)
    if result:
        missing = sorted(gate for gate in REQUIRED_GATES if gate not in raw)
        print("ACCEPTANCE FAILED" + (f"; missing: {', '.join(missing)}" if missing else ""))
    else:
        print("ACCEPTANCE PASSED")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
