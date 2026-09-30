"""Opt-in release gate for the real Query/Memory/telemetry path.

The ordinary E2E suite stays hermetic.  This gate is enabled only by a release
operator after the API, Query Worker, Mem0 provider and an isolated evaluation
run have produced their artifacts.  Missing opt-in configuration skips; an
explicitly enabled but malformed or failed gate is a test failure.
"""

from __future__ import annotations

import json
import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from scripts.verify_acceptance import verify_acceptance


pytestmark = pytest.mark.e2e

_DEGRADATION_EVENTS = frozenset(
    {
        "RETRIEVAL_DEGRADED",
        "CIRCUIT_OPEN",
        "COMPONENT_DEGRADED",
        "COMPONENT_REFUSED",
        "OUTBOX_RETRY",
        "WORKER_DLQ",
    }
)
_SAFE_REASONS = frozenset(
    {
        "lane_failure",
        "lane_timeout",
        "retrieval_unavailable",
        "provider_outage",
        "circuit_open",
        "outbox_retry",
        "worker_dlq",
        "model_unavailable",
        "memory_unavailable",
        "unknown",
    }
)
_SAFE_OUTCOMES = frozenset({"degraded", "refused", "dlq"})


def _required_release_flags() -> tuple[str, ...]:
    return (
        "AGENTIC_RAG_RELEASE_WORKER_STARTED",
        "AGENTIC_RAG_RELEASE_QUERY_E2E",
        "AGENTIC_RAG_RELEASE_MEM0_AVAILABLE",
    )


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    """Read either an event JSON array/object or a JSONL telemetry export."""

    text = path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        records: list[object] = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(decoded, Mapping) and isinstance(decoded.get("events"), list):
            records = decoded["events"]
        elif isinstance(decoded, list):
            records = decoded
        else:
            records = [decoded]
    if any(not isinstance(record, Mapping) for record in records):
        raise AssertionError("release telemetry must contain JSON objects")
    return [record for record in records if isinstance(record, Mapping)]


def test_real_query_release_gate() -> None:
    """Require all production-facing gates once release execution is enabled."""

    missing_flags = [name for name in _required_release_flags() if os.getenv(name) != "1"]
    summary_name = os.getenv("AGENTIC_RAG_RELEASE_SUMMARY", "").strip()
    telemetry_name = os.getenv("AGENTIC_RAG_RELEASE_TELEMETRY", "").strip()
    if missing_flags or not summary_name or not telemetry_name:
        pytest.skip(
            "release gate is opt-in; set worker/query/mem0 flags plus "
            "AGENTIC_RAG_RELEASE_SUMMARY and AGENTIC_RAG_RELEASE_TELEMETRY"
        )

    summary_path = Path(summary_name)
    telemetry_path = Path(telemetry_name)
    assert summary_path.is_file(), f"release summary does not exist: {summary_path}"
    assert telemetry_path.is_file(), f"release telemetry does not exist: {telemetry_path}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(summary, Mapping), "release summary must be a JSON object"
    assert summary.get("evaluation_mode") in {"graph", "api"}
    assert type(summary.get("real_query_count")) is int
    assert summary["real_query_count"] > 0
    from evals.verification import verify_saved_evaluation
    run_dir = os.environ.get("AGENTIC_RAG_EVAL_RUN_DIR", "").strip()
    assert run_dir, "release gate requires AGENTIC_RAG_EVAL_RUN_DIR for live verification"
    verified = asyncio.run(verify_saved_evaluation(Path(run_dir)))
    assert verify_acceptance(summary, verified_summary=verified) == 0

    records = _load_records(telemetry_path)
    degradation_records = [
        record for record in records if record.get("event_type") in _DEGRADATION_EVENTS
    ]
    assert degradation_records, (
        "release telemetry must prove an explicit degradation/circuit signal; "
        "healthy-only runs do not satisfy the operational notice gate"
    )
    for record in degradation_records:
        attributes = record.get("attributes")
        assert isinstance(attributes, Mapping)
        assert attributes.get("reason") in _SAFE_REASONS
        assert attributes.get("outcome") in _SAFE_OUTCOMES
        assert type(attributes.get("retryable")) is bool
        assert not any(
            field in record for field in ("prompt", "tool_input", "chain_of_thought", "provider_response")
        )
