"""Opt-in check for an already-completed isolated real-RAG evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_saved_real_evaluation_has_verified_live_quality_provenance() -> None:
    """Quality E2E must review a real run; it never sends a Query or invokes a judge."""
    configured_run_dir = os.environ.get("AGENTIC_RAG_EVAL_RUN_DIR", "").strip()
    if not configured_run_dir:
        pytest.skip(
            "set AGENTIC_RAG_EVAL_RUN_DIR to a completed isolated real-RAG evaluation"
        )

    from evals.verification import verify_saved_evaluation

    summary = await verify_saved_evaluation(Path(configured_run_dir))

    assert summary["completion_verified"] is True
    assert summary["completed_cases"] == summary["requested_cases"]
    assert summary["real_query_count"] == summary["completed_cases"]
    assert summary["real_query_count"] > 0
    assert summary["ragas_status"] in {"available", "partial"}
    assert summary["operational_cases"]
