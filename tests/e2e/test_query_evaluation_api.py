"""Opt-in evaluation through the deployed Query API boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evals.clients import HttpQueryClient
from evals.models import EvaluationCase
from evals.run import EvalRunner


def _real_api_case() -> EvaluationCase:
    snapshot_id = os.environ.get("AGENTIC_RAG_EVAL_SNAPSHOT_ID", "").strip()
    if not snapshot_id:
        pytest.skip("set AGENTIC_RAG_EVAL_SNAPSHOT_ID for real API evaluation")
    return EvaluationCase.model_validate(
        {
            "case_id": "real-api-case",
            "user_id": os.environ.get("AGENTIC_RAG_EVAL_USER_ID", "default_user"),
            "question": os.environ.get(
                "AGENTIC_RAG_EVAL_QUERY", "What does the seeded document require?"
            ),
            "reference_answer": os.environ.get(
                "AGENTIC_RAG_EVAL_REFERENCE", "The seeded document answer."
            ),
            "reference_parent_ids": [
                os.environ.get("AGENTIC_RAG_EVAL_PARENT_ID", "parent-1")
            ],
            "expected_route": os.environ.get("AGENTIC_RAG_EVAL_ROUTE", "fast_rag"),
            "tags": ["real-api"],
            "runtime_config_snapshot_id": snapshot_id,
        }
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_evaluation_uses_real_query_run_and_persists_provenance(
    tmp_path: Path,
) -> None:
    base_url = os.environ.get("AGENTIC_RAG_EVAL_API_BASE_URL", "").strip()
    if os.environ.get("AGENTIC_RAG_RUN_REAL_QUERY_E2E") != "1" or not base_url:
        pytest.skip(
            "set AGENTIC_RAG_RUN_REAL_QUERY_E2E=1 and "
            "AGENTIC_RAG_EVAL_API_BASE_URL for real API evaluation"
        )
    case = _real_api_case()
    client = HttpQueryClient(base_url)
    try:
        summary = await EvalRunner(
            client,
            output_dir=tmp_path,
            evaluation_mode="api",
            client_provenance=HttpQueryClient.provenance,
        ).run([case])
    finally:
        await client.aclose()

    assert summary["evaluation_mode"] == "api"
    assert summary["client_provenance"] == "real_query_api"
    assert summary["real_query_count"] == 1

