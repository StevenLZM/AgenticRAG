"""Composition-level evaluation through the real QueryGraph client boundary."""

from __future__ import annotations

from typing import Any

import pytest

from agentic_rag.runtime.models import RuntimeConfigSnapshot
from evals.clients import GraphQueryClient
from evals.models import EvaluationCase
from evals.run import EvalRunner
from scripts.verify_acceptance import verify_acceptance


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="eval-test",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main",
    light_model_id="light",
    embedding_model="embedding",
    embedding_dimensions=1024,
    reranker_version="reranker-v1",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)


def _case() -> EvaluationCase:
    return EvaluationCase.model_validate(
        {
            "case_id": "real-graph-case",
            "user_id": "eval-user",
            "question": "Which parent is relevant?",
            "reference_answer": "Parent one is relevant.",
            "reference_parent_ids": ["parent-1"],
            "expected_route": "fast_rag",
            "tags": ["single-hop"],
            "runtime_config_snapshot_id": SNAPSHOT.snapshot_id,
        }
    )


class _ProductionGraphBoundary:
    """A deterministic graph port; the client and runner remain production code."""

    async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        del config
        return {
            **state,
            "answer": {
                "segments": [
                    {
                        "kind": "content",
                        "text": "Parent one is relevant.",
                        "evidence_ids": ["evidence-1"],
                    }
                ],
                "audited": True,
            },
            "evidence": [
                {
                    "evidence_id": "evidence-1",
                    "parent_id": "parent-1",
                    "content": "Parent one",
                }
            ],
            "route": {"route": "fast_rag"},
            "audit_results": [
                {"citation": {"passed": True}, "faithfulness": {"passed": True}}
            ],
            "events": [],
            "termination_reason": "completed",
        }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_evaluation_runs_through_real_client_and_passes_provenance_gate(
    tmp_path: Any,
) -> None:
    runner = EvalRunner(
        GraphQueryClient(_ProductionGraphBoundary(), snapshot=SNAPSHOT),
        output_dir=tmp_path,
        evaluation_mode="graph",
        client_provenance=GraphQueryClient.provenance,
    )

    summary = await runner.run(
        [_case()],
        recovery_drill_passed=True,
        backup_restore_passed=True,
    )

    assert summary["evaluation_mode"] == "graph"
    assert summary["client_provenance"] == "real_query_graph"
    assert summary["real_query_count"] == 1
    assert verify_acceptance(summary) == 0

