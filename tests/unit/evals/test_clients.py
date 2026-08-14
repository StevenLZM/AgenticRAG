"""Real graph/API evaluation client contracts."""

from __future__ import annotations

import pytest

from agentic_rag.runtime.models import RuntimeConfigSnapshot
from evals.clients import GraphQueryClient, HttpQueryClient
from evals.models import EvaluationCase


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main",
    light_model_id="light",
    embedding_model="embed",
    embedding_dimensions=1024,
    reranker_version="reranker",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v2",
    memory_config_version="memory-v1",
)


CASE = EvaluationCase.model_validate(
    {
        "case_id": "graph-case",
        "user_id": "eval-user",
        "question": "Which parent is relevant?",
        "reference_answer": "Parent one is relevant.",
        "reference_parent_ids": ["parent-1"],
        "expected_route": "fast_rag",
        "tags": ["single-hop"],
        "runtime_config_snapshot_id": SNAPSHOT.snapshot_id,
    }
)


class _Graph:
    async def ainvoke(
        self, state: dict[str, object], config: dict[str, object]
    ) -> dict[str, object]:
        assert state["request"] == {"question": CASE.question}
        configurable = config["configurable"]
        assert isinstance(configurable, dict)
        assert str(configurable["thread_id"]).startswith("query:eval-user:")
        return {
            **state,
            "answer": {
                "segments": [
                    {
                        "kind": "content",
                        "text": "Parent one is relevant.",
                        "evidence_ids": ["e1"],
                    }
                ],
                "audited": True,
            },
            "evidence": [
                {"evidence_id": "e1", "parent_id": "parent-1", "content": "Parent one"}
            ],
            "route": {"route": "fast_rag"},
            "audit_results": [
                {"faithfulness": {"passed": True}, "citation": {"passed": True}}
            ],
            "termination_reason": None,
        }


@pytest.mark.asyncio
async def test_graph_client_invokes_real_graph_and_returns_provenance() -> None:
    result = await GraphQueryClient(_Graph(), snapshot=SNAPSHOT).query(CASE)

    assert result["client_provenance"] == "real_query_graph"
    assert result["runtime_config_snapshot_id"] == SNAPSHOT.snapshot_id
    assert result["evidence_parent_ids"] == ["parent-1"]
    assert result["audited"] is True
    assert result["citation_coverage"] == 1.0


class _Response:
    def __init__(self, payload: object, *, text: str = "", status_code: int = 200) -> None:
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def json(self) -> object:
        return self._payload


class _Http:
    def __init__(self) -> None:
        self.status_calls = 0

    async def post(self, path: str, **_: object) -> _Response:
        assert path == "/v1/query-runs"
        return _Response({"run_id": "run-1"}, status_code=202)

    async def get(self, path: str, **_: object) -> _Response:
        if path == "/v1/query-runs/run-1":
            self.status_calls += 1
            status = "queued" if self.status_calls == 1 else "completed"
            return _Response(
                {
                    "run_id": "run-1",
                    "status": status,
                    "thread_id": "thread-1",
                    "runtime_config_snapshot_id": SNAPSHOT.snapshot_id,
                    "answer": {
                        "segments": [],
                        "audited": True,
                        "evidence_parent_ids": ["parent-1"],
                    },
                }
            )
        assert path == "/v1/query-runs/run-1/events"
        return _Response(
            {},
            text=(
                'id: 1\nevent: ANSWER_FINALIZED\n'
                'data: {"event_type":"ANSWER_FINALIZED","summary":"completed"}\n\n'
            ),
        )


@pytest.mark.asyncio
async def test_http_client_polls_public_api_and_preserves_snapshot_provenance() -> None:
    result = await HttpQueryClient(
        "http://api.test", http_client=_Http(), poll_interval_seconds=0.001
    ).query(CASE)

    assert result["client_provenance"] == "real_query_api"
    assert result["evidence_parent_ids"] == ["parent-1"]
    assert result["events"][0]["event_key"] == "api:run-1:1"  # type: ignore[index]
