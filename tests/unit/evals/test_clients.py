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
        self.post_calls = 0

    async def post(self, path: str, **_: object) -> _Response:
        self.post_calls += 1
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
                        "segments": [{"kind": "content", "text": "answer", "evidence_ids": []}],
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


class Collector:
    async def collect(self, **bindings):
        assert bindings["run_id"] == "run-1" and bindings["user_id"] == CASE.user_id
        return {"answer": {"segments": [{"kind": "content", "text": "answer", "evidence_ids": []}],
                           "audited": True, "evidence_parent_ids": ["parent-1"]},
                    "route": "research", "contexts": ["observed context"],
                    "ranked_parent_ids": ["retrieved-not-cited"], "retrieval_rounds": [],
                    "ranking_scope": "single-retrieval", "final_context_parent_ids": ["retrieved-not-cited"]}


@pytest.mark.asyncio
async def test_http_client_polls_public_api_and_preserves_snapshot_provenance(tmp_path) -> None:
    result = await HttpQueryClient(
        "http://api.test", http_client=_Http(), poll_interval_seconds=0.001, collector=Collector(), ledger_dir=tmp_path
    ).query(CASE)

    assert result["client_provenance"] == "real_query_api"
    assert result["evidence_parent_ids"] == ["parent-1"]
    assert result["route"] == "research"
    assert result["contexts"] == ["observed context"]
    assert result["ranked_parent_ids"] == ["retrieved-not-cited"]
    assert result["events"][0]["event_key"] == "api:run-1:1"  # type: ignore[index]


@pytest.mark.asyncio
async def test_poll_interruption_resumes_same_run_without_second_post(tmp_path):
    class InterruptedHttp(_Http):
        async def get(self, path, **kwargs):
            if self.status_calls == 0:
                self.status_calls += 1
                raise TimeoutError("lost poll")
            return await super().get(path, **kwargs)
    http = InterruptedHttp()
    client = HttpQueryClient("http://api.test", http_client=http, collector=Collector(), ledger_dir=tmp_path)
    with pytest.raises(TimeoutError):
        await client.query(CASE)
    result = await client.query(CASE)
    assert result["run_id"] == "run-1" and http.post_calls == 1


@pytest.mark.asyncio
async def test_uncertain_submission_is_never_automatically_repeated(tmp_path):
    class LostPost(_Http):
        async def post(self, *args, **kwargs):
            self.post_calls += 1
            raise TimeoutError("lost response")
    http = LostPost()
    client = HttpQueryClient("http://api.test", http_client=http, collector=Collector(), ledger_dir=tmp_path)
    with pytest.raises(TimeoutError):
        await client.query(CASE)
    with pytest.raises(Exception, match="uncertain"):
        await client.query(CASE)
    assert http.post_calls == 1


@pytest.mark.asyncio
async def test_http_answer_must_equal_collected_answer(tmp_path):
    class WrongCollector(Collector):
        async def collect(self, **kwargs):
            result = await super().collect(**kwargs)
            result["answer"] = {"status": "refuse"}
            return result
    client = HttpQueryClient("http://api.test", http_client=_Http(), collector=WrongCollector(),
                             ledger_dir=tmp_path, poll_interval_seconds=0.001)
    with pytest.raises(Exception, match="answer"):
        await client.query(CASE)


@pytest.mark.asyncio
async def test_http_default_null_fields_do_not_change_answer_identity(tmp_path):
    class DefaultFieldsHttp(_Http):
        async def get(self, path, **kwargs):
            response = await super().get(path, **kwargs)
            if "answer" in response._payload:
                response._payload["answer"].update(status=None, client_provenance=None, citation_coverage=None)
            return response
    result = await HttpQueryClient("http://api.test", http_client=DefaultFieldsHttp(), collector=Collector(),
                                  ledger_dir=tmp_path, poll_interval_seconds=0.001).query(CASE)
    assert result["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_concurrent_query_cannot_duplicate_submission(tmp_path):
    import asyncio
    started, release = asyncio.Event(), asyncio.Event()
    class BlockingHttp(_Http):
        async def post(self, *args, **kwargs):
            started.set()
            await release.wait()
            return await super().post(*args, **kwargs)
    http = BlockingHttp()
    client = HttpQueryClient("http://api.test", http_client=http, collector=Collector(), ledger_dir=tmp_path,
                             poll_interval_seconds=0.001)
    task = asyncio.create_task(client.query(CASE))
    await started.wait()
    try:
        with pytest.raises(Exception, match="in use"):
            await client.query(CASE)
    finally:
        release.set()
        await task
    assert http.post_calls == 1


@pytest.mark.asyncio
async def test_changed_case_does_not_reuse_existing_run(tmp_path):
    http = _Http()
    client = HttpQueryClient("http://api.test", http_client=http, collector=Collector(), ledger_dir=tmp_path,
                             poll_interval_seconds=0.001)
    await client.query(CASE)
    with pytest.raises(Exception, match="binding changed"):
        await client.query(CASE.model_copy(update={"question": "a different question"}))
    assert http.post_calls == 1
