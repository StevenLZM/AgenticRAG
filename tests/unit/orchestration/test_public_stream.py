from datetime import UTC, datetime
from types import SimpleNamespace
from dataclasses import replace

import httpx

from agentic_rag.api.app import create_app
from agentic_rag.domain.models import RunStatus
from agentic_rag.persistence.repositories import QueryRun
from agentic_rag.orchestration.streaming import append_delta, complete_text, complete_quality, QualityResult
from tests.unit.orchestration.test_streaming import frozen
from tests.unit.orchestration.test_evidence import SNAPSHOT


async def test_sse_replays_scoped_v2_deltas_and_quality_without_audited(store):
    claim = await frozen(store)
    await append_delta(store, claim, "hello")
    await append_delta(store, claim, " world")
    await complete_text(store, claim)
    quality = (await store.list_ready("quality"))[0]
    qclaim = await store.claim(quality["id"], "quality")
    answer = (await store.read_run("run-1"))["data"]["answer"]
    await complete_quality(store, qclaim, QualityResult(input_hash=answer["input_hash"], answer_hash=answer["answer_hash"], faithfulness="failed"))
    snapshot = SNAPSHOT.model_copy(update={"orchestration_version": "supervisor-v1", "answer_protocol_version": "answer-v2"})
    run = QueryRun(id="run-1", user_id="user-1", thread_id="thread", checkpoint_thread_id="unused", status=RunStatus.RUNNING, active_slot=1,
                   runtime_config_snapshot_id=snapshot.snapshot_id, runtime_config_snapshot=snapshot.model_dump(mode="json"), created_at=datetime.now(UTC))
    class Manager:
        async def get(self, run_id, scope):
            return run if scope.user_id == run.user_id else None
    container = SimpleNamespace(settings=SimpleNamespace(default_user_id="user-1"), run_manager=Manager(), coordination_store=store)
    app = create_app(container.settings, container=container)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/query-runs/run-1/events", headers={"Last-Event-ID": "1:1"})
        assert response.status_code == 200
        assert '"text": "hello"' not in response.text
        assert '"text": " world"' in response.text
        assert "answer.completed" in response.text
        assert '"audited"' not in response.text
        response = await client.get("/v1/query-runs/run-1")
        assert response.json()["answer"]["text"] == "hello world"
        assert response.json()["answer"]["faithfulness"] == "failed"
        container.settings.default_user_id = "foreign"
        assert (await client.get("/v1/query-runs/run-1/events")).status_code == 404


async def test_revocation_between_replayed_batch_items_stops_the_next_send(store, monkeypatch):
    from agentic_rag.api import query_runs, supervisor_stream
    claim = await frozen(store)
    await append_delta(store, claim, "before revocation")
    await append_delta(store, claim, "must not disclose")
    run = QueryRun(id="run-1", user_id="user-1", thread_id="thread", checkpoint_thread_id="unused", status=RunStatus.RUNNING,
                   active_slot=1, runtime_config_snapshot_id=SNAPSHOT.snapshot_id, runtime_config_snapshot={}, created_at=datetime.now(UTC))
    calls = 0
    async def owned(*args):
        nonlocal calls
        calls += 1
        return replace(run, status=RunStatus.FAILED, answer=None, error_code="SOURCE_ACCESS_REVOKED") if calls >= 4 else run
    async def disconnected():
        return False
    monkeypatch.setattr(query_runs, "_owned_run", owned)
    request = SimpleNamespace(is_disconnected=disconnected, app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(coordination_store=store))))
    response = await supervisor_stream.events(request, "run-1", None)
    output = "".join([chunk async for chunk in response.body_iterator])
    assert "before revocation" in output
    assert "must not disclose" not in output
