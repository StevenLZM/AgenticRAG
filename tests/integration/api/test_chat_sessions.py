"""HTTP contracts backed by real SQL transactions, including lost responses."""

import re
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import update

from agentic_rag.api.app import create_app
from agentic_rag.bootstrap import _TransactionalEventRepository
from agentic_rag.persistence.repositories import agent_runs
from agentic_rag.runtime.chat_sources import ChatSourceService
from agentic_rag.runtime.query_phase_reader import QueryPhaseReader
from tests.fixtures.chat_sessions import SNAPSHOT, chat_service

pytest_plugins = ("tests.fixtures.chat_sessions",)


@pytest.fixture
async def chat_http(chat_database, chat_users):
    factory, _ = chat_database
    a, b = chat_users
    service = chat_service(factory)
    container = SimpleNamespace(
        settings=SimpleNamespace(default_user_id=a.user_id),
        runtime_snapshot=SNAPSHOT,
        run_manager=service.run_manager,
        chat_session_service=service,
        chat_source_service=ChatSourceService(factory),
        query_phase_reader=QueryPhaseReader(factory),
        event_repository=_TransactionalEventRepository(factory),
    )
    app = create_app(container.settings, container=container)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, container, factory, a, b


async def create(client):
    key = str(uuid4())
    response = await client.post("/v1/chat-sessions", json={"creation_request_id": key})
    assert response.status_code == 201
    return response.json()["session_id"], key


async def test_session_turn_replay_and_reload_contract(chat_http):
    client, container, factory, a, b = chat_http
    sid, creation_key = await create(client)
    replay = await client.post(
        "/v1/chat-sessions", json={"creation_request_id": creation_key}
    )
    assert replay.status_code == 200 and replay.json()["session_id"] == sid
    path = f"/v1/chat-sessions/{sid}"
    key = str(uuid4())
    first = await client.post(
        path + "/turns", json={"query": "  首问  ", "client_request_id": key}
    )
    assert first.status_code == 202
    run = first.json()
    assert (
        run["client_request_id"] == key
        and run["answer"] is None
        and run["source_status"] == "none"
    )
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", run["created_at"])
    # Treat the first response as lost: consult durable submission identity.
    checked = await client.get(path + "/submissions/" + key)
    assert checked.json()["run_id"] == run["run_id"]
    again = await client.post(
        path + "/turns", json={"query": "首问", "client_request_id": key}
    )
    assert again.status_code == 200 and again.json()["run_id"] == run["run_id"]
    history = await client.get(path + "/turns")
    assert [item["question"] for item in history.json()["items"]] == ["首问"]
    assert history.headers["cache-control"] == "no-store"
    assert "user_id" not in replay.json() and "runtime_config_snapshot" not in str(
        history.json()
    )
    metadata = (await client.get(path)).json()
    assert metadata["active_run_id"] == run["run_id"]
    assert (await client.get("/v1/chat-sessions")).json()["items"][0]["title"] == "首问"
    assert (await client.patch(path, json={"title": " 自定义 "})).json()[
        "title"
    ] == "自定义"
    busy = await client.post(
        path + "/turns", json={"query": "秘密草稿", "client_request_id": str(uuid4())}
    )
    assert busy.status_code == 409 and busy.json()["error_code"] == "SESSION_BUSY"
    assert (
        busy.headers["location"] == f"/v1/query-runs/{run['run_id']}"
        and "秘密" not in busy.text
    )
    conflict = await client.post(
        path + "/turns", json={"query": "不同", "client_request_id": key}
    )
    assert (
        conflict.status_code == 409
        and conflict.json()["error_code"] == "IDEMPOTENCY_CONFLICT"
    )
    assert (await client.delete(path)).status_code == 409
    await client.post(f"/v1/query-runs/{run['run_id']}/cancel")
    sources = await client.get(path + f"/turns/{run['run_id']}/sources")
    assert sources.status_code == 200 and sources.headers["cache-control"] == "no-store"
    assert sources.json()["status"] == "none"
    assert (
        (await client.delete(path)).status_code
        == (await client.delete(path)).status_code
        == 204
    )
    assert (
        await client.post(
            "/v1/chat-sessions", json={"creation_request_id": creation_key}
        )
    ).status_code == 410
    for url in (
        path,
        path + "/turns",
        path + "/submissions/" + key,
        f"/v1/query-runs/{run['run_id']}",
        f"/v1/query-runs/{run['run_id']}/events",
    ):
        assert (await client.get(url)).status_code == 404
    assert (
        await client.post("/v1/query-runs", json={"query": "revive", "thread_id": sid})
    ).status_code == 404


async def test_scope_and_validation_do_not_leak_session_or_question(chat_http):
    client, container, _, a, b = chat_http
    sid, _ = await create(client)
    path = f"/v1/chat-sessions/{sid}"
    for body in (
        {"creation_request_id": "bad"},
        {"creation_request_id": str(uuid4()), "user_id": b.user_id},
    ):
        assert (await client.post("/v1/chat-sessions", json=body)).status_code == 422
    for body in (
        {"query": "x", "client_request_id": "bad"},
        {"query": "x", "client_request_id": str(uuid4()), "evaluation": {}},
        {"query": "x" * 32001, "client_request_id": str(uuid4())},
    ):
        assert (await client.post(path + "/turns", json=body)).status_code == 422
    for suffix in ("?limit=0", "?limit=101", "?cursor=bad"):
        assert (await client.get(path + "/turns" + suffix)).status_code == 422
    assert (await client.get(path + "/submissions/" + str(uuid4()))).status_code == 404
    assert (await client.get("/v1/chat-sessions/bad")).status_code == 422
    container.settings.default_user_id = b.user_id
    assert (await client.get("/v1/chat-sessions")).json()["items"] == []
    assert (await client.get(path)).status_code == 404
    assert (
        await client.post(
            path + "/turns", json={"query": "x", "client_request_id": str(uuid4())}
        )
    ).status_code == 404
    assert (
        await client.post("/v1/query-runs", json={"query": "x", "thread_id": sid})
    ).status_code == 404


async def test_history_only_publishes_completed_public_answers(chat_http):
    client, _, factory, _, _ = chat_http
    sid, _ = await create(client)
    path = f"/v1/chat-sessions/{sid}/turns"
    run = (
        await client.post(path, json={"query": "q", "client_request_id": str(uuid4())})
    ).json()
    answer = {
        "route": "chat",
        "segments": [{"kind": "content", "text": "已完成", "private": "secret"}],
        "prompt": "secret",
    }
    async with factory.begin() as db:
        await db.execute(
            update(agent_runs)
            .where(agent_runs.c.id == run["run_id"])
            .values(status="running", answer=answer)
        )
    assert (await client.get(path)).json()["items"][0]["answer"] is None
    async with factory.begin() as db:
        await db.execute(
            update(agent_runs)
            .where(agent_runs.c.id == run["run_id"])
            .values(status="completed", active_slot=None)
        )
    item = (await client.get(path)).json()["items"][0]
    assert item["answer"]["segments"][0]["text"] == "已完成" and "secret" not in str(
        item
    )
    assert item["phase"] is None and item["terminal_code"] == "completed"


async def test_unicode_question_limit_round_trips_through_mysql(chat_http):
    client, _, _, _, _ = chat_http
    sid, _ = await create(client)
    text = "😀" * 32000
    result = await client.post(
        f"/v1/chat-sessions/{sid}/turns",
        json={"query": text, "client_request_id": str(uuid4())},
    )
    assert result.status_code == 202 and result.json()["question"] == text


async def test_deleted_session_stops_an_already_open_event_stream(chat_http):
    from agentic_rag.persistence.repositories import AgentEvent

    client, container, _, a, _ = chat_http
    sid, _ = await create(client)
    turn = (
        await client.post(
            f"/v1/chat-sessions/{sid}/turns",
            json={"query": "q", "client_request_id": str(uuid4())},
        )
    ).json()
    rid = turn["run_id"]
    await client.post(f"/v1/query-runs/{rid}/cancel")

    class DeleteDuringRead:
        async def list_after(self, *args):
            await container.chat_session_service.delete(a, sid)
            return [
                AgentEvent(
                    id=1,
                    event_key="e",
                    trace_id=rid,
                    run_id=rid,
                    user_id=a.user_id,
                    event_type="QUERY_PHASE_CHANGED",
                    summary="auditing",
                    runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
                )
            ]

    container.event_repository = DeleteDuringRead()
    response = await client.get(f"/v1/query-runs/{rid}/events")
    assert response.status_code == 200 and not response.text
