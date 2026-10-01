"""Persistent chats cross the real SQL/outbox/Redis/Worker/Graph boundaries."""

import asyncio
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import insert, select, update

from agentic_rag.domain.models import UserScope
from agentic_rag.memory.models import MemoryContext
from agentic_rag.persistence.conversations import SqlAlchemyConversationReader
from agentic_rag.persistence.repositories import (
    agent_runs,
    document_versions,
    documents,
)

pytest_plugins = ("tests.fixtures.query_services",)
pytestmark = pytest.mark.integration


async def session(client):
    response = await client.post(
        "/v1/chat-sessions", json={"creation_request_id": str(uuid4())}
    )
    assert response.status_code == 201
    return response.json()["session_id"]


async def submit(client, sid, question):
    response = await client.post(
        f"/v1/chat-sessions/{sid}/turns",
        json={"query": question, "client_request_id": str(uuid4())},
    )
    assert response.status_code == 202
    return response.json()["run_id"]


async def test_three_turn_chat_survives_new_client_and_preserves_scope(
    real_query_fixture, monkeypatch
):
    fixture = real_query_fixture
    memory = fixture._dependencies.memory
    scopes = []
    contexts = []
    conversations = fixture._dependencies.conversations
    read_context = conversations.load

    async def observe_context(*args, **kwargs):
        context = await read_context(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(conversations, "load", observe_context)

    async def load(scope, query, limit=10):
        scopes.append(scope.user_id)
        return MemoryContext()

    monkeypatch.setattr(memory, "load_context", load)
    sid = await session(fixture.client)
    runs = []
    for question in (
        "What notice is required?",
        "Explain that requirement.",
        "Summarize the notice period.",
    ):
        rid = await submit(fixture.client, sid, question)
        runs.append(rid)
        terminal = await fixture.wait_for_terminal(rid)
        assert terminal["status"] == "completed" and terminal["answer"]["audited"]
    await fixture.client.aclose()
    fixture.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fixture.app), base_url="http://reloaded"
    )
    history = (await fixture.client.get(f"/v1/chat-sessions/{sid}/turns")).json()[
        "items"
    ]
    assert [item["run_id"] for item in history] == runs and len(set(runs)) == 3
    assert (
        sum(
            bool(item["question"]) + bool(item["answer"]["segments"])
            for item in history
        )
        == 6
    )
    fourth = await submit(fixture.client, sid, "Continue the same discussion.")
    assert (await fixture.wait_for_terminal(fourth))["thread_id"] == sid
    scope = UserScope(user_id=fixture.settings.default_user_id)
    reader = SqlAlchemyConversationReader(
        fixture.container.repositories.session_factory
    )
    context = await reader.load(scope, run_id=fourth, thread_id=sid)
    assert (
        len(context.history) == 6
        and sum(len(turn.content) for turn in context.history) <= 8000
    )
    assert context.history[0].content == "What notice is required?"
    async with fixture.container.repositories.session_factory() as db:
        stored = (
            (
                await db.execute(
                    select(agent_runs).where(agent_runs.c.id.in_([*runs, fourth]))
                )
            )
            .mappings()
            .all()
        )
    assert {row["thread_id"] for row in stored} == {sid}
    assert {row["checkpoint_thread_id"] for row in stored} == {
        f"query:{scope.user_id}:{sid}"
    }
    assert scopes == [scope.user_id] * 4
    assert [len(context.history) for context in contexts] == [0, 2, 4, 6]
    original = fixture.container.settings
    fixture.container.settings = original.model_copy(
        update={"default_user_id": "unrelated-user"}
    )
    try:
        assert (
            await fixture.client.get(f"/v1/chat-sessions/{sid}/turns")
        ).status_code == 404
        assert (await fixture.client.get(f"/v1/query-runs/{fourth}")).status_code == 404
    finally:
        fixture.container.settings = original


async def test_background_completion_and_source_versions(
    real_query_fixture, monkeypatch
):
    fixture = real_query_fixture
    entered, release = asyncio.Event(), asyncio.Event()
    generator = fixture._dependencies.generator
    generate = generator.generate

    async def held_generation(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), 10)
        return await generate(*args, **kwargs)

    monkeypatch.setattr(generator, "generate", held_generation)
    scopes = []

    async def remember(scope, run_id, messages):
        scopes.append(scope.user_id)

    monkeypatch.setattr(fixture._dependencies.memory, "extract_and_store", remember)
    a = await session(fixture.client)
    run_a = await submit(fixture.client, a, "What notice is required?")
    await asyncio.wait_for(entered.wait(), 10)
    b = await session(fixture.client)
    assert (await fixture.client.get(f"/v1/chat-sessions/{b}/turns")).json()[
        "items"
    ] == []
    release.set()
    terminal_a = await fixture.wait_for_terminal(run_a)
    run_b = await submit(fixture.client, b, "Summarize the notice requirement.")
    assert (await fixture.wait_for_terminal(run_b))["status"] == "completed"
    source_path = f"/v1/chat-sessions/{a}/turns/{run_a}/sources"
    source_a = (await fixture.client.get(source_path)).json()
    assert source_a["status"] == "available" and source_a["items"][0]["page_from"] == 1
    assert (
        await fixture.client.get(f"/v1/chat-sessions/{b}/turns/{run_a}/sources")
    ).status_code == 404
    factory = fixture.container.repositories.session_factory
    async with factory.begin() as db:
        rows = (
            await db.execute(
                select(agent_runs.c.id, agent_runs.c.answer_sources).where(
                    agent_runs.c.id.in_([run_a, run_b])
                )
            )
        ).all()
        assert all(row.answer_sources["run_id"] == row.id for row in rows)
        doc = (
            (
                await db.execute(
                    select(documents).where(
                        documents.c.user_id == fixture.settings.default_user_id
                    )
                )
            )
            .mappings()
            .one()
        )
        version = (
            (
                await db.execute(
                    select(document_versions).where(
                        document_versions.c.id == doc["active_version_id"]
                    )
                )
            )
            .mappings()
            .one()
        )
        next_id = str(uuid4())
        await db.execute(
            insert(document_versions).values(
                **{
                    **dict(version),
                    "id": next_id,
                    "version_no": version["version_no"] + 1,
                }
            )
        )
        await db.execute(
            update(documents)
            .where(documents.c.id == doc["id"])
            .values(active_version_id=next_id)
        )
    updated = (await fixture.client.get(source_path)).json()
    assert updated["items"][0]["version_status"] == "historical"
    assert updated["items"][0]["excerpt"] == source_a["items"][0]["excerpt"]
    assert (await fixture.client.get(f"/v1/query-runs/{run_a}")).json()[
        "answer"
    ] == terminal_a["answer"]
    assert (
        await fixture.client.delete(f"/v1/documents/{doc['id']}")
    ).status_code == 204
    hidden = (await fixture.client.get(source_path)).json()
    assert hidden["status"] == "unavailable" and not hidden["items"]
    assert scopes == [fixture.settings.default_user_id] * 2
