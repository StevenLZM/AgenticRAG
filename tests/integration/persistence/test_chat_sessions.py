"""Session lifecycle, tenant isolation and stable keyset pagination."""
from datetime import UTC, datetime
import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import update

from tests.fixtures.chat_sessions import SNAPSHOT, chat_service

pytest_plugins = ("tests.fixtures.chat_sessions",)


async def test_create_replay_rename_delete_and_scope(chat_database, chat_users):
    factory, _ = chat_database
    service = chat_service(factory)
    from agentic_rag.domain.chat_sessions import SessionGone, SessionNotFound
    a, b = chat_users
    key = str(uuid4())
    first, created = await service.create(a, key)
    replay, replay_created = await service.create(a, key.upper())
    assert created and not replay_created and first.id == replay.id
    assert first.title == "新对话"
    assert (await service.list(b)).items == ()
    other, _ = await service.create(b, key)
    assert other.id != first.id
    before = (await service.get(a, first.id)).session.last_activity_at
    renamed = await service.rename(a, first.id, "  手动标题  ")
    assert renamed.title == "手动标题" and renamed.title_source == "manual"
    assert renamed.last_activity_at == before
    with pytest.raises(SessionNotFound):
        await service.get(b, first.id)
    with pytest.raises(SessionNotFound):
        await service.delete(b, first.id)
    await service.delete(a, first.id)
    await service.delete(a, first.id)
    assert (await service.list(a)).items == ()
    with pytest.raises(SessionNotFound):
        await service.get(a, first.id)
    with pytest.raises(SessionGone):
        await service.create(a, key)


async def test_delete_requires_no_active_run(chat_database, chat_users):
    factory, _ = chat_database
    service = chat_service(factory)
    from agentic_rag.domain.chat_sessions import SessionBusy
    a, _ = chat_users
    session, _ = await service.create(a, str(uuid4()))
    run = await service.run_manager.create(a, session.id, "question", SNAPSHOT)
    with pytest.raises(SessionBusy) as exc:
        await service.delete(a, session.id)
    assert exc.value.active_run_id == run.id
    summary = await service.get(a, session.id)
    assert summary.active_run_id == run.id


async def test_keyset_pagination_handles_equal_times_and_invalid_cursor(chat_database, chat_users):
    factory, _ = chat_database
    service = chat_service(factory)
    from agentic_rag.persistence.repositories import chat_sessions
    a, _ = chat_users
    created = [(await service.create(a, str(uuid4())))[0] for _ in range(101)]
    async with factory.begin() as db:
        await db.execute(update(chat_sessions).where(chat_sessions.c.user_id == a.user_id)
                         .values(last_activity_at=datetime(2026, 10, 1, tzinfo=UTC)))
    first = await service.list(a)
    assert len(first.items) == 30
    seen, cursor = [s.session.id for s in first.items], first.next_cursor
    while cursor:
        page = await service.list(a, cursor=cursor)
        seen.extend(s.session.id for s in page.items)
        cursor = page.next_cursor
    assert seen == sorted((s.id for s in created), reverse=True)
    with pytest.raises(ValueError):
        await service.list(a, cursor="not-a-cursor")
    with pytest.raises(ValueError):
        await service.list(a, limit=101)


async def test_concurrent_create_replays_same_request(chat_database, chat_users):
    factory, engine = chat_database
    service = chat_service(factory)
    if engine.dialect.name != "mysql":
        pytest.skip("real MySQL validates concurrent unique-key transactions")
    a, _ = chat_users
    key = str(uuid4())
    results = await asyncio.gather(*(service.create(a, key) for _ in range(4)))
    assert len({s.id for s, _ in results}) == 1
    assert sum(created for _, created in results) == 1


@pytest.mark.parametrize("title", ["", "  ", "长" * 101], ids=["empty", "whitespace", "too-long"])
async def test_invalid_title_never_changes_session(chat_database, chat_users, title):
    factory, _ = chat_database
    service = chat_service(factory)
    a, _ = chat_users
    session, _ = await service.create(a, str(uuid4()))
    with pytest.raises(ValueError):
        await service.rename(a, session.id, title)
    assert (await service.get(a, session.id)).session.title == "新对话"
