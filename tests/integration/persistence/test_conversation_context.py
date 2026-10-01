import importlib.util
from datetime import datetime, timedelta

import pytest
from sqlalchemy import event, insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.repositories import agent_runs


async def test_history_is_scoped_bounded_read_only_and_time_stable():
    assert importlib.util.find_spec("agentic_rag.persistence.conversations") is not None
    from agentic_rag.persistence.conversations import SqlAlchemyConversationReader
    from agentic_rag.query.routing_context import RoutingContextUnavailable

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    now = datetime(2026, 9, 30, 16, 1)
    async with engine.begin() as conn:
        await conn.run_sync(agent_runs.create)
        for rid, user, thread, status, offset in [
            ("old", "u", "t", "completed", -2), ("other-user", "v", "t", "completed", -2),
            ("other-thread", "u", "x", "completed", -2), ("late", "u", "t", "completed", 1),
            ("current", "u", "t", "running", 0),
        ]:
            await conn.execute(insert(agent_runs).values(
                id=rid, user_id=user, thread_id=thread, checkpoint_thread_id=f"query:{user}:{thread}",
                status=status, active_slot=1 if status == "running" else None,
                question="京东经历", runtime_config_snapshot_id="s", runtime_config_snapshot={},
                created_at=now + timedelta(minutes=offset),
                finished_at=now + timedelta(minutes=offset) if status == "completed" else None,
                answer={"route": "chat", "segments": [{"kind": "content", "text": "描述经历", "evidence_ids": []}]},
            ))
    statements = []
    event.listen(engine.sync_engine, "before_cursor_execute", lambda c, cur, statement, *args: statements.append(statement))
    reader = SqlAlchemyConversationReader(async_sessionmaker(engine))
    a = await reader.load(UserScope(user_id="u"), run_id="current", thread_id="t")
    b = await reader.load(UserScope(user_id="u"), run_id="current", thread_id="t")
    assert a == b
    assert a.requested_at.startswith("2026-10-01T00:01")
    assert [t.id for t in a.history] == ["query:old:user", "query:old:assistant"]
    assert all(s.lstrip().upper().startswith("SELECT") for s in statements)
    with pytest.raises(RoutingContextUnavailable):
        await reader.load(UserScope(user_id="v"), run_id="current", thread_id="t")
    await engine.dispose()


async def test_second_precision_history_includes_prior_but_excludes_later_run():
    from agentic_rag.persistence.conversations import SqlAlchemyConversationReader
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    anchor = datetime(2026, 10, 1, 0, 0, 0)
    previous = "01a00000-0001-7000-8000-000000000001"
    current = "01a00000-0002-7000-8000-000000000002"
    future = "01a00000-0003-7000-8000-000000000003"
    try:
        async with engine.begin() as conn:
            await conn.run_sync(agent_runs.create)
            for rid in (previous, current, future):
                await conn.execute(insert(agent_runs).values(
                    id=rid, user_id="u", thread_id="t", checkpoint_thread_id="query:u:t",
                    status="running" if rid == current else "completed",
                    active_slot=1 if rid == current else None,
                    question="previous entity" if rid == previous else "unavailable future",
                    runtime_config_snapshot_id="s", runtime_config_snapshot={},
                    created_at=anchor, finished_at=None if rid == current else anchor, answer=None))
        reader = SqlAlchemyConversationReader(async_sessionmaker(engine))
        context = await reader.load(UserScope(user_id="u"), run_id=current, thread_id="t")
        assert [turn.id for turn in context.history] == [f"query:{previous}:user"]
        assert context.history[0].content == "previous entity"
    finally:
        await engine.dispose()


async def test_long_history_keeps_only_six_public_messages_and_8000_characters():
    from agentic_rag.persistence.conversations import SqlAlchemyConversationReader
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    anchor = datetime(2026, 10, 1)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(agent_runs.create)
            for i in range(12):
                status = "running" if i == 11 else ("failed" if i == 10 else "completed")
                await conn.execute(insert(agent_runs).values(
                    id=f"run-{i:02}", user_id="other" if i == 9 else "u",
                    thread_id="other" if i == 8 else "t", checkpoint_thread_id="query:u:t",
                    status=status, active_slot=1 if i == 11 else None,
                    question=f"question-{i}:" + "问" * 2000,
                    answer={"route": "chat", "segments": [{"kind": "content", "text": "答" * 2000}]},
                    runtime_config_snapshot_id="s", runtime_config_snapshot={},
                    created_at=anchor + timedelta(seconds=i),
                    finished_at=anchor + timedelta(seconds=i) if i != 11 else None))
        context = await SqlAlchemyConversationReader(async_sessionmaker(engine)).load(
            UserScope(user_id="u"), run_id="run-11", thread_id="t")
        assert len(context.history) <= 6
        assert sum(len(turn.content) for turn in context.history) == 8000
        assert all(any(f"run-{i:02}" in turn.id for i in (5, 6, 7)) for turn in context.history)
    finally:
        await engine.dispose()
