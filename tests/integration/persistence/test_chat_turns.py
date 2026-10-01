"""Durable turn idempotency, transaction races and lifecycle fences."""
import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select, update

from agentic_rag.domain.chat_sessions import IdempotencyConflict, SessionBusy, SessionNotFound
from agentic_rag.domain.models import RunStatus
from agentic_rag.persistence.repositories import agent_runs, chat_sessions, task_outbox, LeaseLost
from agentic_rag.runtime.run_manager import TransactionalRunRepository
from tests.fixtures.chat_sessions import SNAPSHOT, chat_service

pytest_plugins = ("tests.fixtures.chat_sessions",)


async def setup_turns(database, users):
    factory, _ = database
    service = chat_service(factory)
    assert hasattr(service, "submit"), "chat submissions are not implemented"
    a, _ = users
    session, _ = await service.create(a, str(uuid4()))
    return service, session, a


async def test_submission_replay_conflict_cancel_and_next_turn(chat_database, chat_users):
    service, session, a = await setup_turns(chat_database, chat_users)
    key = str(uuid4())
    first = await service.submit(a, session.id, "  第一行\n第二行  ", SNAPSHOT, client_request_id=key)
    replay = await service.submit(a, session.id, "第一行\n第二行", SNAPSHOT, client_request_id=key)
    assert first.created and not replay.created and first.run.id == replay.run.id
    assert first.run.client_request_id == key and first.run.created_at is not None
    with pytest.raises(IdempotencyConflict):
        await service.submit(a, session.id, "第一行 第二行", SNAPSHOT, client_request_id=key)
    with pytest.raises(SessionBusy):
        await service.submit(a, session.id, "另一个问题", SNAPSHOT, client_request_id=str(uuid4()))
    await service.run_manager.request_cancel(a, first.run.id)
    replay = await service.submit(a, session.id, "第一行\n第二行", SNAPSHOT, client_request_id=key)
    assert not replay.created and replay.run.status == RunStatus.CANCELLED
    second = await service.submit(a, session.id, "继续", SNAPSHOT, client_request_id=str(uuid4()))
    assert first.run.thread_id == second.run.thread_id == session.id
    assert (await service.find_submission(a, session.id, key)).id == first.run.id
    assert await service.find_submission(a, session.id, str(uuid4())) is None


async def test_duplicate_submission_is_one_run_and_outbox(chat_database, chat_users):
    factory, engine = chat_database
    if engine.dialect.name != "mysql":
        pytest.skip("MySQL row-lock concurrency contract")
    service, session, a = await setup_turns(chat_database, chat_users)
    key = str(uuid4())
    results = await asyncio.gather(*(service.submit(a, session.id, "首问", SNAPSHOT, client_request_id=key) for _ in range(4)))
    assert len({r.run.id for r in results}) == 1
    assert sum(r.created for r in results) == 1
    async with factory() as db:
        count = (await db.execute(select(func.count()).select_from(agent_runs).where(
            agent_runs.c.user_id == a.user_id, agent_runs.c.thread_id == session.id))).scalar_one()
        outbox = (await db.execute(select(func.count()).select_from(task_outbox).where(
            task_outbox.c.aggregate_type == "query_run", task_outbox.c.aggregate_id == results[0].run.id))).scalar_one()
    assert count == outbox == 1


async def test_title_is_unicode_bounded_and_manual_title_wins(chat_database, chat_users):
    service, session, a = await setup_turns(chat_database, chat_users)
    await service.submit(a, session.id, "😀" * 31, SNAPSHOT, client_request_id=str(uuid4()))
    assert (await service.get(a, session.id)).session.title == "😀" * 30 + "…"
    other, _ = await service.create(a, str(uuid4()))
    await service.rename(a, other.id, "自定义")
    await service.submit(a, other.id, "不能覆盖", SNAPSHOT, client_request_id=str(uuid4()))
    assert (await service.get(a, other.id)).session.title == "自定义"


async def test_outbox_failure_rolls_back_turn_title_and_activity(chat_database, chat_users):
    factory, engine = chat_database
    service, session, a = await setup_turns(chat_database, chat_users)
    def fail_outbox(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().startswith("INSERT INTO task_outbox"):
            raise RuntimeError("injected outbox write failure")
    event.listen(engine.sync_engine, "before_cursor_execute", fail_outbox)
    try:
        with pytest.raises(RuntimeError, match="injected outbox"):
            await service.submit(a, session.id, "首问", SNAPSHOT, client_request_id=str(uuid4()))
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail_outbox)
    assert (await service.get(a, session.id)).session == session
    assert (await service.turns(a, session.id)).items == ()


async def test_cancel_finish_and_stale_lease_are_atomic(chat_database, chat_users):
    factory, _ = chat_database
    service, session, a = await setup_turns(chat_database, chat_users)
    first = await service.submit(a, session.id, "首问", SNAPSHOT, client_request_id=str(uuid4()))
    async with factory.begin() as db:
        await db.execute(update(chat_sessions).where(chat_sessions.c.id == session.id).values(last_activity_at=datetime(2000, 1, 1, tzinfo=UTC)))
    await service.run_manager.request_cancel(a, first.run.id)
    cancelled_at = (await service.get(a, session.id)).session.last_activity_at
    assert cancelled_at.year > 2000
    await service.run_manager.request_cancel(a, first.run.id)
    assert (await service.get(a, session.id)).session.last_activity_at == cancelled_at
    second = await service.submit(a, session.id, "第二问", SNAPSHOT, client_request_id=str(uuid4()))
    runs = TransactionalRunRepository(factory)
    claim = await runs.claim(second.run.id, "owner", 60)
    before = (await service.get(a, session.id)).session.last_activity_at
    with pytest.raises(LeaseLost):
        await runs.finish(claim.id, RunStatus.COMPLETED, None, None, owner="owner",
                          claim_generation=claim.claim_generation + 1, answer={"status": "test"}, answer_sources={"marker": "source"})
    unchanged = await runs.get(claim.id, a)
    assert unchanged.status == RunStatus.RUNNING and unchanged.answer is None and unchanged.answer_sources is None
    assert (await service.get(a, session.id)).session.last_activity_at == before
    await runs.finish(claim.id, RunStatus.COMPLETED, None, None, owner="owner",
                      claim_generation=claim.claim_generation, answer={"status": "test"}, answer_sources={"marker": "source"})
    completed = await runs.get(claim.id, a)
    assert completed.answer == {"status": "test"} and completed.answer_sources == {"marker": "source"}
    assert completed.finished_at is not None


async def test_deleted_and_foreign_session_cannot_use_legacy_run_ports(chat_database, chat_users):
    service, session, a = await setup_turns(chat_database, chat_users)
    _, b = chat_users
    with pytest.raises(SessionNotFound):
        await service.run_manager.create(b, session.id, "foreign", SNAPSHOT)
    with pytest.raises(SessionNotFound):
        await service.turns(b, session.id)
    first = await service.run_manager.create(a, session.id, "legacy", SNAPSHOT)
    assert (await service.get(a, session.id)).session.title == "legacy"
    await service.run_manager.request_cancel(a, first.id)
    await service.delete(a, session.id)
    assert await service.run_manager.get(first.id, a) is None
    assert await service.run_manager.get_for_delivery(first.id) is not None
    with pytest.raises(SessionNotFound):
        await service.run_manager.create(a, session.id, "revive", SNAPSHOT)
    with pytest.raises(SessionNotFound):
        await service.run_manager.request_cancel(a, first.id)


async def test_turn_pages_include_all_statuses_in_stable_order(chat_database, chat_users):
    factory, _ = chat_database
    service, session, a = await setup_turns(chat_database, chat_users)
    ids = []
    for i in range(5):
        turn = await service.submit(a, session.id, f"问{i}", SNAPSHOT, client_request_id=str(uuid4()))
        ids.append(turn.run.id)
        await service.run_manager.request_cancel(a, turn.run.id)
    async with factory.begin() as db:
        await db.execute(update(agent_runs).where(agent_runs.c.thread_id == session.id).values(created_at=datetime(2026, 10, 1, tzinfo=UTC)))
    newest = await service.turns(a, session.id, limit=2)
    older = await service.turns(a, session.id, cursor=newest.next_cursor, limit=2)
    oldest = await service.turns(a, session.id, cursor=older.next_cursor, limit=2)
    assert [r.id for r in (*oldest.items, *older.items, *newest.items)] == sorted(ids)
    assert oldest.next_cursor is None


@pytest.mark.parametrize("winner", ["delete", "submit"])
async def test_delete_and_submit_race_respects_session_lock(chat_database, chat_users, monkeypatch, winner):
    factory, engine = chat_database
    if engine.dialect.name != "mysql":
        pytest.skip("MySQL row-lock concurrency contract")
    service, session, a = await setup_turns(chat_database, chat_users)
    async def submit():
        return await service.submit(a, session.id, "question", SNAPSHOT, client_request_id=str(uuid4()))
    if winner == "delete":
        async with factory.begin() as db:
            await db.execute(select(chat_sessions).where(chat_sessions.c.id == session.id).with_for_update())
            waiting = asyncio.create_task(submit())
            await db.execute(update(chat_sessions).where(chat_sessions.c.id == session.id).values(deleted_at=datetime.now(UTC)))
        with pytest.raises(SessionNotFound):
            await asyncio.wait_for(waiting, 5)
    else:
        inserted, release = asyncio.Event(), asyncio.Event()
        real_create = service.run_manager.runs.create_queued
        async def held_create(*args, **kwargs):
            run = await real_create(*args, **kwargs)
            inserted.set()
            await release.wait()
            return run
        monkeypatch.setattr(service.run_manager.runs, "create_queued", held_create)
        submission = asyncio.create_task(submit())
        await asyncio.wait_for(inserted.wait(), 5)
        deletion = asyncio.create_task(service.delete(a, session.id))
        release.set()
        result = await asyncio.wait_for(submission, 5)
        with pytest.raises(SessionBusy) as exc:
            await asyncio.wait_for(deletion, 5)
        assert exc.value.active_run_id == result.run.id


async def test_different_keys_compete_for_one_active_turn(chat_database, chat_users):
    _, engine = chat_database
    if engine.dialect.name != "mysql":
        pytest.skip("MySQL row-lock concurrency contract")
    service, session, a = await setup_turns(chat_database, chat_users)
    results = await asyncio.gather(*(service.submit(a, session.id, "问题", SNAPSHOT,
        client_request_id=str(uuid4())) for _ in range(4)), return_exceptions=True)
    assert sum(isinstance(result, SessionBusy) for result in results) == 3
    assert len((await service.turns(a, session.id)).items) == 1
