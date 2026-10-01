import importlib.util
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import insert, update, event

from agentic_rag.domain.chat_sessions import SessionNotFound
from agentic_rag.domain.models import RunStatus
from agentic_rag.persistence.repositories import (
    agent_runs,
    documents,
    document_versions,
)
from tests.fixtures.chat_sessions import SNAPSHOT, chat_service
from tests.unit.query.test_answer_sources import answer, payload

pytest_plugins = ("tests.fixtures.chat_sessions",)


async def setup_source(database, users):
    assert importlib.util.find_spec("agentic_rag.runtime.chat_sources"), (
        "source service missing"
    )
    from agentic_rag.runtime.chat_sources import ChatSourceService
    from agentic_rag.query.answer_sources import build_answer_sources

    factory, _ = database
    a, _ = users
    service = chat_service(factory)
    session, _ = await service.create(a, str(uuid4()))
    turn = (
        await service.submit(
            a, session.id, "问", SNAPSHOT, client_request_id=str(uuid4())
        )
    ).run
    doc, version, newer = (str(uuid4()) for _ in range(3))
    packed = payload(1)
    item = packed.items[0].model_copy(
        update={"document_id": doc, "document_version_id": version}
    )
    packed = packed.model_copy(
        update={
            "items": (item,),
            "manifest": {
                "e1": packed.manifest["e1"].model_copy(
                    update={"document_id": doc, "document_version_id": version}
                )
            },
        }
    )
    public = answer(("e1",))
    snapshot = build_answer_sources(
        public, packed, run_id=turn.id, snapshot_id=SNAPSHOT.snapshot_id
    )
    now = datetime.now(UTC)
    async with factory.begin() as db:
        await db.execute(
            insert(documents).values(
                id=doc,
                user_id=a.user_id,
                filename="历史文件.pdf",
                source_type="upload",
                mime_type="application/pdf",
                content_hash=uuid4().hex,
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        for n, ver in enumerate((version, newer)):
            await db.execute(
                insert(document_versions).values(
                    id=ver,
                    document_id=doc,
                    version_no=n + 1,
                    parser_version="p",
                    pipeline_version="p",
                    embedding_version="e",
                    index_generation="i",
                    status="active",
                    created_at=now,
                )
            )
        await db.execute(
            update(documents)
            .where(documents.c.id == doc)
            .values(active_version_id=newer)
        )
        await db.execute(
            update(agent_runs)
            .where(agent_runs.c.id == turn.id)
            .values(
                status="completed",
                active_slot=None,
                answer=public.model_dump(mode="json"),
                answer_sources=snapshot.model_dump(mode="json"),
            )
        )
    return service, ChatSourceService(factory), session, turn, doc


async def test_historical_snapshot_is_authorized_again_on_each_expansion(
    chat_database, chat_users
):
    service, sources, session, turn, doc = await setup_source(chat_database, chat_users)
    factory, _ = chat_database
    a, b = chat_users
    view = await sources.get(a, session.id, turn.id)
    assert view.status == "available" and view.items[0].excerpt == "历史片段"
    assert view.items[0].version_status == "historical"
    assert view.items[0].filename == "历史文件.pdf"
    with pytest.raises(SessionNotFound):
        await sources.get(b, session.id, turn.id)
    other, _ = await service.create(a, str(uuid4()))
    with pytest.raises(SessionNotFound):
        await sources.get(a, other.id, turn.id)
    async with factory.begin() as db:
        await db.execute(
            update(documents)
            .where(documents.c.id == doc)
            .values(deletion_status="pending")
        )
    unavailable = await sources.get(a, session.id, turn.id)
    assert unavailable.status == "unavailable" and not unavailable.items
    await service.delete(a, session.id)
    with pytest.raises(SessionNotFound):
        await sources.get(a, session.id, turn.id)


@pytest.mark.parametrize(
    "damage", ["run", "snapshot", "version", "oversize", "missing"]
)
async def test_corrupt_or_missing_snapshot_never_returns_raw_data(
    chat_database, chat_users, damage
):
    _, sources, session, turn, _ = await setup_source(chat_database, chat_users)
    factory, _ = chat_database
    a, _ = chat_users
    from agentic_rag.persistence.repositories import SqlAlchemyRunRepository

    async with factory.begin() as db:
        run = await SqlAlchemyRunRepository(db).get(turn.id, a)
        stored = run.answer_sources
        if damage == "run":
            stored["run_id"] = "other"
        if damage == "snapshot":
            stored["runtime_config_snapshot_id"] = "other"
        if damage == "version":
            stored["items"][0]["document_version_id"] = "missing"
        if damage == "oversize":
            stored["items"][0]["excerpt"] = "secret" * 100000
        if damage == "missing":
            stored = None
        await db.execute(
            update(agent_runs)
            .where(agent_runs.c.id == turn.id)
            .values(answer_sources=stored)
        )
    result = await sources.get(a, session.id, turn.id)
    assert result.status == "unavailable" and not result.items


async def test_source_write_failure_rolls_back_answer_and_status(
    chat_database, chat_users
):
    factory, engine = chat_database
    service = chat_service(factory)
    a, _ = chat_users
    session, _ = await service.create(a, str(uuid4()))
    run = (
        await service.submit(
            a, session.id, "问", SNAPSHOT, client_request_id=str(uuid4())
        )
    ).run
    runs = service.run_manager.runs
    claim = await runs.claim(run.id, "worker", 60)
    before = (await service.get(a, session.id)).session.last_activity_at

    def fail(conn, cursor, statement, *args):
        if statement.startswith("UPDATE agent_runs") and "answer_sources" in statement:
            raise RuntimeError("source write failed")

    event.listen(engine.sync_engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="source write failed"):
            await runs.finish(
                run.id,
                RunStatus.COMPLETED,
                None,
                None,
                owner="worker",
                claim_generation=claim.claim_generation,
                answer={"status": "cannot_answer"},
                answer_sources={"x": "y"},
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail)
    unchanged = await runs.get(run.id, a)
    assert unchanged.status == RunStatus.RUNNING and unchanged.answer is None
    assert (await service.get(a, session.id)).session.last_activity_at == before
