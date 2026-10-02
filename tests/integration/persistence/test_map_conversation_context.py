from datetime import datetime, timedelta

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.conversations import SqlAlchemyConversationReader
from agentic_rag.persistence.repositories import agent_runs
from agentic_rag.query.tool_answers import build_tool_answer


async def test_map_references_retain_display_order_and_are_session_scoped():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    now = datetime(2026, 10, 2)
    answer = build_tool_answer([{"call_id": "c1", "tool_id": "mcp.amap_maps.maps_text_search",
        "source_kind": "external", "status": "success", "observed_at": "2026-10-02T00:00:00Z",
        "data": {"structured": {"pois": [
            {"name": "停车场一", "location": "120.1,30.1"},
            {"id": "B023B08WDR", "name": "停车场二", "location": "120.2,30.2"},
        ]}}}], route="fast_rag")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(agent_runs.create)
            for rid, user, session, current in (("old", "u", "s", False), ("foreign", "v", "s", False),
                ("other-session", "u", "x", False), ("current", "u", "s", True)):
                await conn.execute(insert(agent_runs).values(id=rid, user_id=user, thread_id=session,
                    checkpoint_thread_id=f"query:{user}:{session}", status="running" if current else "completed",
                    active_slot=1 if current else None, question="从东站到第二个怎么走", answer=answer if not current else None,
                    runtime_config_snapshot_id="s", runtime_config_snapshot={}, created_at=now if current else now-timedelta(minutes=1),
                    finished_at=None if current else now-timedelta(seconds=30)))
        context = await SqlAlchemyConversationReader(async_sessionmaker(engine)).load(UserScope(user_id="u"),
            run_id="current", thread_id="s")
        assert len(context.map_references) == 2
        assert [item["index"] for item in context.map_references] == [1, 2]
        assert context.map_references[1]["location"] == "120.2,30.2"
        assert context.map_references[1]["poi_id"] == "B023B08WDR"
        assert all(item["run_id"] == "old" for item in context.map_references)
        assert all("structured" not in item for item in context.map_references)
    finally:
        await engine.dispose()
