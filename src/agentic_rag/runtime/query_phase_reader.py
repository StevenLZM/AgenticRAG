"""Batch-read the last safe phase without reading event artifacts."""

from collections.abc import Sequence
from typing import cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.repositories import agent_events
from agentic_rag.query.phases import PHASES, QueryPhase


class QueryPhaseReader:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def latest(
        self, scope: UserScope, run_ids: Sequence[str]
    ) -> dict[str, QueryPhase]:
        if not run_ids:
            return {}
        latest = (
            select(func.max(agent_events.c.id))
            .where(
                agent_events.c.user_id == scope.user_id,
                agent_events.c.run_id.in_(run_ids),
                agent_events.c.event_type == "QUERY_PHASE_CHANGED",
                agent_events.c.summary.in_(PHASES),
            )
            .group_by(agent_events.c.run_id)
        )
        async with self.session_factory() as db:
            rows = (
                await db.execute(
                    select(agent_events.c.run_id, agent_events.c.summary).where(
                        agent_events.c.user_id == scope.user_id,
                        agent_events.c.id.in_(latest),
                    )
                )
            ).all()
        return {row.run_id: cast(QueryPhase, row.summary) for row in rows}
