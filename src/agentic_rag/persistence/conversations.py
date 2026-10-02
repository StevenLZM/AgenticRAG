"""Read immutable completed public exchanges, without relying on tool messages."""

from datetime import timezone
import json
from zoneinfo import ZoneInfo

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.repositories import agent_runs
from agentic_rag.query.public_answer import project_public_answer
from agentic_rag.query.routing_context import (
    RoutingContext, RoutingContextUnavailable, RoutingTurn, bound_history,
)


class SqlAlchemyConversationReader:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    async def load(self, scope: UserScope, *, run_id: str, thread_id: str) -> RoutingContext:
        try:
            async with self._factory() as session:
                anchor = (await session.execute(select(agent_runs.c.created_at).where(
                    agent_runs.c.id == run_id, agent_runs.c.user_id == scope.user_id,
                    agent_runs.c.thread_id == thread_id,
                ))).scalar_one_or_none()
                if anchor is None:
                    raise RoutingContextUnavailable("routing run not found in scope")
                rows = (await session.execute(select(
                    agent_runs.c.id, agent_runs.c.question, agent_runs.c.answer,
                ).where(
                    agent_runs.c.user_id == scope.user_id, agent_runs.c.thread_id == thread_id,
                    agent_runs.c.status == "completed", agent_runs.c.finished_at <= anchor,
                    # MySQL DATETIME is second precision. IDs are server UUIDv7:
                    # break creation-time ties without including a later Run on replay.
                    or_(agent_runs.c.created_at < anchor, and_(
                        agent_runs.c.created_at == anchor, agent_runs.c.id < run_id)),
                ).order_by(agent_runs.c.created_at.desc(), agent_runs.c.id.desc()).limit(6))).mappings().all()
            turns = []
            map_references = []
            for row in reversed(rows):
                turns.append(RoutingTurn(id=f"query:{row['id']}:user", role="user", content=row["question"]))
                answer = project_public_answer(row["answer"])
                if answer and answer.tool_audited:
                    for index, card in enumerate(answer.cards or (), start=1):
                        reference = card.model_dump(mode="json", exclude_none=True, include={
                            "id", "kind", "title", "poi_id", "address", "location", "coordinate_system", "origin", "destination", "mode"})
                        map_references.append({"run_id": row["id"], "index": index, **reference})
                if answer and answer.segments:
                    content = "\n".join(s.text for s in answer.segments if s.kind != "references")
                    if content:
                        turns.append(RoutingTurn(id=f"query:{row['id']}:assistant", role="assistant", content=content))
            utc = anchor.replace(tzinfo=timezone.utc) if anchor.tzinfo is None else anchor
            # Keep complete references and their original visible ordinal; never
            # mix in a raw tool response or a foreign session's coordinates.
            map_references = map_references[-24:]
            while map_references and len(json.dumps(map_references, ensure_ascii=False)) > 6000:
                map_references.pop(0)
            return RoutingContext(run_id=run_id, user_id=scope.user_id, thread_id=thread_id,
                                  requested_at=utc.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
                                  history=bound_history(turns), map_references=tuple(map_references))
        except RoutingContextUnavailable:
            raise
        except Exception as error:
            raise RoutingContextUnavailable("routing context unavailable") from error
