"""Short, caller-owned SQL operations for scoped chat sessions."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, insert, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.domain.chat_sessions import (
    ChatSession, ChatSessionSummary, Page, decode_cursor, encode_cursor, page_limit, utc_datetime,
)
from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.persistence.repositories import agent_runs, chat_sessions, QueryRun, _run_from_row
from agentic_rag.runtime.ids import new_id


def _from_row(row: Mapping[str, Any]) -> ChatSession:
    return ChatSession(
        id=row["id"], user_id=row["user_id"], creation_request_id=row["creation_request_id"],
        title=row["title"], title_source=row["title_source"],
        created_at=utc_datetime(row["created_at"]), updated_at=utc_datetime(row["updated_at"]),
        last_activity_at=utc_datetime(row["last_activity_at"]),
        deleted_at=utc_datetime(row["deleted_at"]) if row["deleted_at"] else None,
    )


class SqlAlchemyChatSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_owned(self, scope: UserScope, session_id: str, *, for_update: bool = False,
                        include_deleted: bool = False) -> ChatSession | None:
        query = select(chat_sessions).where(chat_sessions.c.id == session_id, chat_sessions.c.user_id == scope.user_id)
        if not include_deleted:
            query = query.where(chat_sessions.c.deleted_at.is_(None))
        if for_update:
            query = query.with_for_update()
        row = (await self.session.execute(query)).mappings().one_or_none()
        return _from_row(row) if row else None

    async def get_by_creation(self, scope: UserScope, request_id: str) -> ChatSession | None:
        row = (await self.session.execute(select(chat_sessions).where(
            chat_sessions.c.user_id == scope.user_id, chat_sessions.c.creation_request_id == request_id,
        ))).mappings().one_or_none()
        return _from_row(row) if row else None

    async def create(self, scope: UserScope, request_id: str) -> ChatSession:
        now = datetime.now(UTC)
        values = dict(id=new_id(), user_id=scope.user_id, creation_request_id=request_id,
                      title="新对话", title_source="default", created_at=now, updated_at=now,
                      last_activity_at=now, deleted_at=None)
        await self.session.execute(insert(chat_sessions).values(**values))
        return _from_row(values)

    async def rename(self, scope: UserScope, session_id: str, title: str) -> None:
        await self.session.execute(update(chat_sessions).where(
            chat_sessions.c.user_id == scope.user_id, chat_sessions.c.id == session_id,
            chat_sessions.c.deleted_at.is_(None),
        ).values(title=title, title_source="manual", updated_at=datetime.now(UTC)))

    async def delete(self, scope: UserScope, session_id: str) -> None:
        now = datetime.now(UTC)
        await self.session.execute(update(chat_sessions).where(
            chat_sessions.c.user_id == scope.user_id, chat_sessions.c.id == session_id,
            chat_sessions.c.deleted_at.is_(None),
        ).values(deleted_at=now, updated_at=now))

    async def list(self, scope: UserScope, *, cursor: str | None = None, limit: int = 30) -> Page[ChatSessionSummary]:
        limit = page_limit(limit)
        query = select(chat_sessions, agent_runs.c.id.label("active_run_id"), agent_runs.c.status.label("active_run_status")).select_from(
            chat_sessions.outerjoin(agent_runs, and_(
                agent_runs.c.user_id == chat_sessions.c.user_id,
                agent_runs.c.thread_id == chat_sessions.c.id, agent_runs.c.active_slot == 1,
            ))
        ).where(chat_sessions.c.user_id == scope.user_id, chat_sessions.c.deleted_at.is_(None))
        if cursor is not None:
            stamp, item_id = decode_cursor(cursor, "sessions")
            query = query.where(or_(chat_sessions.c.last_activity_at < stamp, and_(
                chat_sessions.c.last_activity_at == stamp, chat_sessions.c.id < item_id)))
        rows = (await self.session.execute(query.order_by(
            chat_sessions.c.last_activity_at.desc(), chat_sessions.c.id.desc()).limit(limit + 1))).mappings().all()
        items = tuple(ChatSessionSummary(_from_row(row), row["active_run_id"],
                      RunStatus(row["active_run_status"]) if row["active_run_status"] else None) for row in rows[:limit])
        next_cursor = encode_cursor("sessions", items[-1].session.last_activity_at, items[-1].session.id) if len(rows) > limit else None
        return Page(items, next_cursor)

    async def find_submission(self, scope: UserScope, session_id: str, client_request_id: str) -> QueryRun | None:
        row = (await self.session.execute(select(agent_runs).where(
            agent_runs.c.user_id == scope.user_id, agent_runs.c.thread_id == session_id,
            agent_runs.c.client_request_id == client_request_id,
        ))).mappings().one_or_none()
        return _run_from_row(dict(row)) if row else None

    async def list_turns(self, scope: UserScope, session_id: str, *, cursor: str | None = None,
                         limit: int = 30) -> Page[QueryRun]:
        limit = page_limit(limit)
        query = select(agent_runs).where(agent_runs.c.user_id == scope.user_id, agent_runs.c.thread_id == session_id)
        if cursor is not None:
            stamp, item_id = decode_cursor(cursor, "turns")
            query = query.where(or_(agent_runs.c.created_at < stamp, and_(
                agent_runs.c.created_at == stamp, agent_runs.c.id < item_id)))
        rows = (await self.session.execute(query.order_by(agent_runs.c.created_at.desc(), agent_runs.c.id.desc())
                                           .limit(limit + 1))).mappings().all()
        selected = rows[:limit]
        next_cursor = encode_cursor("turns", selected[-1]["created_at"], selected[-1]["id"]) if len(rows) > limit else None
        return Page(tuple(_run_from_row(dict(row)) for row in reversed(selected)), next_cursor)
