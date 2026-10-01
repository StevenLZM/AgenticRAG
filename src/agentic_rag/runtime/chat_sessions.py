"""Transactional session management, independent of the transport layer."""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.chat_sessions import (
    ChatSession, ChatSessionSummary, Page, SessionBusy, SessionGone, SessionNotFound, request_uuid,
)
from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.chat_sessions import SqlAlchemyChatSessionRepository
from agentic_rag.persistence.repositories import SqlAlchemyRunRepository, _is_mysql_duplicate_for
from agentic_rag.runtime.run_manager import RunManager


class ChatSessionService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], run_manager: RunManager) -> None:
        self.session_factory = session_factory
        self.run_manager = run_manager

    async def create(self, scope: UserScope, creation_request_id: str) -> tuple[ChatSession, bool]:
        request_id = request_uuid(creation_request_id)
        try:
            async with self.session_factory.begin() as db:
                repo = SqlAlchemyChatSessionRepository(db)
                existing = await repo.get_by_creation(scope, request_id)
                if existing is not None:
                    return self._creation_replay(existing)
                return await repo.create(scope, request_id), True
        except IntegrityError as error:
            if not (_is_mysql_duplicate_for(error, "uq_chat_session_creation") or
                    "UNIQUE constraint failed: chat_sessions.user_id, chat_sessions.creation_request_id" in str(error.orig)):
                raise
            # The failed INSERT transaction has exited and rolled back here.
            async with self.session_factory() as db:
                existing = await SqlAlchemyChatSessionRepository(db).get_by_creation(scope, request_id)
                if existing is None:
                    raise
                return self._creation_replay(existing)

    @staticmethod
    def _creation_replay(session: ChatSession) -> tuple[ChatSession, bool]:
        if session.deleted_at is not None:
            raise SessionGone("The created session has been deleted.")
        return session, False

    async def get(self, scope: UserScope, session_id: str) -> ChatSessionSummary:
        async with self.session_factory() as db:
            session = await SqlAlchemyChatSessionRepository(db).get_owned(scope, session_id)
            if session is None:
                raise SessionNotFound(session_id)
            run = await SqlAlchemyRunRepository(db).get_active(scope, session_id)
            return ChatSessionSummary(session, run.id if run else None, run.status if run else None)

    async def list(self, scope: UserScope, *, cursor: str | None = None, limit: int = 30) -> Page[ChatSessionSummary]:
        async with self.session_factory() as db:
            return await SqlAlchemyChatSessionRepository(db).list(scope, cursor=cursor, limit=limit)

    async def rename(self, scope: UserScope, session_id: str, title: str) -> ChatSession:
        title = title.strip()
        if not 1 <= len(title) <= 100:
            raise ValueError("title must contain 1 to 100 characters")
        async with self.session_factory.begin() as db:
            repo = SqlAlchemyChatSessionRepository(db)
            if await repo.get_owned(scope, session_id, for_update=True) is None:
                raise SessionNotFound(session_id)
            await repo.rename(scope, session_id, title)
            result = await repo.get_owned(scope, session_id)
            assert result is not None
            return result

    async def delete(self, scope: UserScope, session_id: str) -> None:
        async with self.session_factory.begin() as db:
            repo = SqlAlchemyChatSessionRepository(db)
            session = await repo.get_owned(scope, session_id, for_update=True, include_deleted=True)
            if session is None:
                raise SessionNotFound(session_id)
            if session.deleted_at is not None:
                return
            active = await SqlAlchemyRunRepository(db).get_active(scope, session_id)
            if active is not None:
                raise SessionBusy(active.id)
            await repo.delete(scope, session_id)
