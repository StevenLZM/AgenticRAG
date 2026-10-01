"""Re-authorize immutable answer excerpts each time a user opens sources."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.chat_sessions import SessionNotFound
from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.persistence.chat_sessions import SqlAlchemyChatSessionRepository
from agentic_rag.persistence.repositories import SqlAlchemyRunRepository
from agentic_rag.query.answer_sources import AnswerSources, cited_ids, within_budget
from agentic_rag.query.public_answer import project_public_answer


class SourceItemView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    filename: str
    heading_path: tuple[str, ...]
    heading_truncated: bool
    page_from: int | None
    page_to: int | None
    excerpt: str
    truncated: bool
    excerpt_omitted: bool
    version_status: Literal["current", "historical"]


class SourceView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["available", "partial", "unavailable", "none"]
    items: tuple[SourceItemView, ...] = ()
    omitted_source_count: int = 0


class ChatSourceService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def get(self, scope: UserScope, session_id: str, run_id: str) -> SourceView:
        async with self.session_factory() as db:
            repo = SqlAlchemyChatSessionRepository(db)
            if await repo.get_owned(scope, session_id) is None:
                raise SessionNotFound(session_id)
            run = await SqlAlchemyRunRepository(db).get(run_id, scope)
            if run is None or run.thread_id != session_id:
                raise SessionNotFound(run_id)
            ids = cited_ids(project_public_answer(run.answer))
            if not ids or run.status != RunStatus.COMPLETED:
                return SourceView(status="none")
            raw = run.answer_sources
            try:
                if not isinstance(raw, dict) or not within_budget(raw):
                    return SourceView(status="unavailable")
                snapshot = AnswerSources.model_validate(raw)
            except (ValidationError, TypeError, ValueError):
                return SourceView(status="unavailable")
            if (
                snapshot.run_id != run.id
                or snapshot.runtime_config_snapshot_id != run.runtime_config_snapshot_id
                or tuple(item.evidence_id for item in snapshot.items) != ids[:64]
                or snapshot.omitted_source_count != max(0, len(ids) - 64)
            ):
                return SourceView(status="unavailable")
            access = await repo.source_documents(
                scope,
                [
                    (item.document_id, item.document_version_id)
                    for item in snapshot.items
                ],
            )
            items = []
            for source in snapshot.items:
                document = access.get((source.document_id, source.document_version_id))
                if document is None:
                    continue
                items.append(
                    SourceItemView(
                        **source.model_dump(
                            exclude={"document_id", "document_version_id", "parent_id"}
                        ),
                        filename=document.filename,
                        version_status="current"
                        if document.active_version_id == source.document_version_id
                        else "historical",
                    )
                )
            omitted = snapshot.omitted_source_count + len(snapshot.items) - len(items)
            return SourceView(
                status=("partial" if omitted else "available")
                if items
                else "unavailable",
                items=tuple(items),
                omitted_source_count=omitted,
            )
