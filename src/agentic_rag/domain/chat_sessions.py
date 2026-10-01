"""Values shared by chat persistence, orchestration and HTTP projections."""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, Literal, TypeVar
from uuid import UUID

from agentic_rag.domain.models import RunStatus

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ChatSession:
    id: str
    user_id: str
    creation_request_id: str
    title: str
    title_source: Literal["default", "first_question", "manual"]
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ChatSessionSummary:
    session: ChatSession
    active_run_id: str | None
    active_run_status: RunStatus | None


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    next_cursor: str | None


class SessionNotFound(LookupError):
    """The session is absent, deleted or outside the caller's scope."""


class SessionGone(LookupError):
    """A replayed creation belongs to a deleted session."""


class SessionBusy(RuntimeError):
    def __init__(self, active_run_id: str) -> None:
        super().__init__("The session has an active query.")
        self.active_run_id = active_run_id


class IdempotencyConflict(RuntimeError):
    """A request identifier has already been used for a different question."""


def request_uuid(value: str) -> str:
    return str(UUID(value))


def utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def page_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("page limit must be between 1 and 100")
    return limit


def encode_cursor(kind: str, created_at: datetime, item_id: str) -> str:
    payload = [1, kind, utc_datetime(created_at).isoformat(timespec="microseconds"), item_id]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


def decode_cursor(value: str, kind: str) -> tuple[datetime, str]:
    try:
        if not value or len(value) > 2048:
            raise ValueError("invalid cursor length")
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        payload = json.loads(raw)
        if not isinstance(payload, list) or len(payload) != 4 or payload[:2] != [1, kind]:
            raise ValueError("invalid cursor kind")
        stamp = datetime.fromisoformat(payload[2])
        if stamp.tzinfo is None:
            raise ValueError("cursor must include timezone")
        return utc_datetime(stamp), request_uuid(payload[3])
    except (ValueError, TypeError, OverflowError, binascii.Error) as error:
        raise ValueError("invalid page cursor") from error


@dataclass(frozen=True)
class SourceDocumentAccess:
    document_id: str
    document_version_id: str
    filename: str
    active_version_id: str | None
