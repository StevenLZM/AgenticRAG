"""Tenant-scoped contracts for long-term memory.

Memory is retrieved text, not instructions.  These models deliberately keep
the associated user and source provenance alongside every returned value.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.domain.models import UserScope
from agentic_rag.safety.context import DataEnvelope


class MemoryType(StrEnum):
    """The permitted durable-memory categories."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class PublicMessage(BaseModel):
    """A public conversation message eligible for user-confirmed extraction."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)
    confirmed_by_user: bool = False
    confirmation_message_id: str | None = None


class MemoryRecord(BaseModel):
    """A normalised, user-owned memory returned from the provider."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    memory_type: MemoryType = MemoryType.SEMANTIC
    source_run_id: str | None = None
    source_message_ids: tuple[str, ...] = ()
    policy_version: str | None = None


class MemoryContext(BaseModel):
    """Prompt-safe memory context with an explicit availability signal."""

    model_config = ConfigDict(frozen=True)

    records: tuple[MemoryRecord, ...] = ()
    envelopes: tuple[DataEnvelope, ...] = ()
    rendered_context: str = ""
    degraded: bool = False


class Tombstone(BaseModel):
    """A deletion work item; pending state is intentionally retryable."""

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    status: Literal["pending", "completed", "failed"] = "pending"
    last_error: str | None = None


@runtime_checkable
class MemoryTombstoneStore(Protocol):
    """Durable deletion log owned by the application, not Mem0."""

    async def request(self, scope: UserScope, memory_id: str) -> Tombstone: ...

    async def list_pending(self, limit: int = 100) -> list[Tombstone]: ...

    async def mark_completed(self, scope: UserScope, memory_id: str) -> None: ...

    async def mark_retry(self, scope: UserScope, memory_id: str, error: str) -> None: ...


@runtime_checkable
class MemoryClient(Protocol):
    """Small async subset of mem0ai used by the application boundary."""

    async def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, object],
    ) -> object: ...

    async def search(
        self, query: str, *, user_id: str, limit: int
    ) -> object: ...

    async def get_all(self, *, user_id: str) -> object: ...

    async def delete(self, memory_id: str) -> object: ...
