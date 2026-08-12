"""Fail-closed policy layer over a tenant-scoped Mem0 client."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

from agentic_rag.domain.models import UserScope
from agentic_rag.memory.mem0_adapter import Mem0Adapter, as_mem0_adapter
from agentic_rag.memory.models import (
    MemoryClient,
    MemoryContext,
    MemoryRecord,
    MemoryTombstoneStore,
    MemoryType,
    PublicMessage,
)
from agentic_rag.safety.context import DataEnvelope


@runtime_checkable
class MemoryService(Protocol):
    """The only memory API exposed to query-graph dependencies."""

    async def load_context(
        self, scope: UserScope, query: str, limit: int = 10
    ) -> MemoryContext: ...

    async def extract_and_store(
        self, scope: UserScope, run_id: str, messages: Sequence[PublicMessage]
    ) -> None: ...

    async def list(self, scope: UserScope) -> list[MemoryRecord]: ...

    async def delete(self, scope: UserScope, memory_id: str) -> None: ...

    async def reconcile_deletions(self) -> None: ...


class MemoryServiceImpl:
    """Enforce user isolation and durable deletion around an unreliable provider.

    Provider outages are operational degradation: reads are empty and writes are
    ignored.  Scope mismatches are security failures and therefore return no
    data / perform no mutation.
    """

    def __init__(
        self,
        client: MemoryClient | Mem0Adapter,
        *,
        tombstones: MemoryTombstoneStore,
        policy_version: str,
    ) -> None:
        self._mem0 = as_mem0_adapter(client)
        self._tombstones = tombstones
        self._policy_version = policy_version

    async def load_context(
        self, scope: UserScope, query: str, limit: int = 10
    ) -> MemoryContext:
        if limit < 1:
            raise ValueError("limit must be positive")
        try:
            records = self._records(await self._mem0.search(query, user_id=scope.user_id, limit=limit), scope)
        except _operational_errors():
            return MemoryContext(degraded=True)
        envelopes = tuple(
            DataEnvelope(
                source_label=f"memory:{scope.user_id}",
                evidence_id=record.id,
                content=record.text,
            )
            for record in records
        )
        return MemoryContext(
            records=tuple(records),
            envelopes=envelopes,
            rendered_context="\n".join(envelope.render() for envelope in envelopes),
        )

    async def extract_and_store(
        self, scope: UserScope, run_id: str, messages: Sequence[PublicMessage]
    ) -> None:
        for message in messages:
            # Assistant content only crosses this boundary when a preceding user
            # confirmation is explicitly represented by its public message ID.
            source_id = (
                message.id
                if message.role == "user"
                else message.confirmation_message_id
                if message.role == "assistant" and message.confirmed_by_user
                else None
            )
            if source_id is None or not message.content.strip():
                continue
            metadata: dict[str, object] = {
                "user_id": scope.user_id,
                "memory_type": MemoryType.SEMANTIC.value,
                "source_run_id": run_id,
                "source_message_ids": [source_id],
                "policy_version": self._policy_version,
            }
            try:
                await self._mem0.add(
                    [{"role": "user", "content": message.content}],
                    user_id=scope.user_id,
                    metadata=metadata,
                )
            except _operational_errors():
                # Memory capture must not turn an otherwise valid query into a
                # failed run; a later user statement may safely be captured.
                return

    async def list(self, scope: UserScope) -> builtins.list[MemoryRecord]:
        try:
            return self._records(await self._mem0.get_all(user_id=scope.user_id), scope)
        except _operational_errors():
            return []

    async def delete(self, scope: UserScope, memory_id: str) -> None:
        # Refuse an unowned / nonexistent ID before creating a deletion record.
        try:
            visible = self._records(await self._mem0.get_all(user_id=scope.user_id), scope)
        except _operational_errors():
            return
        if memory_id not in {record.id for record in visible}:
            return

        try:
            await self._tombstones.request(scope, memory_id)
        except _operational_errors():
            return
        await self._attempt_delete(scope, memory_id)

    async def reconcile_deletions(self) -> None:
        try:
            pending = await self._tombstones.list_pending()
        except _operational_errors():
            return
        for tombstone in pending:
            # Tombstones are application-owned; reconstructing this scope is safe.
            await self._attempt_delete(UserScope(user_id=tombstone.user_id), tombstone.memory_id)

    async def _attempt_delete(self, scope: UserScope, memory_id: str) -> None:
        try:
            await self._mem0.delete(memory_id)
            remaining = self._records(
                await self._mem0.get_all(user_id=scope.user_id), scope
            )
            if memory_id in {record.id for record in remaining}:
                raise OSError("provider still returns deleted memory")
        except _operational_errors() as error:
            try:
                await self._tombstones.mark_retry(scope, memory_id, str(error))
            except _operational_errors():
                pass
            return
        try:
            await self._tombstones.mark_completed(scope, memory_id)
        except _operational_errors():
            # Delete verification succeeded, but only durable completion permits
            # dropping the retry work item; reconciliation remains idempotent.
            return

    @staticmethod
    def _records(response: object, scope: UserScope) -> builtins.list[MemoryRecord]:
        rows = _result_rows(response)
        records: builtins.list[MemoryRecord] = []
        for row in rows:
            record = _record_from_row(row)
            # The client namespace is not enough: provider response metadata is
            # untrusted and must independently agree with the server scope.
            if record is not None and record.user_id == scope.user_id:
                records.append(record)
        return records


def _result_rows(response: object) -> builtins.list[Mapping[str, object]]:
    raw_rows: object = response
    if isinstance(response, Mapping):
        raw_rows = response.get("results", response.get("memories", []))
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return []
    return [cast(Mapping[str, object], row) for row in raw_rows if isinstance(row, Mapping)]


def _record_from_row(row: Mapping[str, object]) -> MemoryRecord | None:
    metadata = row.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    record_id = row.get("id")
    text = row.get("memory", row.get("text"))
    user_id = metadata_map.get("user_id", row.get("user_id"))
    if not all(isinstance(value, str) and value for value in (record_id, text, user_id)):
        return None
    assert isinstance(record_id, str)
    assert isinstance(text, str)
    assert isinstance(user_id, str)
    memory_type_raw = metadata_map.get("memory_type", MemoryType.SEMANTIC.value)
    try:
        memory_type = MemoryType(str(memory_type_raw))
    except ValueError:
        return None
    source_ids = metadata_map.get("source_message_ids", ())
    if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)):
        source_ids = ()
    return MemoryRecord(
        id=record_id,
        user_id=user_id,
        text=text,
        memory_type=memory_type,
        source_run_id=_optional_string(metadata_map.get("source_run_id")),
        source_message_ids=tuple(item for item in source_ids if isinstance(item, str)),
        policy_version=_optional_string(metadata_map.get("policy_version")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


# Network/SDK failures are intentionally narrow; invalid provider data becomes
# an empty fail-closed result instead of a cross-tenant leak.
def _operational_errors() -> tuple[type[BaseException], ...]:
    return (OSError, TimeoutError, ConnectionError)
