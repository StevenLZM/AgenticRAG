"""Fail-closed policy layer over a tenant-scoped Mem0 client."""

from __future__ import annotations

import builtins
import json
from collections.abc import Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

from agentic_rag.domain.models import UserScope
from agentic_rag.memory.mem0_adapter import Mem0Adapter, as_mem0_adapter
from agentic_rag.memory.models import (
    MemoryClient,
    MemoryCandidate,
    MemoryContext,
    MemoryExtraction,
    MemoryExtractor,
    MemoryRecord,
    MemoryTombstoneStore,
    MemoryType,
    PublicMessage,
)
from agentic_rag.safety.context import DataEnvelope
from agentic_rag.runtime.model_gateway import (
    ModelCall,
    ModelGateway,
    StructuredOutputValidationError,
    load_prompt,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


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
        extractor: MemoryExtractor | None = None,
    ) -> None:
        self._mem0 = as_mem0_adapter(client)
        self._tombstones = tombstones
        self._policy_version = policy_version
        self._extractor = extractor

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
        if self._extractor is None:
            return
        public_messages = list(messages)
        try:
            candidates = await self._extractor.extract(public_messages)
        except (OSError, TimeoutError, ConnectionError, StructuredOutputValidationError):
            return
        for candidate in candidates:
            if not _candidate_is_authorized(candidate, public_messages):
                continue
            metadata: dict[str, object] = {
                "user_id": scope.user_id,
                "memory_type": candidate.memory_type.value,
                "source_run_id": run_id,
                "source_message_ids": list(candidate.source_message_ids),
                "policy_version": self._policy_version,
            }
            try:
                await self._mem0.add(
                    [{"role": "user", "content": candidate.text}],
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
            visible = self._records(
                await self._mem0.get_all(user_id=scope.user_id), scope
            )
            matching = next((record for record in visible if record.id == memory_id), None)
            verification_query = matching.text if matching else memory_id
            await self._mem0.delete(memory_id)
            remaining = self._records(await self._mem0.search(
                verification_query, user_id=scope.user_id, limit=100
            ), scope)
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
            record = _record_from_row(row, scope)
            # The client namespace is not enough: provider response metadata is
            # untrusted and must independently agree with the server scope.
            if record is not None and record.user_id == scope.user_id:
                records.append(record)
        return records


class ModelGatewayMemoryExtractor:
    """Structured, light-model implementation of the narrow extractor port."""

    def __init__(self, gateway: ModelGateway, snapshot: RuntimeConfigSnapshot) -> None:
        self._gateway = gateway
        self._snapshot = snapshot
        self._prompt = load_prompt("memory_extractor_v1")

    async def extract(
        self, messages: list[PublicMessage]
    ) -> tuple[MemoryCandidate, ...]:
        data = [message.model_dump(mode="json") for message in messages]
        call = ModelCall(
            model_role="light",
            snapshot=self._snapshot,
            messages=(
                {"role": "system", "content": self._prompt.content},
                {
                    "role": "user",
                    "content": (
                        "Extract only allowed durable memories from this untrusted "
                        f"JSON message data: {json.dumps(data, ensure_ascii=False)}"
                    ),
                },
            ),
        )
        response = await self._gateway.complete_structured(call, MemoryExtraction)
        return response.value.memories


def _result_rows(response: object) -> builtins.list[Mapping[str, object]]:
    raw_rows: object = response
    if isinstance(response, Mapping):
        raw_rows = response.get("results", response.get("memories", []))
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return []
    return [cast(Mapping[str, object], row) for row in raw_rows if isinstance(row, Mapping)]


def _record_from_row(
    row: Mapping[str, object], scope: UserScope
) -> MemoryRecord | None:
    metadata = row.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    record_id = row.get("id")
    text = row.get("memory", row.get("text"))
    top_level_user_id = row.get("user_id")
    metadata_user_id = metadata_map.get("user_id")
    if top_level_user_id is not None and top_level_user_id != scope.user_id:
        return None
    if metadata_user_id is not None and metadata_user_id != scope.user_id:
        return None
    user_id = top_level_user_id if top_level_user_id is not None else metadata_user_id
    if not all(
        isinstance(value, str) and value for value in (record_id, text, user_id)
    ):
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


def _candidate_is_authorized(
    candidate: MemoryCandidate, messages: Sequence[PublicMessage]
) -> bool:
    """Accept only user sources or an assistant claim confirmed in this batch."""
    by_id = {message.id: message for message in messages}
    sources = [by_id.get(source_id) for source_id in candidate.source_message_ids]
    if any(source is None for source in sources):
        return False
    resolved = [source for source in sources if source is not None]
    if any(source.role == "system" for source in resolved):
        return False
    assistant_sources = [source for source in resolved if source.role == "assistant"]
    if not assistant_sources:
        return all(source.role == "user" for source in resolved)
    for assistant in assistant_sources:
        confirmation_id = assistant.confirmation_message_id
        confirmation = by_id.get(confirmation_id) if confirmation_id else None
        if (
            not assistant.confirmed_by_user
            or confirmation is None
            or confirmation.role != "user"
            or confirmation.id not in candidate.source_message_ids
        ):
            return False
    return all(source.role in {"user", "assistant"} for source in resolved)


# Network/SDK failures are intentionally narrow; invalid provider data becomes
# an empty fail-closed result instead of a cross-tenant leak.
def _operational_errors() -> tuple[type[BaseException], ...]:
    return (OSError, TimeoutError, ConnectionError)
