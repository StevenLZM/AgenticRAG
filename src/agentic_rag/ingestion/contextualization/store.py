"""Durable context-build port; explicit in-memory test double only."""

from __future__ import annotations
from typing import Protocol
from uuid import uuid4
from pydantic import BaseModel, ConfigDict
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.models.contextual import ContextArtifactRef
from agentic_rag.runtime.ids import content_id


class BudgetExhausted(RuntimeError):
    pass


class BuildKey(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: str
    document_id: str
    document_version_id: str
    build_fingerprint: str
    source_child_id: str

    @property
    def build_id(self):
        return content_id(
            self.user_id,
            self.document_id,
            self.document_version_id,
            self.build_fingerprint,
        )

    @property
    def item_id(self):
        return content_id(self.build_id, self.source_child_id)

    def check_claim(self, claim):
        if (self.user_id, self.document_id, self.document_version_id) != (
            claim.user_id,
            claim.document_id,
            claim.document_version_id,
        ):
            raise ValueError("context build identity differs from job")


class ContextBuildStore(Protocol):
    async def get_ready(
        self, key: BuildKey, claim: IngestionJobClaim
    ) -> ContextArtifactRef | None: ...
    async def reserve(
        self,
        key: BuildKey,
        claim: IngestionJobClaim,
        *,
        tokens: int,
        cost: float,
        build,
    ) -> str: ...
    async def settle(
        self,
        key: BuildKey,
        claim: IngestionJobClaim,
        reservation_id: str,
        *,
        tokens: int,
        cost: float,
    ) -> None: ...
    async def accept(
        self, key: BuildKey, claim: IngestionJobClaim, ref: ContextArtifactRef
    ) -> ContextArtifactRef: ...


class MemoryContextBuildStore:
    """Explicit offline double. Never constructed by production composition."""

    def __init__(self):
        self.ready = {}
        self.usage = {}
        self.reservations = {}

    async def get_ready(self, key, claim):
        key.check_claim(claim)
        return self.ready.get(key.item_id)

    async def reserve(self, key, claim, *, tokens, cost, build):
        key.check_claim(claim)
        requests, used_tokens, used_cost = self.usage.get(key.build_id, (0, 0, 0.0))
        if (
            requests + 1 > build.max_requests
            or used_tokens + tokens > build.max_tokens
            or used_cost + cost > build.max_cost
        ):
            raise BudgetExhausted("context document budget exhausted")
        self.usage[key.build_id] = (
            requests + 1,
            used_tokens + tokens,
            used_cost + cost,
        )
        rid = uuid4().hex
        self.reservations[rid] = (key.build_id, tokens, cost, False)
        return rid

    async def settle(self, key, claim, reservation_id, *, tokens, cost):
        key.check_claim(claim)
        bid, old_tokens, old_cost, settled = self.reservations[reservation_id]
        if bid != key.build_id:
            raise ValueError("reservation scope mismatch")
        if settled:
            return
        req, ut, uc = self.usage[bid]
        self.usage[bid] = (req, ut - old_tokens + tokens, uc - old_cost + cost)
        self.reservations[reservation_id] = (bid, tokens, cost, True)

    async def accept(self, key, claim, ref):
        key.check_claim(claim)
        return self.ready.setdefault(key.item_id, ref)
