"""Narrow, injection-friendly research tools with safe observations only."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from agentic_rag.domain.models import UserScope
from agentic_rag.query.calculator import SafeCalculator, UnsafeExpression
from agentic_rag.query.evidence_builder import EvidenceBuilder, EvidenceCoverageTarget, PackedEvidence
from agentic_rag.retrieval.models import EvidenceBatch, RetrievalRequest
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class RetrievalPort(Protocol):
    async def retrieve(
        self,
        request: RetrievalRequest,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> EvidenceBatch: ...


@dataclass(frozen=True, slots=True)
class ResearchContext:
    """Server-derived collaborators deliberately outside serialized query state."""

    scope: UserScope
    snapshot: RuntimeConfigSnapshot


class ResearchToolset:
    """Expose only bounded retrieval packing and arithmetic to the agent loop."""

    def __init__(self, retrieval: RetrievalPort, evidence_builder: EvidenceBuilder) -> None:
        self._retrieval = retrieval
        self._evidence_builder = evidence_builder
        self._calculator = SafeCalculator()

    async def retrieve_evidence(
        self, *, query: str, ctx: ResearchContext, target_id: str
    ) -> tuple[EvidenceBatch, PackedEvidence]:
        request = RetrievalRequest(query=query)
        batch = await self._retrieval.retrieve(request, ctx.scope, ctx.snapshot)
        if target_id not in batch.target_ids:
            batch = batch.model_copy(update={"target_ids": (*batch.target_ids, target_id)})
        packed = self._evidence_builder.build(
            [batch],
            [EvidenceCoverageTarget(target_id=target_id, description=query)],
            ctx.scope,
            ctx.snapshot,
        )
        return batch, packed

    async def calculator(self, expression: str) -> dict[str, object]:
        try:
            value = self._calculator.evaluate(expression)
        except asyncio.CancelledError:
            raise
        except UnsafeExpression:
            return {"ok": False, "error_code": "calculator_input_invalid", "retryable": False}
        return {"ok": True, "value": value}
