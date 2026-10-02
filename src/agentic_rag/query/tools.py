"""Narrow, injection-friendly research tools with safe observations only."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag.domain.models import UserScope
from agentic_rag.query.calculator import SafeCalculator, UnsafeExpression
from agentic_rag.query.evidence_builder import EvidenceBuilder, EvidenceCoverageTarget, PackedEvidence
from agentic_rag.retrieval.models import EvidenceBatch, RetrievalRequest
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.tool_runtime.models import ToolContext, ToolError


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
    tool_context: ToolContext | None = None
    call_prefix: str = "research"


class ResearchToolset:
    """Expose only bounded retrieval packing and arithmetic to the agent loop."""

    def __init__(self, retrieval: RetrievalPort, evidence_builder: EvidenceBuilder, tool_runtime: Any = None) -> None:
        self._retrieval = retrieval
        self._evidence_builder = evidence_builder
        self._calculator = SafeCalculator()
        self._runtime = tool_runtime

    async def retrieve_evidence(
        self, *, query: str, ctx: ResearchContext, target_id: str
    ) -> tuple[EvidenceBatch, PackedEvidence]:
        request = RetrievalRequest(query=query)
        if self._runtime is None:
            batch = await self._retrieval.retrieve(request, ctx.scope, ctx.snapshot)
        else:
            if ctx.tool_context is None:
                raise ToolError("tool_context_missing")
            result = await self._runtime.call("local.knowledge_search", {"query": query}, ctx.tool_context,
                call_id=f"{ctx.call_prefix}:retrieve:{target_id}")
            if result.status != "success":
                raise ToolError(result.error_code or "retrieval_unavailable", result.retryable)
            batch = EvidenceBatch.model_validate(result.data["batch"])
        if target_id not in batch.target_ids:
            batch = batch.model_copy(update={"target_ids": (*batch.target_ids, target_id)})
        packed = self._evidence_builder.build(
            [batch],
            [EvidenceCoverageTarget(target_id=target_id, description=query)],
            ctx.scope,
            ctx.snapshot,
        )
        return batch, packed

    async def calculator(self, expression: str, *, ctx: ResearchContext | None = None) -> dict[str, object]:
        if self._runtime is not None:
            if ctx is None or ctx.tool_context is None:
                raise ToolError("tool_context_missing")
            result = await self._runtime.call("local.calculator", {"expression": expression}, ctx.tool_context,
                                             call_id=f"{ctx.call_prefix}:calculator")
            return {"ok": result.status == "success", **result.data,
                    **({"error_code": result.error_code, "retryable": result.retryable} if result.status != "success" else {})}
        try:
            value = self._calculator.evaluate(expression)
        except asyncio.CancelledError:
            raise
        except UnsafeExpression:
            return {"ok": False, "error_code": "calculator_input_invalid", "retryable": False}
        return {"ok": True, "value": value}
