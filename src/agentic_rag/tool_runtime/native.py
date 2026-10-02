"""Approved native retrieval and arithmetic through the shared adapter API."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from agentic_rag.query.calculator import SafeCalculator, UnsafeExpression
from agentic_rag.query.tools import RetrievalPort
from agentic_rag.retrieval.models import RetrievalRequest
from agentic_rag.tool_runtime.models import ToolContext, ToolDefinition, ToolError


class NativeAdapter:
    adapter_id = "local"

    def __init__(self, retrieval: RetrievalPort | None = None) -> None:
        self._retrieval = retrieval
        self._calculator = SafeCalculator()

    async def list_tools(self, context: ToolContext) -> tuple[ToolDefinition, ...]:
        calculator = ToolDefinition(
            tool_id="local.calculator", adapter_id=self.adapter_id, name="calculator",
            description="Evaluate bounded arithmetic. 安全数学计算、加减乘除。", source_kind="calculation",
            capabilities=("calculation", "arithmetic", "计算"),
            input_schema={"type": "object", "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 1000}},
                          "required": ["expression"], "additionalProperties": False},
        )
        if self._retrieval is None:
            return (calculator,)
        # Retrieval budgets are server owned. Explicit selectors remain available,
        # but the model cannot enlarge recall through top_k_override.
        schema = RetrievalRequest.model_json_schema()
        schema["properties"].pop("top_k_override", None)
        schema["properties"]["query"]["maxLength"] = 8000
        retrieval = ToolDefinition(
            tool_id="local.knowledge_search", adapter_id=self.adapter_id, name="knowledge_search",
            description="Search authorized document knowledge. 检索知识库文档、合同与资料。", source_kind="document",
            capabilities=("document", "knowledge", "search", "文档", "知识库", "检索"), input_schema=schema,
        )
        return retrieval, calculator

    async def call_tool(
        self, definition: ToolDefinition, arguments: dict[str, Any], context: ToolContext,
    ) -> dict[str, Any]:
        try:
            if definition.adapter_id != self.adapter_id:
                raise ToolError("tool_not_found")
            if definition.tool_id == "local.calculator":
                if set(arguments) != {"expression"}:
                    raise ToolError("invalid_arguments")
                return {"value": self._calculator.evaluate(arguments["expression"])}
            if definition.tool_id == "local.knowledge_search" and self._retrieval is not None:
                if "top_k_override" in arguments:
                    raise ToolError("invalid_arguments")
                request = RetrievalRequest.model_validate(arguments)
                batch = await self._retrieval.retrieve(request, context.scope, context.snapshot)
                for parent in batch.parents:
                    if not parent.child_hits or any(
                        hit.user_id != context.scope.user_id
                        or hit.parent_id != parent.parent_id
                        or hit.document_id != parent.document_id
                        or hit.document_version_id != parent.document_version_id
                        for hit in parent.child_hits
                    ):
                        raise ToolError("evidence_scope_mismatch")
                if batch.observation is not None and (
                    batch.observation.user_id != context.scope.user_id
                    or batch.observation.snapshot_id != context.snapshot.snapshot_id
                ):
                    raise ToolError("evidence_scope_mismatch")
                return {"batch": batch.model_dump(mode="json")}
            raise ToolError("tool_not_found")
        except asyncio.CancelledError:
            raise
        except UnsafeExpression:
            raise ToolError("calculator_input_invalid") from None
        except ValidationError:
            raise ToolError("invalid_arguments") from None
        except ToolError:
            raise
        except Exception:
            raise ToolError("tool_unavailable", retryable=True) from None

    async def aclose(self) -> None:
        """Retrieval is borrowed from composition and closed by its owner."""
