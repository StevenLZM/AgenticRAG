"""Native adapters preserve server scope and return checkpoint-safe values."""

from __future__ import annotations

import pytest

from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.tool_runtime.models import ToolError
from agentic_rag.tool_runtime.native import NativeAdapter
from tests.unit.tool_runtime.test_runtime import context


class ScopedRetrieval:
    async def retrieve(self, request, scope, snapshot):
        assert scope.user_id == "alice"
        assert snapshot.index_generation == "test"
        return EvidenceBatch(query=request.query, parents=(), document_ids=request.document_ids)


async def test_native_retrieval_returns_evidence_batch_with_selectors():
    adapter = NativeAdapter(ScopedRetrieval())
    tools = {tool.tool_id: tool for tool in await adapter.list_tools(context())}
    result = await adapter.call_tool(tools["local.knowledge_search"], {"query": "contract", "document_ids": ["doc-1"]}, context())
    assert result["batch"]["query"] == "contract"
    assert result["batch"]["document_ids"] == ["doc-1"]
    assert EvidenceBatch.model_validate(result["batch"]).parents == ()


async def test_native_calculator_uses_safe_evaluator():
    adapter = NativeAdapter(ScopedRetrieval())
    tools = {tool.tool_id: tool for tool in await adapter.list_tools(context())}
    result = await adapter.call_tool(tools["local.calculator"], {"expression": "(12 + 8) / 4"}, context())
    assert result == {"value": 5.0}
    with pytest.raises(ToolError, match="calculator_input_invalid"):
        await adapter.call_tool(tools["local.calculator"], {"expression": "__import__('os').environ"}, context())


async def test_native_retrieval_rejects_identity_injection_even_without_runtime():
    adapter = NativeAdapter(ScopedRetrieval())
    tools = {tool.tool_id: tool for tool in await adapter.list_tools(context())}
    with pytest.raises(ToolError, match="invalid_arguments"):
        await adapter.call_tool(tools["local.knowledge_search"], {"query": "contract", "user_id": "bob"}, context())


async def test_native_retrieval_never_serializes_cross_user_evidence():
    class Contaminated:
        async def retrieve(self, request, scope, snapshot):
            return EvidenceBatch(query=request.query, parents=(ParentEvidence(
                parent_id="parent-1", document_id="doc-1", document_version_id="v1", content="private",
                child_hits=(ChildHit(child_id="child-1", parent_id="parent-1", user_id="bob",
                                     document_id="doc-1", document_version_id="v1", content="private",
                                     ast_locator="#", lane="dense", lane_rank=1, retrieval_score=1),),
            ),))
    adapter = NativeAdapter(Contaminated())
    tools = {tool.tool_id: tool for tool in await adapter.list_tools(context())}
    with pytest.raises(ToolError, match="evidence_scope_mismatch"):
        await adapter.call_tool(tools["local.knowledge_search"], {"query": "contract"}, context())
