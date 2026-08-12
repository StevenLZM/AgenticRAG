"""Memory loading and fail-closed routing for the query entry graph."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agentic_rag.memory.models import MemoryContext
from agentic_rag.memory.service import MemoryService
from agentic_rag.models.schemas import RouteDecision
from agentic_rag.query.state import QueryState, question_from_state, scope_from_state, snapshot_from_state
from agentic_rag.runtime.model_gateway import (
    ModelCall,
    ModelGateway,
    StructuredOutputValidationError,
    load_prompt,
)

if TYPE_CHECKING:
    from agentic_rag.query.fast_rag import FastRagDependencies


@dataclass(frozen=True, slots=True)
class QueryRuntimeDependencies:
    """Clients and services intentionally excluded from checkpoint state."""

    memory: MemoryService
    gateway: ModelGateway
    fast_rag: "FastRagDependencies"


class MemoryContextLoader:
    """Load memory once, before route selection, and cache its JSON envelope."""

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    async def load(self, state: QueryState) -> dict[str, object]:
        cached = state.get("memory_context")
        if isinstance(cached, dict):
            return {"memory_context": cached}
        scope = scope_from_state(state)
        question = question_from_state(state)
        try:
            context = await self._memory.load_context(scope, question)
        except asyncio.CancelledError:
            raise
        except (OSError, TimeoutError, ConnectionError) as error:
            context = MemoryContext(degraded=True)
            errors = [*state.get("errors", []), {"code": "memory_unavailable", "detail": str(error)}]
            return {
                "memory_context": context.model_dump(mode="json"),
                "errors": errors,
            }
        return {"memory_context": context.model_dump(mode="json")}


async def route_query(state: QueryState, gateway: ModelGateway) -> dict[str, object]:
    """Ask the light router once; invalid structured output always escalates."""
    prompt = load_prompt("router_v1")
    question = question_from_state(state)
    snapshot = snapshot_from_state(state)
    context = state.get("memory_context", {})
    call = ModelCall(
        model_role="light",
        snapshot=snapshot,
        messages=(
            {"role": "system", "content": prompt.content},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "memory_context": context},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ),
    )
    try:
        response = await gateway.complete_structured(call, RouteDecision)
        decision = _route_decision(response.value)
    except asyncio.CancelledError:
        raise
    except (StructuredOutputValidationError, ValidationError, TypeError, ValueError) as error:
        return _research_fallback(state, "router_schema_invalid", str(error))
    except (OSError, TimeoutError, ConnectionError) as error:
        return _research_fallback(state, "router_unavailable", str(error))
    next_node = "fast_rag" if decision.route == "fast_rag" else "research_agent"
    return {"route": decision.model_dump(mode="json"), "next_node": next_node}


def build_query_entry_graph(
    dependencies: QueryRuntimeDependencies,
) -> CompiledStateGraph[QueryState, None, QueryState, QueryState]:
    """Compile the Phase 4 entry subgraph with all dependencies in closures.

    The research and generation branches are implemented by later tasks.  This
    entry graph safely ends after recording the selected next node so those
    tasks can attach their own nodes without exposing clients in state.
    """
    from agentic_rag.query.fast_rag import run_fast_rag

    loader = MemoryContextLoader(dependencies.memory)

    async def load_memory(state: QueryState) -> dict[str, object]:
        return await loader.load(state)

    async def route(state: QueryState) -> dict[str, object]:
        return await route_query(state, dependencies.gateway)

    async def fast_rag(state: QueryState) -> dict[str, object]:
        return await run_fast_rag(state, dependencies.fast_rag)

    def after_route(state: QueryState) -> str:
        return "fast_rag" if state.get("next_node") == "fast_rag" else END

    builder = StateGraph(QueryState)
    builder.add_node("load_memory", load_memory)
    builder.add_node("route", route)
    builder.add_node("fast_rag", fast_rag)
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "route")
    builder.add_conditional_edges("route", after_route, {"fast_rag": "fast_rag", END: END})
    builder.add_edge("fast_rag", END)
    return builder.compile(name="QueryEntryGraph")


def _route_decision(value: object) -> RouteDecision:
    if isinstance(value, RouteDecision):
        return value
    if isinstance(value, Mapping):
        return RouteDecision.model_validate(value)
    raise TypeError("router response must be a RouteDecision")


def _research_fallback(state: QueryState, code: str, detail: str) -> dict[str, object]:
    question = question_from_state(state)
    return {
        "route": RouteDecision(
            route="research", normalized_query=question, reason_code="fail_closed"
        ).model_dump(mode="json"),
        "next_node": "research_agent",
        "errors": [*state.get("errors", []), {"code": code, "detail": detail}],
    }
