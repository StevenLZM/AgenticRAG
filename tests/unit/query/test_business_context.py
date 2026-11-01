"""Frozen business meaning survives graph, tools and child projection."""

from dataclasses import replace
from agentic_rag.domain.models import UserScope

from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.query.graph import build_query_graph, query_checkpoint_config
from agentic_rag.query.state import business_from_state
from agentic_rag.query.tool_loop import tool_context
from agentic_rag.query.tools import ResearchContext, ResearchToolset
from agentic_rag.query.evidence_builder import EvidenceBuilder
from agentic_rag.retrieval.query_compiler import QueryCompiler
from agentic_rag.retrieval.query_context import TrustedBusinessProfile
from agentic_rag.retrieval.models import EvidenceBatch
from agentic_rag.tool_runtime.native import NativeAdapter
from tests.unit.query.test_graph import _deps, _state


class Profiles:
    revision = "p1"
    calls = 0

    async def get_profile(self, scope):
        self.calls += 1
        return TrustedBusinessProfile(
            owner_id=scope.user_id, revision=self.revision, companies=("星河公司",)
        )


async def test_graph_freezes_once_per_run_and_fast_path_inherits_context():
    deps, _, retrieval, _ = _deps(grades=["sufficient"] * 3)
    deps.gateway.answer_values[:] = [
        {"passed": True, "unsupported_claim_ids": [], "reasons": []}
    ] * 3
    profiles = Profiles()
    received = []
    original = retrieval.retrieve

    class Retrieval:
        async def retrieve(self, request, scope, snapshot, *, business_context):
            received.append(business_context)
            return await original(request, scope, snapshot)

    deps = replace(
        deps, query_compiler=QueryCompiler(profiles=profiles), retrieval=Retrieval()
    )
    graph = build_query_graph(deps, InMemorySaver())
    state = _state()
    config = query_checkpoint_config(state)
    first = await graph.ainvoke(state, config)
    profiles.revision = "p2"
    second = await graph.ainvoke(state, config)
    assert profiles.calls == 1
    assert (
        received[0]
        == received[1]
        == business_from_state(first)
        == business_from_state(second)
    )
    third = await graph.ainvoke(dict(state, run_id="next-run"), config)
    assert (
        profiles.calls == 2
        and business_from_state(third).trusted_profile.revision == "p2"
    )


async def test_tool_and_direct_research_share_frozen_context_and_reject_model_injection():
    import pytest
    from agentic_rag.tool_runtime.models import ToolError

    state = _state()
    compiler = QueryCompiler(profiles=Profiles())
    business = await compiler.freeze(
        "我出差住宿标准", UserScope.model_validate(state["scope"])
    )
    state["business_context"] = business.model_dump(mode="json")
    context = tool_context(state)

    class Retrieval:
        async def retrieve(self, request, scope, snapshot, *, business_context):
            assert business_context == business
            return EvidenceBatch(query=request.query, parents=())

    retrieval = Retrieval()
    native = NativeAdapter(retrieval)
    definition = next(
        t
        for t in await native.list_tools(context)
        if t.tool_id == "local.knowledge_search"
    )
    await native.call_tool(definition, {"query": "住宿标准"}, context)
    with pytest.raises(ToolError, match="invalid_arguments"):
        await native.call_tool(
            definition,
            {"query": "住宿标准", "business_context": business.model_dump(mode="json")},
            context,
        )
    tools = ResearchToolset(retrieval, EvidenceBuilder())
    await tools.retrieve_evidence(
        query="住宿标准",
        target_id="t1",
        ctx=ResearchContext(
            scope=context.scope, snapshot=context.snapshot, business_context=business
        ),
    )


async def test_composed_parallel_children_inherit_frozen_profile_and_task_bounds():
    from agentic_rag.runtime.query_composition import build_subagent_dispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager
    from agentic_rag.query.todos import TodoItem
    from tests.unit.query.test_graph import SCOPE, SNAPSHOT

    profiles = Profiles()
    compiler = QueryCompiler(profiles=profiles)
    business = await compiler.freeze("我出差住宿标准", SCOPE)
    profiles.revision = "p2"
    received = []

    class Retrieval:
        async def retrieve(self, request, scope, snapshot, *, business_context):
            received.append(business_context)
            assert business_context.trusted_profile.revision == "p1"
            return EvidenceBatch(query=request.query, parents=())

    dispatcher = build_subagent_dispatcher(
        retrieval=Retrieval(),
        evidence_builder=EvidenceBuilder(),
        snapshot=SNAPSHOT,
        concurrency=ConcurrencyManager(),
    )
    todos = [
        TodoItem(id=f"todo-{i}", title="星河公司住宿标准", owner="supervisor")
        for i in range(2)
    ]
    result = await dispatcher.delegate(
        todos,
        ResearchContext(scope=SCOPE, snapshot=SNAPSHOT, business_context=business),
        max_parallel=2,
    )
    assert len(result.results) == len(received) == 2 and profiles.calls == 1
    assert {c.task_intent_ref for c in received} == {"todo:todo-0", "todo:todo-1"}


async def test_child_projection_keeps_directory_objective_ancestors():
    import pytest
    from agentic_rag.runtime.query_composition import build_subagent_dispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager
    from agentic_rag.query.todos import TodoItem
    from agentic_rag.retrieval.query_compiler import BusinessScopeConflict
    from tests.unit.retrieval.test_query_compiler import NamedDirectory
    from tests.unit.query.test_graph import SCOPE, SNAPSHOT

    compiler = QueryCompiler(directory=NamedDirectory())
    root = await compiler.freeze("比较星河集团与远山集团标准", SCOPE)
    parent = compiler.for_task(root, "星河集团标准", task_ref="task:parent")
    plans = []
    inherited = []

    class Retrieval:
        async def retrieve(self, request, scope, snapshot, *, business_context):
            inherited.append(business_context)
            plans.append(
                await compiler.compile(request, scope, snapshot, business_context)
            )
            return EvidenceBatch(query=request.query, parents=())

    dispatcher = build_subagent_dispatcher(
        retrieval=Retrieval(),
        evidence_builder=EvidenceBuilder(),
        snapshot=SNAPSHOT,
        concurrency=ConcurrencyManager(),
    )
    ctx = ResearchContext(scope=SCOPE, snapshot=SNAPSHOT, business_context=parent)
    await dispatcher.delegate(
        [TodoItem(id="neutral", title="住宿条件", owner="supervisor")],
        ctx,
        queries={"neutral": "住宿标准"},
    )
    assert "远山集团" not in plans[0].dense_query
    assert [o.source_ref for o in inherited[0].task_objectives] == [
        "task:parent",
        "todo:neutral",
    ]
    with pytest.raises(BusinessScopeConflict, match="task target"):
        await dispatcher.delegate(
            [TodoItem(id="conflict", title="远山集团住宿条件", owner="supervisor")],
            ctx,
            queries={"conflict": "住宿标准"},
        )
    assert len(plans) == 1
    invalid_parent = compiler.for_task(
        parent, "远山集团标准", task_ref="task:invalid-ancestor"
    )
    with pytest.raises(BusinessScopeConflict, match="task target"):
        await dispatcher.delegate(
            [TodoItem(id="grandchild", title="住宿条件", owner="supervisor")],
            replace(ctx, business_context=invalid_parent),
            queries={"grandchild": "住宿标准"},
        )
    assert len(plans) == 1
