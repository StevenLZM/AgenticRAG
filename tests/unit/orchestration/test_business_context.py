"""Durable run-owned profile snapshots survive new worker processes."""

from agentic_rag.domain.models import UserScope
from agentic_rag.orchestration.store import CoordinationStore
from agentic_rag.orchestration.execution import ModelExecution
from agentic_rag.retrieval.query_compiler import QueryCompiler
from agentic_rag.retrieval.query_context import TrustedBusinessProfile
from agentic_rag.retrieval.models import RetrievalRequest
from tests.unit.orchestration.test_evidence import SNAPSHOT


async def test_run_freezes_profile_before_dispatch_and_worker_resume(store):
    class Profiles:
        revision = "p1"
        calls = 0

        async def get_profile(self, scope):
            self.calls += 1
            return TrustedBusinessProfile(
                owner_id=scope.user_id,
                revision=self.revision,
                companies=("星河公司" if self.revision == "p1" else "远山公司",),
            )

    profiles = Profiles()
    compiler = QueryCompiler(profiles=profiles)
    server = CoordinationStore(store.session_factory, query_compiler=compiler)
    await server.initialize(
        run_id="scope-run",
        user_id="user-1",
        question="我出差住宿标准",
        snapshot=SNAPSHOT,
    )
    profiles.revision = "p2"
    await server.initialize(
        run_id="scope-run",
        user_id="user-1",
        question="我出差住宿标准",
        snapshot=SNAPSHOT,
    )
    work = next(
        w for w in await server.list_ready("route") if w["run_id"] == "scope-run"
    )
    claim = await server.claim(work["id"], "worker2", 30)
    execution = ModelExecution(server, claim, None)
    plan = await compiler.compile(
        RetrievalRequest(query="上海住宿上限"),
        UserScope(user_id="user-1"),
        SNAPSHOT,
        execution.business_context,
    )
    assert "星河公司" in plan.dense_query and plan.profile_revision == "p1"
    assert profiles.calls == 1


async def test_retrieval_agent_propagates_assigned_scope_and_distinct_query_plans():
    from agentic_rag.orchestration.agents import RetrievalAgent
    from agentic_rag.orchestration.contracts import CandidateResult, TaskSpec
    from tests.unit.orchestration.test_agents import Execution
    from tests.unit.orchestration.test_contracts import task_payload
    from tests.unit.orchestration.test_evidence import ranked

    compiler = QueryCompiler()
    execution = Execution(
        {"queries": ["星河公司标准", "星河公司住宿条件"], "reason": "two details"},
        {"findings": [], "gaps": []},
    )
    execution.business_context = await compiler.freeze(
        "比较星河公司与远山公司标准", execution.scope
    )
    raw = task_payload()
    raw["objective"] = "星河公司标准"
    raw["budget"] = {"max_queries": 3, "max_rounds": 1}
    raw["sources"]["document_ids"] = ["d-p1"]

    class Retrieval:
        async def retrieve_candidates(
            self, request, scope, snapshot, *, business_context
        ):
            assert business_context.permitted_document_ids == ("d-p1",)
            plan = await compiler.compile(request, scope, snapshot, business_context)
            assert "远山公司" not in plan.dense_query
            return ranked().model_copy(
                update={
                    "query": request.query,
                    "query_plan": plan,
                    "query_plan_hash": plan.plan_hash,
                }
            )

    result = await RetrievalAgent(Retrieval()).execute(
        TaskSpec.model_validate(raw), {}, execution
    )
    restored = CandidateResult.model_validate_json(result.model_dump_json())
    assert len({b.query_plan_hash for b in restored.candidates}) == 2
    assert all(b.query_plan.task_intent_ref == "task:r1:1" for b in restored.candidates)


async def test_stage_executor_fast_retrieval_inherits_durable_business_context(store):
    from dataclasses import replace
    from agentic_rag.orchestration.execution import StageExecutor
    from tests.unit.orchestration.test_evidence import ranked

    compiler = QueryCompiler()
    server = CoordinationStore(store.session_factory, query_compiler=compiler)
    await server.initialize(
        run_id="fast-scope",
        user_id="user-1",
        question="星河公司标准",
        snapshot=SNAPSHOT,
    )
    work = next(
        w for w in await server.list_ready("route") if w["run_id"] == "fast-scope"
    )
    claim = await server.claim(work["id"], "worker", 30)

    class Retrieval:
        async def retrieve_candidates(
            self, request, scope, snapshot, *, business_context
        ):
            assert business_context.source_question == "星河公司标准"
            plan = await compiler.compile(request, scope, snapshot, business_context)
            return ranked().model_copy(
                update={"query_plan": plan, "query_plan_hash": plan.plan_hash}
            )

    result = await StageExecutor(server, None, retrieval=Retrieval())(
        replace(claim, stage="fast_retrieval")
    )
    assert result["candidates"][0]["query_plan"]["original_query"] == "星河公司标准"


async def test_evidence_preparation_forwards_each_frozen_plan_to_builder(tmp_path):
    from agentic_rag.orchestration.evidence import EvidencePreparer
    from agentic_rag.orchestration.contracts import Criterion
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from tests.unit.orchestration.test_evidence import (
        Hydrator,
        SCOPE,
        candidate,
        ranked,
    )

    compiler = QueryCompiler(artifacts=LocalArtifactStore(tmp_path))
    plan = await compiler.compile(
        RetrievalRequest(query="星河公司标准"), SCOPE, SNAPSHOT, None
    )
    ref = compiler.persist(plan)
    batch = ranked().model_copy(
        update={
            "query": plan.original_query,
            "query_plan": plan,
            "query_plan_hash": plan.plan_hash,
            "query_plan_ref": ref,
        }
    )

    class Builder:
        def build(self, batches, *args, **kwargs):
            assert batches[0].query_plan == plan
            assert batches[0].query_plan_hash == plan.plan_hash
            assert batches[0].query_plan_ref == ref
            return EvidenceBuilder().build(batches, *args, **kwargs)

    await EvidencePreparer(Hydrator(), builder=Builder()).prepare(
        [candidate("r1", "revenue", batch)],
        [Criterion(id="revenue", description="Revenue")],
        SCOPE,
        SNAPSHOT,
    )


async def test_agent_rejects_conflicting_objective_before_neutral_search():
    import pytest
    from agentic_rag.orchestration.agents import RetrievalAgent
    from agentic_rag.orchestration.contracts import TaskSpec
    from agentic_rag.retrieval.query_compiler import BusinessScopeConflict
    from tests.unit.orchestration.test_agents import Execution
    from tests.unit.orchestration.test_contracts import task_payload

    compiler = QueryCompiler()
    execution = Execution(
        {"queries": ["住宿标准"], "reason": "search policy"},
        {"findings": [], "gaps": []},
    )
    execution.business_context = await compiler.freeze("星河公司标准", execution.scope)
    raw = task_payload()
    raw["objective"] = "远山公司住宿标准"

    class Retrieval:
        calls = 0

        async def retrieve_candidates(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("conflicting task must not reach retrieval")

    retrieval = Retrieval()
    with pytest.raises(BusinessScopeConflict, match="task target"):
        await RetrievalAgent(retrieval).execute(
            TaskSpec.model_validate(raw), {}, execution
        )
    assert retrieval.calls == 0
