"""Trusted meaning is frozen independently from tenant authorization."""

import pytest

from agentic_rag.retrieval import query_compiler as module
from agentic_rag.retrieval.query_context import (
    BusinessScopeCoverage,
    QueryBusinessContext,
    TrustedBusinessProfile,
)
from agentic_rag.retrieval.models import QueryPlan, RetrievalRequest
from agentic_rag.persistence.artifacts import LocalArtifactStore
from tests.unit.retrieval.test_graph import SNAPSHOT, SCOPE


class Profiles:
    calls = 0
    value = TrustedBusinessProfile(
        owner_id=SCOPE.user_id,
        revision="p1",
        companies=("星河公司",),
        population="普通员工",
    )

    async def get_profile(self, scope):
        self.calls += 1
        return self.value


@pytest.mark.asyncio
async def test_profile_only_defaults_self_directed_and_freezes(tmp_path):
    profiles = Profiles()
    compiler = module.QueryCompiler(
        profiles=profiles, artifacts=LocalArtifactStore(tmp_path)
    )
    business = await compiler.freeze("我去上海出差住宿标准", SCOPE)
    profiles.value = profiles.value.model_copy(
        update={"revision": "p2", "companies": ("远山公司",)}
    )
    plan = await compiler.compile(
        RetrievalRequest(query=business.source_question), SCOPE, SNAPSHOT, business
    )
    assert "星河公司" in plan.dense_query and "普通员工" in plan.dense_query
    assert plan.lexical_core_query == business.source_question
    assert plan.lexical_scope_terms == ("星河公司", "普通员工")
    assert plan.profile_revision == "p1" and profiles.calls == 1
    assert (
        plan.rerank_query == plan.protected_query_context + "\n\n" + plan.original_query
    )
    saved = compiler.persist(plan)
    assert compiler.load(saved, scope=SCOPE, snapshot=SNAPSHOT) == plan
    assert QueryPlan.model_validate_json(plan.model_dump_json()) == plan
    inherited = QueryBusinessContext.model_validate_json(business.model_dump_json())
    later = await compiler.compile(
        RetrievalRequest(query="上海住宿费用上限"), SCOPE, SNAPSHOT, inherited
    )
    assert later.profile_revision == "p1" and "星河公司" in later.dense_query
    assert profiles.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["远山公司管理人员标准", "差旅制度一般怎么设计"])
async def test_explicit_other_company_or_general_never_uses_profile(question):
    plan = await module.QueryCompiler(profiles=Profiles()).compile(
        RetrievalRequest(query=question), SCOPE, SNAPSHOT, None
    )
    assert plan.dense_query == question and not plan.lexical_scope_terms
    assert "星河公司" not in plan.rerank_query
    assert plan.effective_filter.user_id == SCOPE.user_id


@pytest.mark.asyncio
async def test_missing_ambiguous_profile_and_task_conflict():
    compiler = module.QueryCompiler()
    plan = await compiler.compile(
        RetrievalRequest(query="我们公司标准"), SCOPE, SNAPSHOT, None
    )
    assert plan.resolution_status == "unresolved"
    profiles = Profiles()
    profiles.value = profiles.value.model_copy(
        update={"companies": ("星河公司", "远山公司")}
    )
    plan = await module.QueryCompiler(profiles=profiles).compile(
        RetrievalRequest(query="我们公司标准"), SCOPE, SNAPSHOT, None
    )
    assert (
        plan.resolution_status == "ambiguous"
        and plan.dense_query == plan.original_query
    )
    business = await compiler.freeze("星河公司普通员工标准", SCOPE)
    with pytest.raises(module.BusinessScopeConflict):
        await compiler.compile(
            RetrievalRequest(query="远山公司标准"), SCOPE, SNAPSHOT, business
        )


class Coverage:
    def __init__(self, certificate, current):
        self.certificate, self.current = certificate, current

    async def coverage(self, scope, snapshot):
        return self.certificate

    async def current_binding(self, scope, snapshot):
        return self.current


def certificate(**changes):
    return BusinessScopeCoverage(
        owner_id=SCOPE.user_id,
        corpus_snapshot="corpus1",
        index_generation=SNAPSHOT.index_generation,
        version_set_hash="a" * 64,
        mapping_revision="m1",
        issuer="directory1",
        rule_version="1",
        complete=True,
        document_companies=(("doc-x", "星河公司"), ("doc-y", "远山公司")),
        **changes,
    )


@pytest.mark.asyncio
async def test_coverage_exact_binding_and_empty_intersection():
    cert = certificate()
    binding = cert.binding
    request = RetrievalRequest(query="星河公司标准", document_ids=("doc-y",))
    compiler = module.QueryCompiler(coverage=Coverage(cert, binding))
    plan = await compiler.compile(request, SCOPE, SNAPSHOT, None)
    assert plan.effective_filter.match_none and plan.business_filter == ("doc-x",)
    assert plan.authorization_filter.document_ids == ("doc-y",)
    for bad in (
        cert.model_copy(update={"complete": False}),
        cert.model_copy(update={"mapping_revision": "old"}),
    ):
        plan = await module.QueryCompiler(coverage=Coverage(bad, binding)).compile(
            request, SCOPE, SNAPSHOT, None
        )
        assert plan.business_filter is None and plan.effective_filter.document_ids == (
            "doc-y",
        )
    empty = cert.model_copy(update={"document_companies": (("doc-y", "远山公司"),)})
    plan = await module.QueryCompiler(coverage=Coverage(empty, binding)).compile(
        RetrievalRequest(query="星河公司标准"), SCOPE, SNAPSHOT, None
    )
    assert plan.effective_filter.match_none


@pytest.mark.asyncio
async def test_explicit_long_query_preserves_scope_without_dense_bm25_duplication():
    question = "星河公司普通员工" + "上海住宿政策条件" * 80
    plan = await module.QueryCompiler().compile(
        RetrievalRequest(query=question), SCOPE, SNAPSHOT, None
    )
    assert plan.dense_query == plan.lexical_core_query == question
    assert plan.lexical_scope_terms == ()
    assert plan.protected_query_context == "" and not plan.allow_query_windows
    assert plan.effective_business_scope[0].source_span == (0, 4)


@pytest.mark.asyncio
async def test_retrieval_both_entry_points_compile_once_and_persist(tmp_path):
    from agentic_rag.retrieval.graph import RetrievalDependencies, RetrievalService
    from tests.unit.retrieval.test_graph import FakeVector, FakeParentFetcher, hit
    from agentic_rag.retrieval.reranker import RerankResult

    seen = {}

    class Embedding:
        async def embed_query(self, text):
            seen["dense"] = text
            return [0.0] * 1024

    class Lexical:
        async def search(self, text, filter, top_k, *, scope_terms=()):
            seen["lexical"] = (text, scope_terms)
            return [hit("lexical", lane="bm25")]

    class Rerank:
        async def rerank(
            self,
            text,
            hits,
            *,
            limit,
            protected_query_context="",
            allow_query_windows=True,
        ):
            seen["rerank"] = (text, protected_query_context, allow_query_windows)
            return RerankResult(hits=hits, degraded=False, model_version="test")

    profiles = Profiles()
    compiler = module.QueryCompiler(
        profiles=profiles, artifacts=LocalArtifactStore(tmp_path)
    )
    service = RetrievalService(
        RetrievalDependencies(
            embedding=Embedding(),
            vector=FakeVector([hit("dense")]),
            lexical=Lexical(),
            reranker=Rerank(),
            parent_fetcher=FakeParentFetcher(),
            query_compiler=compiler,
        )
    )
    business = await compiler.freeze("我去上海出差住宿标准", SCOPE)
    for call in (service.retrieve, service.retrieve_candidates):
        result = await call(
            RetrievalRequest(query=business.source_question),
            SCOPE,
            SNAPSHOT,
            business_context=business,
        )
        assert (
            result.query_plan_ref
            and result.query_plan_hash == result.query_plan.plan_hash
        )
        assert seen["dense"] == result.query_plan.dense_query
        assert seen["lexical"] == (business.source_question, ("星河公司", "普通员工"))
        assert seen["rerank"][:2] == (
            result.query_plan.rerank_query,
            result.query_plan.protected_query_context,
        )
    assert profiles.calls == 1


@pytest.mark.asyncio
async def test_profile_defaults_do_not_erase_explicit_population_and_tools_cannot_replace_defaults():
    compiler = module.QueryCompiler(profiles=Profiles())
    business = await compiler.freeze("我出差，管理人员住宿标准", SCOPE)
    plan = await compiler.compile(
        RetrievalRequest(query=business.source_question), SCOPE, SNAPSHOT, business
    )
    assert "星河公司" in plan.dense_query and "普通员工" not in plan.dense_query
    assert not plan.allow_query_windows
    own = await compiler.freeze("我出差住宿标准", SCOPE)
    plan = await compiler.compile(
        RetrievalRequest(query="星河公司普通员工上海住宿标准"), SCOPE, SNAPSHOT, own
    )
    assert plan.dense_query == plan.original_query
    with pytest.raises(module.BusinessScopeConflict):
        await compiler.compile(
            RetrievalRequest(query="远山公司住宿标准"), SCOPE, SNAPSHOT, own
        )


@pytest.mark.asyncio
async def test_comparison_queries_can_select_only_assigned_target():
    compiler = module.QueryCompiler()
    business = await compiler.freeze("比较星河公司与远山公司标准", SCOPE)
    assert {r.value for r in business.source_scope} == {"星河公司", "远山公司"}
    plan = await compiler.compile(
        RetrievalRequest(query="远山公司标准"), SCOPE, SNAPSHOT, business
    )
    assert "星河公司" not in plan.dense_query
    task = compiler.for_task(
        business, "星河公司标准", task_ref="task:1", document_ids=("doc-x",)
    )
    with pytest.raises(module.BusinessScopeConflict):
        await compiler.compile(
            RetrievalRequest(query="远山公司标准"), SCOPE, SNAPSHOT, task
        )
    assert task.task_intent_ref == "task:1" and task.permitted_document_ids == (
        "doc-x",
    )


def test_snapshot_opt_in_preserves_legacy_hash_and_plan_nested_filters_are_frozen():
    from pydantic import ValidationError

    old = SNAPSHOT.snapshot_id
    assert (
        SNAPSHOT.model_copy(update={"query_context_version": None}).snapshot_id == old
    )
    assert (
        SNAPSHOT.model_copy(update={"query_context_version": "business-v1"}).snapshot_id
        != old
    )
    from agentic_rag.retrieval.models import DateRange

    value = DateRange()
    with pytest.raises(ValidationError):
        value.start = __import__("datetime").date(2026, 1, 1)


@pytest.mark.asyncio
async def test_unknown_shorthand_and_ambiguous_directory_never_invent_entities():
    from agentic_rag.retrieval.query_context import ScopeResolution

    plan = await module.QueryCompiler().compile(
        RetrievalRequest(query="比较星河与远山两家公司标准"), SCOPE, SNAPSHOT, None
    )
    assert (
        plan.resolution_status == "unresolved"
        and plan.dense_query == plan.original_query
    )

    class Directory:
        async def resolve_mentions(self, text):
            return (
                ScopeResolution(
                    field="company",
                    value="星河",
                    source_kind="explicit_user",
                    source_ref="directory:1",
                    source_span=(0, 2),
                    resolution_status="ambiguous",
                ),
            )

    plan = await module.QueryCompiler(
        directory=Directory(), profiles=Profiles()
    ).compile(RetrievalRequest(query="星河标准"), SCOPE, SNAPSHOT, None)
    assert plan.resolution_status == "ambiguous" and plan.business_filter is None
    assert plan.dense_query == "星河标准"


@pytest.mark.asyncio
async def test_task_profile_projection_preserves_explicit_population():
    compiler = module.QueryCompiler(profiles=Profiles())
    business = await compiler.freeze("我出差，管理人员标准", SCOPE)
    task = compiler.for_task(business, "星河公司住宿标准", task_ref="task:1")
    plan = await compiler.compile(
        RetrievalRequest(query="住宿标准"), SCOPE, SNAPSHOT, task
    )
    assert "星河公司" in plan.dense_query and "管理人员" in plan.dense_query


@pytest.mark.asyncio
async def test_compiler_preserves_process_owned_authorization_builder():
    from agentic_rag.retrieval.filters import FilterBuilder

    class Authorization(FilterBuilder):
        def build(self, request, scope, snapshot):
            return (
                super()
                .build(request, scope, snapshot)
                .model_copy(update={"document_ids": ("permitted",)})
            )

    compiler = module.QueryCompiler(
        filter_builder=Authorization(),
        coverage=Coverage(certificate(), certificate().binding),
    )
    plan = await compiler.compile(
        RetrievalRequest(query="星河公司标准"), SCOPE, SNAPSHOT, None
    )
    assert plan.authorization_filter.document_ids == ("permitted",)
    assert plan.effective_filter.match_none


@pytest.mark.asyncio
async def test_empty_task_bound_never_becomes_unrestricted_even_with_coverage():
    compiler = module.QueryCompiler(
        coverage=Coverage(certificate(), certificate().binding)
    )
    business = await compiler.freeze("星河公司标准", SCOPE)
    business = business.model_copy(update={"permitted_document_ids": ()})
    plan = await compiler.compile(
        RetrievalRequest(query="星河公司标准"), SCOPE, SNAPSHOT, business
    )
    assert plan.effective_filter.match_none


@pytest.mark.asyncio
async def test_request_and_task_selectors_intersect_server_authorization():
    from agentic_rag.retrieval.filters import FilterBuilder

    class Authorization(FilterBuilder):
        def build(self, request, scope, snapshot):
            return (
                super()
                .build(request, scope, snapshot)
                .model_copy(update={"document_ids": ("allowed",)})
            )

    compiler = module.QueryCompiler(filter_builder=Authorization())
    business = await compiler.freeze("住宿标准", SCOPE)
    business = business.model_copy(
        update={"permitted_document_ids": ("allowed", "outside")}
    )
    plan = await compiler.compile(
        RetrievalRequest(query="住宿标准", document_ids=("outside",)),
        SCOPE,
        SNAPSHOT,
        business,
    )
    assert plan.effective_filter.match_none


@pytest.mark.asyncio
async def test_serialized_plan_cannot_change_scope_or_drop_protected_qualifiers():
    from pydantic import ValidationError

    plan = await module.QueryCompiler(profiles=Profiles()).compile(
        RetrievalRequest(query="我出差住宿标准"), SCOPE, SNAPSHOT, None
    )
    for edit in (
        {"rerank_query": "住宿标准"},
        {
            "effective_filter": {
                **plan.effective_filter.model_dump(),
                "user_id": "other",
            }
        },
        {"lexical_core_query": "公司"},
    ):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate({**plan.model_dump(), **edit})


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [False, True])
async def test_confirmed_company_survives_explicit_population_per_field(profile):
    from agentic_rag.retrieval.query_context import ScopeResolution

    compiler = module.QueryCompiler(
        profiles=Profiles() if profile else None,
        coverage=Coverage(certificate(), certificate().binding),
    )
    business = await compiler.freeze("我出差，管理人员住宿标准", SCOPE)
    business = business.model_copy(
        update={
            "confirmed_reference": (
                ScopeResolution(
                    field="company",
                    value="远山公司",
                    source_kind="confirmed_reference",
                    source_ref="confirmed:1",
                ),
                ScopeResolution(
                    field="population",
                    value="普通员工",
                    source_kind="confirmed_reference",
                    source_ref="confirmed:1",
                ),
            )
        }
    )
    plan = await compiler.compile(
        RetrievalRequest(query=business.source_question), SCOPE, SNAPSHOT, business
    )
    assert {r.field: r.value for r in plan.effective_business_scope} == {
        "company": "远山公司",
        "population": "管理人员",
    }
    assert "星河公司" not in plan.dense_query and "普通员工" not in plan.dense_query
    assert plan.business_filter == ("doc-y",)
    assert (
        next(
            r for r in plan.effective_business_scope if r.field == "company"
        ).source_ref
        == "confirmed:1"
    )


@pytest.mark.asyncio
async def test_task_objective_cannot_reassign_before_neutral_query():
    compiler = module.QueryCompiler()
    business = await compiler.freeze("星河公司标准", SCOPE)
    with pytest.raises(module.BusinessScopeConflict, match="task target"):
        task = compiler.for_task(business, "远山公司住宿标准", task_ref="task:conflict")
        await compiler.compile(
            RetrievalRequest(query="住宿标准"), SCOPE, SNAPSHOT, task
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unresolved", "ambiguous"])
async def test_uncertain_explicit_scope_disables_windows_and_stays_unasserted(status):
    from agentic_rag.retrieval.query_context import ScopeResolution
    from pydantic import ValidationError

    class Directory:
        async def resolve_mentions(self, text):
            return (
                ScopeResolution(
                    field="company",
                    value="星河",
                    source_kind="explicit_user",
                    source_ref="directory:uncertain",
                    source_span=(0, 2),
                    resolution_status=status,
                ),
            )

    question = "星河" + "住宿政策条件" * 100
    plan = await module.QueryCompiler(
        directory=Directory(), coverage=Coverage(certificate(), certificate().binding)
    ).compile(RetrievalRequest(query=question), SCOPE, SNAPSHOT, None)
    assert not plan.allow_query_windows
    assert plan.resolution_status == status
    assert plan.dense_query == plan.rerank_query == question
    assert (
        plan.protected_query_context == ""
        and plan.lexical_scope_terms == ()
        and plan.business_filter is None
    )
    with pytest.raises(ValidationError, match="unwindowed"):
        QueryPlan.model_validate({**plan.model_dump(), "allow_query_windows": True})


@pytest.mark.asyncio
async def test_field_precedence_keeps_confirmed_company_under_task_population():
    from agentic_rag.retrieval.query_context import ScopeResolution

    compiler = module.QueryCompiler(profiles=Profiles())
    business = await compiler.freeze("我出差住宿标准", SCOPE)
    business = business.model_copy(
        update={
            "confirmed_reference": (
                ScopeResolution(
                    field="company",
                    value="远山公司",
                    source_kind="confirmed_reference",
                    source_ref="confirmed:1",
                ),
            ),
            "task_targets": (
                ScopeResolution(
                    field="population",
                    value="管理人员",
                    source_kind="task_target",
                    source_ref="task:1",
                ),
            ),
        }
    )
    plan = await compiler.compile(
        RetrievalRequest(query=business.source_question), SCOPE, SNAPSHOT, business
    )
    assert {r.field: r.value for r in plan.effective_business_scope} == {
        "company": "远山公司",
        "population": "管理人员",
    }
    assert "星河公司" not in plan.dense_query and "普通员工" not in plan.dense_query


@pytest.mark.asyncio
async def test_self_directed_confirmed_own_company_defaults_only_missing_population():
    from agentic_rag.retrieval.query_context import ScopeResolution

    compiler = module.QueryCompiler(profiles=Profiles())
    business = await compiler.freeze("我出差住宿标准", SCOPE)
    business = business.model_copy(
        update={
            "confirmed_reference": (
                ScopeResolution(
                    field="company",
                    value="星河公司",
                    source_kind="confirmed_reference",
                    source_ref="confirmed:1",
                ),
            )
        }
    )
    plan = await compiler.compile(
        RetrievalRequest(query=business.source_question), SCOPE, SNAPSHOT, business
    )
    assert {
        r.field: (r.value, r.source_kind) for r in plan.effective_business_scope
    } == {
        "company": ("星河公司", "confirmed_reference"),
        "population": ("普通员工", "trusted_profile"),
    }
