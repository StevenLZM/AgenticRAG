"""Original context dependencies remain explicit across packing and publication."""

import pytest
from agentic_rag.models.contextual import (
    ContextArtifact,
    ContextClaim,
    ContextScopeRegion,
)
from tests.unit.ingestion.test_context_service import setup
from tests.unit.query.test_evidence_builder import SCOPE, SNAPSHOT, child


def test_all_accepted_claims_including_unrendered_exceptions_are_requirements(tmp_path):
    from agentic_rag.retrieval.context_evidence import EvidenceRequirementResolver
    from agentic_rag.ingestion.contextualization.sources import SourceCatalog

    artifacts, ast, ref, parent, _ = setup(
        tmp_path, ("本制度适用于星河公司，不含外包人员。", "上海差旅标准500一晚。")
    )
    catalog = SourceCatalog.from_ast(ast, ref.sha256)
    refs = [
        catalog.resolve_quote(s, catalog.source_text(s)) for s in catalog.source_ids
    ]
    claim = ContextClaim(
        id="claim-0",
        field="company",
        value="星河公司",
        display_value="星河公司",
        scope_region=ContextScopeRegion(kind="document", source_refs=(refs[1],)),
        source_refs=(refs[0],),
        derivation="llm",
        status="accepted",
        verification_method="llm_supported",
        verification_status="supported",
        allowed_uses=("search",),
        exclusion_conditions=("不含外包人员",),
    )
    artifact = ContextArtifact(
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        canonical_ast_sha256=ref.sha256,
        source_child_id="original",
        build_fingerprint="b" * 64,
        claims=(claim,),
        rendered_claim_ids=(),
    )
    hit = child("p", content="上海差旅标准500一晚。")
    requirements = EvidenceRequirementResolver().resolve(None, hit, artifact)
    assert len(requirements.entries) == 1
    requirement = requirements.entries[0]
    assert requirement.source_refs == (refs[0],)
    assert requirement.exclusion_conditions == ("不含外包人员",)
    assert requirement.claim_ids != ("claim-0",)
    assert requirement.semantic_status == "pending"


async def built(
    tmp_path,
    *,
    same_parent=False,
    structural=False,
    texts=None,
    two_children=False,
    prefix_omit=False,
    all_support=False,
    split_parents=False,
    preserve_parent=False,
):
    from agentic_rag.ingestion.contextualization.service import (
        ContextualizationService,
        ContextBuildConfig,
    )
    from agentic_rag.ingestion.contextualization.store import MemoryContextBuildStore
    from agentic_rag.ingestion.contextualization.gateway import (
        ExtractionResponse,
        CandidateClaim,
        Quote,
        VerificationResponse,
    )
    from agentic_rag.ingestion.manifest import VersionManifest, child_identity_digest
    from agentic_rag.models.contextual import ContextArtifactRef
    from tests.unit.ingestion.test_context_service import config, fence
    from tests.unit.ingestion.test_context_assembly import isolated_target
    from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
    from agentic_rag.retrieval.adapters.elasticsearch import ElasticsearchBm25Index
    from tests.unit.persistence.test_elasticsearch_staging import (
        _Client,
        _Response,
        _context,
        _embedded_child,
    )
    from dataclasses import replace

    texts = texts or (
        "本制度适用于星河公司。",
        "本章仅适用于普通员工，不含外包人员。",
        "上海差旅标准500一晚。",
    )
    artifacts, ast, ref, parent, claim = setup(tmp_path, texts)
    if not same_parent:
        full_parent = parent
        parent, _ = isolated_target(parent, len(texts) - 1, texts[-1])
        if preserve_parent:
            child = parent.children[0]
            offset = len(full_parent.content) - len(child.content)
            span = child.ast_locator.spans[0].model_copy(
                update={
                    "parent_char_from": offset,
                    "parent_char_to": offset + len(child.content),
                }
            )
            locator = child.ast_locator.model_copy(
                update={
                    "spans": (span,),
                    "parent_char_from": offset,
                    "parent_char_to": offset + len(child.content),
                }
            )
            parent = full_parent.model_copy(
                update={
                    "children": (child.model_copy(update={"ast_locator": locator}),)
                }
            )

    if two_children:
        parent = parent.model_copy(
            update={
                "children": (
                    *parent.children,
                    parent.children[0].model_copy(
                        update={"id": parent.children[0].id + "-second"}
                    ),
                )
            }
        )

    parents = [parent]
    if split_parents:
        from tests.unit.ingestion.test_context_sources import _child
        from agentic_rag.ingestion.chunker import ParentChunk

        first_child = _child(*texts[:-1]).model_copy(update={"id": "first-child"})
        first = ParentChunk(
            **{
                **first_child.model_dump(
                    exclude={"parent_id", "parent_ordinal", "search_representation"}
                ),
                "id": first_child.parent_id,
                "children": (first_child,),
            }
        )
        second_child = parent.children[0].model_copy(
            update={"id": "second-child", "parent_id": "parent-2"}
        )
        second = parent.model_copy(
            update={"id": "parent-2", "children": (second_child,)}
        )
        parents = [first, second]

    class Model:
        async def extract(self, input):
            claims = []
            for field, value in [("company", "星河公司"), ("population", "普通员工")]:
                source = next(s for s in input.sources if value in s.text)
                claims.append(
                    CandidateClaim(
                        field=field,
                        value=value,
                        scope_region_id="document",
                        source_quotes=(
                            Quote(source_id=source.source_id, quote=source.text),
                        ),
                        exclusion_conditions=("不含外包人员",)
                        if field == "population" or all_support
                        else (),
                    )
                )
            return ExtractionResponse(
                target_source_child_id=input.target_source_child_id,
                claims=tuple(claims),
            )

        async def verify(self, input):
            return VerificationResponse(
                status="supported",
                source_ids=tuple(
                    s.source_id for s in input.context.sources if s.text in texts[:-1]
                )
                if all_support
                else tuple(q.source_id for q in input.candidate.source_quotes),
            )

    build = ContextBuildConfig(enabled=True) if structural else config()
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=None if structural else Model(),
        count_tokens=(lambda text: len(text) * 2) if prefix_omit else len,
        count_model_tokens=lambda text, _: len(text),
    ).enrich(
        ast,
        parents,
        canonical_ref=ref,
        build=build,
        claim=claim,
        before_side_effect=fence,
    )
    manifest = VersionManifest(
        schema_version=2,
        canonical_ast_sha256=ref.sha256,
        parent_count=len(result.parents),
        child_count=sum(len(p.children) for p in result.parents),
        embedding_model="test",
        embedding_dimensions=1024,
        index_generation=SNAPSHOT.index_generation,
        context_artifact_ref=result.context_artifact_ref,
        enriched_chunks_ref=result.enriched_chunks_ref,
        build_fingerprint=result.build_fingerprint,
        representation_version="context-v1",
        child_id_hash_digest=child_identity_digest(
            [(c.id, c.content_hash) for p in result.parents for c in p.children]
        ),
        enrichment_counts={
            k: sum(
                c.search_representation.enrichment_status == k
                for p in result.parents
                for c in p.children
            )
            for k in ("enriched", "partial", "structural_only")
        },
    )
    mref = artifacts.put_json("manifest.json", manifest.payload())

    class Authority:
        revoked = False

        async def manifest(
            self, doc, version, scope, snapshot, *, build_fingerprint=None
        ):
            if self.revoked or (
                doc,
                version,
                scope.user_id,
                snapshot.index_generation,
            ) != ("document-1", "version-1", SCOPE.user_id, SNAPSHOT.index_generation):
                raise ValueError("source revoked or stale")
            return ContextArtifactRef(**mref.__dict__)

    authority = Authority()
    es = ElasticsearchBm25Index(
        _Client(),
        index_generation=SNAPSHOT.index_generation,
        expected_representation_version="context-v1",
    )
    hits = []
    for final_child in (c for p in result.parents for c in p.children):
        embedded = replace(_embedded_child(), chunk=final_child)
        document = ElasticsearchChildIndexStore._document(_context(), embedded)
        hits.extend(
            es._hits(
                _Response({"hits": {"hits": [{"_source": document, "_score": 1.0}]}}),
                lane="bm25",
            )
        )
    from agentic_rag.retrieval.reranker import Reranker

    class Encoder:
        def predict(self, pairs):
            assert all(text == h.search_content for (_, text), h in zip(pairs, hits))
            return [1.0] * len(pairs)

    reranker = Reranker(Encoder(), model_version="offline")
    try:
        ranked = await reranker.rerank("上海差旅标准", hits)
        hits = ranked.hits
    finally:
        reranker.close()
    hit = hits[0]
    from agentic_rag.retrieval.context_evidence import ContextSourceHydrator

    hydrator = ContextSourceHydrator(artifacts=artifacts, authority=authority)
    from agentic_rag.retrieval.models import ParentEvidence

    if split_parents:
        selected = tuple(
            ParentEvidence(
                parent_id=p.id,
                document_id=p.document_id,
                document_version_id=p.document_version_id,
                content=p.content,
                ast_locator=p.ast_locator.model_dump_json(),
                heading_path=p.heading_path,
                child_hits=tuple(h for h in hits if h.parent_id == p.id),
            )
            for p in result.parents
        )
        return hydrator, authority, hits, selected
    parent_evidence = ParentEvidence(
        parent_id=parent.id,
        document_id=parent.document_id,
        document_version_id=parent.document_version_id,
        content=parent.content,
        ast_locator=parent.ast_locator.model_dump_json(),
        child_hits=tuple(hits),
    )
    return hydrator, authority, hit, parent_evidence


async def plan():
    from agentic_rag.retrieval.query_compiler import QueryCompiler
    from agentic_rag.retrieval.models import RetrievalRequest

    return await QueryCompiler().compile(
        RetrievalRequest(query="星河公司普通员工上海标准？"), SCOPE, SNAPSHOT, None
    )


async def pack(hydrator, hit, parent, *, budget=12000, query_plan=None):
    from agentic_rag.query.evidence_builder import (
        EvidenceBuilder,
        EvidenceCoverageTarget,
    )
    from agentic_rag.retrieval.models import EvidenceBatch

    supporting = await hydrator.hydrate([hit], SCOPE, SNAPSHOT, query_plan)
    batch = EvidenceBatch(
        query="q",
        parents=(parent,),
        supporting_context=supporting,
        query_plan=query_plan,
        target_ids=("amount",),
    )
    return EvidenceBuilder().build(
        [batch],
        [EvidenceCoverageTarget(target_id="amount", description="applicable amount")],
        SCOPE,
        SNAPSHOT,
        max_tokens=budget,
    )


@pytest.mark.asyncio
async def test_off_parent_original_sources_pack_cite_and_public_pages(tmp_path):
    from agentic_rag.query.audit import (
        CitationValidator,
        ParentRepositoryAuthorizationResolver,
    )
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment
    from agentic_rag.query.answer_sources import build_answer_sources, AnswerSources
    from tests.unit.query.test_answer_sources import answer
    from agentic_rag.query.evidence_builder import PackedEvidence
    from types import SimpleNamespace

    hydrator, authority, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent, query_plan=await plan())
    assert len(packed.items) == 3
    assert len(packed.evidence_requirements) == 2
    assert not packed.scope_evidence_missing
    excerpts = [i for i in packed.items if i.source_kind == "canonical_excerpt"]
    assert {i.source_ref.page_from for i in excerpts} == {1, 2}
    assert all(i.parent_id is None for i in excerpts)
    assert PackedEvidence.model_validate_json(packed.model_dump_json()) == packed

    class Parents:
        async def get_many(self, ids, scope):
            assert ids == [parent.parent_id]
            return [
                SimpleNamespace(
                    id=parent.parent_id,
                    user_id=SCOPE.user_id,
                    status="active",
                    index_generation=SNAPSHOT.index_generation,
                    document_id=parent.document_id,
                    document_version_id=parent.document_version_id,
                    ast_locator=parent.ast_locator,
                    heading_path=(),
                )
            ]

    resolver = ParentRepositoryAuthorizationResolver(
        Parents(), context_sources=hydrator
    )
    draft = AnswerDraft(
        segments=(
            AnswerSegment(
                kind="content",
                text="普通员工标准500，不含外包人员",
                evidence_ids=tuple(packed.manifest),
            ),
        )
    )
    result = await CitationValidator().validate_async(
        draft, packed, SCOPE, SNAPSHOT, {}, authorization_resolver=resolver
    )
    assert result.passed, result.reasons
    public = build_answer_sources(
        answer(tuple(packed.manifest)),
        packed,
        run_id="r",
        snapshot_id=SNAPSHOT.snapshot_id,
    )
    assert public.schema_version == 2
    assert [
        i.page_from for i in public.items if i.source_kind == "canonical_excerpt"
    ] == [1, 2]
    assert AnswerSources.model_validate_json(public.model_dump_json()) == public
    authority.revoked = True
    assert not (
        await CitationValidator().validate_async(
            draft, packed, SCOPE, SNAPSHOT, {}, authorization_resolver=resolver
        )
    ).passed


@pytest.mark.asyncio
async def test_missing_or_over_budget_support_cannot_leave_naked_amount(tmp_path):
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.retrieval.models import EvidenceBatch

    hydrator, _, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent, budget=500)
    assert not packed.items and packed.scope_evidence_missing
    support = await hydrator.hydrate([hit], SCOPE, SNAPSHOT, await plan())
    broken = support.model_copy(update={"sources": support.sources[:1]})
    packed = EvidenceBuilder().build(
        [EvidenceBatch(query="q", parents=(parent,), supporting_context=broken)],
        (),
        SCOPE,
        SNAPSHOT,
    )
    assert not packed.items and packed.scope_evidence_missing


@pytest.mark.asyncio
async def test_same_parent_exact_source_coverage_deduplicates(tmp_path):
    hydrator, _, hit, parent = await built(tmp_path, same_parent=True)
    packed = await pack(hydrator, hit, parent)
    assert len(packed.items) == 1
    assert len(packed.items[0].supports_claim_ids) == 2
    assert len(packed.supporting_context.sources) == 2
    assert not packed.scope_evidence_missing


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["hash", "ref", "version", "owner", "generation"])
async def test_forged_cached_hit_or_stale_authority_fails(tmp_path, mutation):
    from agentic_rag.domain.models import UserScope

    hydrator, _, hit, _ = await built(tmp_path)
    scope, snapshot = SCOPE, SNAPSHOT
    if mutation == "hash":
        hit = hit.model_copy(update={"content_hash": "f" * 64})
    elif mutation == "ref":
        hit = hit.model_copy(
            update={
                "context_ref": hit.context_ref.model_copy(update={"sha256": "f" * 64})
            }
        )
    elif mutation == "version":
        hit = hit.model_copy(update={"document_version_id": "v2"})
    elif mutation == "owner":
        scope = UserScope(user_id="u2")
    else:
        snapshot = SNAPSHOT.model_copy(update={"index_generation": "old"})
    with pytest.raises(ValueError):
        await hydrator.hydrate([hit], scope, snapshot, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "本制度适用于星河公司普通员工，上海标准500一晚。",
        "星河公司普通员工是客户案例，上海标准500一晚。",
    ],
)
async def test_structural_literal_is_only_semantic_pending_not_verified(tmp_path, text):
    hydrator, _, hit, parent = await built(tmp_path, structural=True, texts=(text,))
    packed = await pack(hydrator, hit, parent, query_plan=await plan())
    assert packed.items
    assert all(r.semantic_status == "pending" for r in packed.evidence_requirements)
    assert all(
        not r.claim_ids and r.reason == "original_body_requires_semantic_review"
        for r in packed.evidence_requirements
    )


@pytest.mark.asyncio
async def test_business_legacy_without_scope_original_is_missing():
    from agentic_rag.retrieval.context_evidence import EvidenceRequirementResolver

    requirements = EvidenceRequirementResolver().resolve(
        await plan(), child("p", content="上海500一晚"), None
    )
    assert requirements.entries and all(
        r.status == "missing" for r in requirements.entries
    )


@pytest.mark.asyncio
async def test_known_counterevidence_cannot_be_overridden_by_literal_fallback(tmp_path):
    from agentic_rag.retrieval.context_evidence import EvidenceRequirementResolver

    hydrator, _, hit, _ = await built(
        tmp_path,
        structural=True,
        texts=("星河公司普通员工是客户案例，上海标准500一晚。",),
    )
    artifact = ContextArtifact.model_validate(
        hydrator.artifacts.read_json(hit.context_ref)
    )
    rejected = ContextClaim(
        id="claim-0",
        field="company",
        value="星河公司",
        display_value="星河公司",
        scope_region=ContextScopeRegion(kind="document"),
        derivation="llm",
        status="rejected",
        verification_method="llm_supported",
        verification_status="contradicted",
        rejection_reason="客户案例不代表适用公司",
    )
    artifact = artifact.model_copy(update={"claims": (rejected,)})
    req = EvidenceRequirementResolver().resolve(await plan(), hit, artifact)
    company = next(r for r in req.entries if r.field == "company")
    assert company.status == "conflicting"
    assert "客户案例" in company.reason


@pytest.mark.asyncio
async def test_sql_authority_binds_active_pointer_deletion_generation_and_hash(
    tmp_path,
):
    from sqlalchemy import update
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from agentic_rag.persistence.repositories import (
        metadata,
        documents,
        document_versions,
    )
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from agentic_rag.retrieval.context_evidence import SqlContextSourceAuthority
    from tests.unit.ingestion.test_worker_store import _seed

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/auth.db")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await _seed(factory, job_status="queued", version_status="active")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    ref = artifacts.put_json("manifest.json", {"immutable": True})
    authority = SqlContextSourceAuthority(factory, artifacts)
    async with factory.begin() as db:
        await db.execute(
            update(documents).values(status="active", active_version_id="version-1")
        )
        await db.execute(
            update(document_versions).values(
                index_generation=SNAPSHOT.index_generation,
                manifest_path=ref.uri,
                manifest_hash=ref.sha256,
            )
        )
    assert (
        await authority.manifest("doc-1", "version-1", SCOPE, SNAPSHOT)
    ).sha256 == ref.sha256
    for table, change, restore in [
        (documents, {"deletion_status": "pending"}, {"deletion_status": None}),
        (documents, {"active_version_id": None}, {"active_version_id": "version-1"}),
        (documents, {"user_id": "u2"}, {"user_id": SCOPE.user_id}),
        (document_versions, {"status": "inactive"}, {"status": "active"}),
        (
            document_versions,
            {"index_generation": "old"},
            {"index_generation": SNAPSHOT.index_generation},
        ),
        (document_versions, {"manifest_hash": "f" * 64}, {"manifest_hash": ref.sha256}),
    ]:
        async with factory.begin() as db:
            await db.execute(update(table).values(**change))
        with pytest.raises(ValueError):
            await authority.manifest("doc-1", "version-1", SCOPE, SNAPSHOT)
        async with factory.begin() as db:
            await db.execute(update(table).values(**restore))
    await engine.dispose()


@pytest.mark.asyncio
async def test_supervisor_keeps_distinct_query_plan_requirements_and_hash_identity(
    tmp_path,
):
    from agentic_rag.orchestration.evidence import EvidencePreparer
    from agentic_rag.orchestration.contracts import (
        Criterion,
        CandidateResult,
        PreparedBundle,
    )
    from agentic_rag.retrieval.models import RankedCandidates, RetrievalRequest
    from agentic_rag.retrieval.query_compiler import QueryCompiler

    hydrator, _, hit, parent = await built(tmp_path)
    first = await plan()
    second = await QueryCompiler().compile(
        RetrievalRequest(query="远山公司普通员工上海标准？"), SCOPE, SNAPSHOT, None
    )

    def candidate(key, query_plan, hit=hit):
        return CandidateResult(
            origin_key=key,
            criterion_ids=("amount",),
            candidates=(
                RankedCandidates(
                    query=query_plan.original_query,
                    query_plan=query_plan,
                    query_plan_hash=query_plan.plan_hash,
                    hits=(hit,),
                    user_id=SCOPE.user_id,
                    snapshot_id=SNAPSHOT.snapshot_id,
                    index_generation=SNAPSHOT.index_generation,
                ),
            ),
        )

    class Parents:
        async def hydrate(self, selected, scope):
            return [parent]

    preparer = EvidencePreparer(Parents(), context_sources=hydrator)
    bundle = await preparer.prepare(
        [candidate("a", first), candidate("b", second)],
        [Criterion(id="amount", description="amount")],
        SCOPE,
        SNAPSHOT,
    )
    assert {r.query_plan_hash for r in bundle.packed.evidence_requirements} == {
        first.plan_hash,
        second.plan_hash,
    }
    assert bundle.packed.scope_evidence_missing and not bundle.packed.items
    assert PreparedBundle.model_validate_json(bundle.model_dump_json()) == bundle
    with pytest.raises(ValueError, match="conflicting candidate identity"):
        await preparer.prepare(
            [
                candidate("a", first),
                candidate(
                    "b",
                    first,
                    hit.model_copy(
                        update={
                            "context_ref": hit.context_ref.model_copy(
                                update={"sha256": "f" * 64}
                            )
                        }
                    ),
                ),
            ],
            [Criterion(id="amount", description="amount")],
            SCOPE,
            SNAPSHOT,
        )


@pytest.mark.asyncio
async def test_reducer_preserves_scope_requirements_and_sources(tmp_path):
    from agentic_rag.query.subagents import EvidenceReducer
    from types import SimpleNamespace

    hydrator, _, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent, query_plan=await plan())
    result = EvidenceReducer.merge([SimpleNamespace(todo_id="a", evidence=packed)])
    assert result.evidence_requirements == packed.evidence_requirements
    assert result.supporting_context == packed.supporting_context


@pytest.mark.asyncio
async def test_cross_child_local_claim_ids_remain_distinct_when_sources_deduplicate(
    tmp_path,
):
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.retrieval.models import EvidenceBatch

    hydrator, _, _, parent = await built(tmp_path, two_children=True)
    support = await hydrator.hydrate(parent.child_hits, SCOPE, SNAPSHOT, await plan())
    assert len(support.requirements) == 4
    assert len({cid for r in support.requirements for cid in r.claim_ids}) == 4
    assert len(support.sources) == 2
    assert all(
        len(s.context_refs) == len(s.source_child_ids) == len(s.supports_claim_ids) == 2
        for s in support.sources
    )
    packed = EvidenceBuilder().build(
        [EvidenceBatch(query="q", parents=(parent,), supporting_context=support)],
        (),
        SCOPE,
        SNAPSHOT,
    )
    assert len(packed.items) == 3
    for entry in packed.manifest.values():
        if entry.source_kind == "canonical_excerpt":
            authorized = await hydrator.authorize(entry, SCOPE, SNAPSHOT)
            assert authorized["is_active"]


@pytest.mark.asyncio
async def test_citation_must_include_required_scope_sources(tmp_path):
    from agentic_rag.query.audit import CitationValidator
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment

    hydrator, _, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent, query_plan=await plan())
    authorization = {}
    for eid, entry in packed.manifest.items():
        authorization[eid] = (
            (await hydrator.authorize(entry, SCOPE, SNAPSHOT))
            if entry.source_kind == "canonical_excerpt"
            else dict(
                user_id=SCOPE.user_id,
                index_generation=SNAPSHOT.index_generation,
                is_active=True,
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                document_version_id=parent.document_version_id,
                ast_locator=parent.ast_locator,
                heading_path=(),
            )
        )
    body = packed.items[0]
    draft = AnswerDraft(
        segments=(
            AnswerSegment(
                kind="content",
                text="星河公司普通员工500",
                evidence_ids=(body.evidence_id,),
            ),
        )
    )
    result = CitationValidator().validate(draft, packed, SCOPE, SNAPSHOT, authorization)
    assert not result.passed and "missing_scope_citation" in result.reasons


@pytest.mark.asyncio
async def test_grader_receives_original_pending_scope_and_cannot_waive_missing(
    tmp_path,
):
    from agentic_rag.query.audit import EvidenceGrader
    from tests.unit.query.test_audit import FakeGateway
    import json

    hydrator, _, hit, parent = await built(
        tmp_path,
        structural=True,
        texts=("星河公司普通员工是客户案例，上海标准500一晚。",),
    )
    packed = await pack(hydrator, hit, parent, query_plan=await plan())
    gateway = FakeGateway(
        [{"decision": "insufficient", "gaps": ["客户案例不证明制度适用"]}]
    )
    grade = await EvidenceGrader(gateway).grade(
        "适用标准", packed, scope=SCOPE, snapshot=SNAPSHOT
    )
    assert grade.decision == "insufficient"
    payload = json.loads(gateway.calls[0].messages[-1]["content"])
    assert "客户案例" in payload["packed_context"]
    assert {r["field"] for r in payload["evidence_requirements"]} == {
        "company",
        "population",
    }
    assert all(
        r["semantic_status"] == "pending" for r in payload["evidence_requirements"]
    )
    gateway = FakeGateway([{"decision": "sufficient", "gaps": []}])
    packed = packed.model_copy(update={"scope_evidence_missing": ("required-missing",)})
    assert (
        await EvidenceGrader(gateway).grade("q", packed, scope=SCOPE, snapshot=SNAPSHOT)
    ).decision == "insufficient"


@pytest.mark.asyncio
async def test_unrendered_accepted_population_and_exception_still_pack(tmp_path):
    hydrator, _, hit, parent = await built(tmp_path, prefix_omit=True)
    artifact = ContextArtifact.model_validate(
        hydrator.artifacts.read_json(hit.context_ref)
    )
    assert len(artifact.claims) == 2 and len(artifact.rendered_claim_ids) == 1
    packed = await pack(hydrator, hit, parent, query_plan=await plan())
    assert len(packed.items) == 3
    assert any(
        r.field == "population" and r.exclusion_conditions == ("不含外包人员",)
        for r in packed.evidence_requirements
    )
    assert "不含外包人员" in packed.rendered_context


@pytest.mark.asyncio
async def test_same_parent_crop_cannot_remove_required_support(tmp_path):
    hydrator, _, hit, parent = await built(tmp_path)
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.retrieval.models import EvidenceBatch

    support = await hydrator.hydrate([hit], SCOPE, SNAPSHOT, await plan())
    # Legitimate full-parent source coordinates cover the support, but the
    # packing budget can retain only the body at the end of a large Parent.
    import json

    locator = json.loads(parent.ast_locator)
    locator["spans"] = [
        dict(
            block_id=s.source_ref.block_id,
            canonical_path=s.source_ref.canonical_path,
            char_from=s.source_ref.char_from,
            char_to=s.source_ref.char_to,
        )
        for s in support.sources
    ]
    parent = parent.model_copy(
        update={
            "content": "\n".join(s.content for s in support.sources)
            + "x" * 2000
            + hit.content,
            "ast_locator": json.dumps(locator),
        }
    )
    packed = EvidenceBuilder().build(
        [EvidenceBatch(query="q", parents=(parent,), supporting_context=support)],
        (),
        SCOPE,
        SNAPSHOT,
        max_tokens=400,
    )
    assert not packed.items and packed.scope_evidence_missing


@pytest.mark.asyncio
async def test_cross_child_reducer_unions_support_links_without_identity_loss(tmp_path):
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.query.subagents import EvidenceReducer
    from agentic_rag.retrieval.models import EvidenceBatch
    from types import SimpleNamespace

    hydrator, _, _, parent = await built(tmp_path, two_children=True)
    packs = []
    for hit in parent.child_hits:
        support = await hydrator.hydrate([hit], SCOPE, SNAPSHOT, await plan())
        selected_parent = parent.model_copy(update={"child_hits": (hit,)})
        packs.append(
            EvidenceBuilder().build(
                [
                    EvidenceBatch(
                        query="q",
                        parents=(selected_parent,),
                        supporting_context=support,
                    )
                ],
                (),
                SCOPE,
                SNAPSHOT,
            )
        )
    merged = EvidenceReducer.merge(
        [SimpleNamespace(todo_id=str(i), evidence=p) for i, p in enumerate(packs)]
    )
    assert len(merged.items) == 3
    assert len(merged.evidence_requirements) == 4
    assert all(len(s.source_child_ids) == 2 for s in merged.supporting_context.sources)
    for entry in merged.manifest.values():
        if entry.source_kind == "canonical_excerpt":
            assert len(entry.supports_claim_ids) == len(entry.context_refs) == 2
            await hydrator.authorize(entry, SCOPE, SNAPSHOT)


@pytest.mark.asyncio
async def test_supervisor_source_snapshot_uses_typed_excerpt_and_original_pages(
    tmp_path,
):
    from agentic_rag.orchestration.streaming import source_snapshot

    hydrator, _, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent)
    frozen = dict(
        evidence=[i.model_dump(mode="json") for i in packed.items],
        mode="full",
        assessments=[],
        analyses=[],
        limitations=[],
    )
    run = dict(
        id="r",
        data=dict(
            snapshot=SNAPSHOT.model_dump(mode="json"),
            answer=dict(
                input=frozen,
                epoch=1,
                seq=2,
                text=" ".join(f"[cite:{eid}]" for eid in packed.manifest),
                status="completed",
                faithfulness="passed",
            ),
        ),
    )
    source = source_snapshot(run)
    assert source["schema_version"] == 2
    assert {
        i["page_from"]
        for i in source["items"]
        if i["source_kind"] == "canonical_excerpt"
    } == {1, 2}


@pytest.mark.asyncio
async def test_supervisor_disclosure_requires_context_authority_even_without_legacy_sql_check(
    tmp_path,
):
    from agentic_rag.orchestration.authorization import (
        authorize_sources,
        SourceAccessRevoked,
    )

    hydrator, _, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent)
    run = dict(
        user_id=SCOPE.user_id,
        data=dict(
            enforce_source_access=False, snapshot=SNAPSHOT.model_dump(mode="json")
        ),
    )
    with pytest.raises(SourceAccessRevoked):
        await authorize_sources(
            None, run, [i.model_dump(mode="json") for i in packed.items]
        )


def test_final_answer_budget_keeps_context_support_with_analysis_body():
    from agentic_rag.orchestration.finalization import bound_inputs

    evidence = [
        {"evidence_id": "body", "content": "500"},
        {"evidence_id": "scope", "content": "x" * 2000},
    ]
    analysis = {
        "task_key": "a",
        "result": {
            "findings": [],
            "calculations": [
                {"operands": [{"evidence_ref": "body", "source_kind": "evidence"}]}
            ],
        },
    }
    selected, analyses, _, gaps = bound_inputs(
        evidence,
        [analysis],
        [],
        {},
        budget=1000,
        mode="partial",
        support_links={"body": ["scope"]},
    )
    assert not selected and not analyses and gaps


@pytest.mark.asyncio
async def test_public_source_snapshot_preserves_private_authorization_provenance(
    tmp_path,
):
    from agentic_rag.query.answer_sources import build_answer_sources
    from tests.unit.query.test_answer_sources import answer

    hydrator, _, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent)
    sources = build_answer_sources(
        answer(tuple(packed.manifest)),
        packed,
        run_id="r",
        snapshot_id=SNAPSHOT.snapshot_id,
    )
    for source in sources.items:
        assert source.source_provenance == packed.manifest[source.evidence_id]


@pytest.mark.asyncio
async def test_full_retrieval_entry_uses_shared_original_source_service(tmp_path):
    from agentic_rag.retrieval.graph import RetrievalDependencies, RetrievalService
    from tests.unit.retrieval.test_graph import (
        FakeEmbedding,
        FakeVector,
        FakeLexical,
        FakeReranker,
        REQUEST,
    )

    hydrator, _, hit, parent = await built(tmp_path)

    class Parents:
        async def hydrate(self, selected, scope):
            return [parent]

    service = RetrievalService(
        RetrievalDependencies(
            embedding=FakeEmbedding(),
            vector=FakeVector([hit]),
            lexical=FakeLexical([]),
            reranker=FakeReranker(),
            parent_fetcher=Parents(),
            context_sources=hydrator,
        )
    )
    batch = await service.retrieve(REQUEST, SCOPE, SNAPSHOT)
    assert len(batch.supporting_context.sources) == 2
    assert len(batch.supporting_context.requirements) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "full",
        "truncated",
        "heading",
        "page",
        "flags",
        "missing_provenance",
        "missing_version_and_provenance",
        "version_mismatch",
        "old_version_marker",
        "valid_prefix",
    ],
)
@pytest.mark.parametrize("source_mode", ["contextual", "legacy_v2"])
async def test_chat_sources_rechecks_private_excerpt_authority_on_each_read(
    tmp_path, tamper, source_mode
):
    from sqlalchemy import insert, update
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from agentic_rag.persistence.repositories import (
        metadata,
        documents,
        document_versions,
        agent_runs,
    )
    from agentic_rag.runtime.chat_sources import ChatSourceService
    from tests.fixtures.chat_sessions import chat_service
    from tests.unit.query.test_answer_sources import answer
    from agentic_rag.query.answer_sources import build_answer_sources
    from datetime import UTC, datetime
    from uuid import uuid4

    if source_mode == "legacy_v2":
        from agentic_rag.retrieval.models import ChildHit, EvidenceBatch
        from agentic_rag.query.evidence_builder import EvidenceBuilder

        hydrator, authority, hit, parent = await built(
            tmp_path / "artifacts",
            structural=True,
            texts=("星河公司普通员工上海标准500一晚。",),
        )
        legacy = ChildHit(
            **{
                name: getattr(hit, name)
                for name in (
                    "child_id",
                    "parent_id",
                    "user_id",
                    "document_id",
                    "document_version_id",
                    "content",
                    "ast_locator",
                    "lane",
                    "lane_rank",
                    "retrieval_score",
                )
            }
        )
        parent = parent.model_copy(update={"child_hits": (legacy,)})
        packed = EvidenceBuilder().build(
            [EvidenceBatch(query="q", query_plan=await plan(), parents=(parent,))],
            (),
            SCOPE,
            SNAPSHOT,
        )
        assert packed.items and not packed.items[0].context_refs
    else:
        hydrator, authority, hit, parent = await built(tmp_path / "artifacts")
        packed = await pack(hydrator, hit, parent)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/chat.db")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    chats = chat_service(factory)
    session, _ = await chats.create(SCOPE, str(uuid4()))
    run = (
        await chats.submit(
            SCOPE, session.id, "问", SNAPSHOT, client_request_id=str(uuid4())
        )
    ).run
    public = answer(tuple(packed.manifest))
    sources = build_answer_sources(
        public, packed, run_id=run.id, snapshot_id=SNAPSHOT.snapshot_id
    )
    now = datetime.now(UTC)
    async with factory.begin() as db:
        await db.execute(
            insert(documents).values(
                id=parent.document_id,
                user_id=SCOPE.user_id,
                filename="policy.pdf",
                source_type="upload",
                mime_type="application/pdf",
                content_hash="a" * 64,
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        await db.execute(
            insert(document_versions).values(
                id=parent.document_version_id,
                document_id=parent.document_id,
                version_no=1,
                parser_version="p",
                pipeline_version="p",
                embedding_version="e",
                index_generation=SNAPSHOT.index_generation,
                status="active",
                created_at=now,
            )
        )
        await db.execute(
            update(documents).values(active_version_id=parent.document_version_id)
        )
        if source_mode == "legacy_v2":
            from agentic_rag.persistence.repositories import parent_chunks
            from agentic_rag.models.contextual import text_sha256

            await db.execute(
                insert(parent_chunks).values(
                    id=parent.parent_id,
                    user_id=SCOPE.user_id,
                    document_id=parent.document_id,
                    document_version_id=parent.document_version_id,
                    ordinal=0,
                    heading_path=list(parent.heading_path),
                    content_type="text",
                    content=parent.content,
                    ast_locator=parent.ast_locator,
                    content_hash=text_sha256(parent.content),
                    status="active",
                )
            )
            from agentic_rag.orchestration.authorization import (
                authorize_sources,
                SourceAccessRevoked,
            )

            proof_run = dict(
                user_id=SCOPE.user_id,
                data=dict(
                    enforce_source_access=False,
                    snapshot=SNAPSHOT.model_dump(mode="json"),
                ),
            )
            await authorize_sources(
                db, proof_run, [i.model_dump(mode="json") for i in packed.items]
            )
            bad = packed.items[0].model_dump(mode="json")
            bad.update(
                content="forged",
                content_hash=text_sha256("forged"),
                parent_excerpt_length=6,
            )
            with pytest.raises(SourceAccessRevoked):
                await authorize_sources(db, proof_run, [bad])
        await db.execute(
            update(agent_runs)
            .where(agent_runs.c.id == run.id)
            .values(
                status="completed",
                active_slot=None,
                answer=public.model_dump(mode="json"),
                answer_sources=sources.model_dump(mode="json"),
            )
        )
    reader = ChatSourceService(factory, context_sources=hydrator)
    view = await reader.get(SCOPE, session.id, run.id)
    assert view.status == "available"
    assert {
        s.page_from for s in view.items if s.source_kind == "canonical_excerpt"
    } == ({1, 2} if source_mode == "contextual" else set())
    assert all("source_provenance" not in s.model_dump() for s in view.items)
    if tamper:
        changed = sources.model_dump(mode="json")
        for item in changed["items"]:
            if tamper in {"full", "truncated"}:
                item.update(
                    excerpt="伪造：本公司99999一晚。", truncated=tamper == "truncated"
                )
            elif tamper == "missing_provenance":
                item["source_provenance"] = None
            elif tamper == "missing_version_and_provenance":
                item.pop("evidence_schema_version", None)
                item["source_provenance"] = None
            elif tamper == "version_mismatch":
                item["evidence_schema_version"] = 1
            elif tamper == "old_version_marker":
                item.pop("evidence_schema_version", None)
            elif tamper == "heading":
                item.update(heading_path=["伪造适用范围"], heading_truncated=True)
            elif tamper == "page":
                item.update(page_from=99, page_to=99)
            elif tamper == "flags":
                item.update(truncated=True, excerpt_omitted=True)
            else:
                item.update(excerpt=item["excerpt"][:3], truncated=True)
        async with factory.begin() as db:
            await db.execute(
                update(agent_runs)
                .where(agent_runs.c.id == run.id)
                .values(answer_sources=changed)
            )
        changed_view = await reader.get(SCOPE, session.id, run.id)
        assert changed_view.status == (
            "available"
            if tamper in {"valid_prefix", "old_version_marker"}
            else "unavailable"
        )
    authority.revoked = True
    if source_mode == "legacy_v2":
        async with factory.begin() as db:
            await db.execute(update(documents).values(deletion_status="pending"))
    view = await reader.get(SCOPE, session.id, run.id)
    assert view.status == "unavailable" and not view.items
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unresolved", "ambiguous"])
async def test_global_unresolved_self_scope_never_adopts_company_from_candidate(
    tmp_path, status
):
    from agentic_rag.retrieval.query_compiler import QueryCompiler
    from agentic_rag.retrieval.models import RetrievalRequest
    from agentic_rag.query.audit import EvidenceGrader
    from tests.unit.query.test_audit import FakeGateway

    hydrator, _, hit, parent = await built(tmp_path)
    query_plan = await QueryCompiler().compile(
        RetrievalRequest(query="我们公司上海差旅标准？"), SCOPE, SNAPSHOT, None
    )
    assert (
        query_plan.resolution_status == "unresolved"
        and not query_plan.effective_business_scope
    )
    query_plan = query_plan.model_copy(update={"resolution_status": status})
    packed = await pack(hydrator, hit, parent, query_plan=query_plan)
    assert packed.scope_evidence_missing and not packed.items
    assert any(
        r.field == "query_scope" and r.status != "resolved"
        for r in packed.evidence_requirements
    )
    grade = await EvidenceGrader(FakeGateway([])).grade(
        "我们公司标准", packed, scope=SCOPE, snapshot=SNAPSHOT
    )
    assert grade.decision != "sufficient" and any(
        "query_scope" in gap for gap in grade.gaps
    )


@pytest.mark.asyncio
async def test_research_checkpoint_merge_accepts_distinct_support_relations(tmp_path):
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.query.research_loop import _validate_packed_inputs
    from agentic_rag.retrieval.models import EvidenceBatch

    hydrator, _, _, parent = await built(tmp_path, two_children=True)
    values = []
    for hit in parent.child_hits:
        support = await hydrator.hydrate([hit], SCOPE, SNAPSHOT, await plan())
        batch = EvidenceBatch(
            query="q",
            parents=(parent.model_copy(update={"child_hits": (hit,)}),),
            supporting_context=support,
        )
        packed = EvidenceBuilder().build([batch], (), SCOPE, SNAPSHOT)
        values.append((packed, (batch,)))
    _validate_packed_inputs(tuple(values), SNAPSHOT)


@pytest.mark.asyncio
async def test_plan_ref_and_hash_cannot_disagree_with_frozen_meaning():
    from agentic_rag.retrieval.models import EvidenceBatch, RankedCandidates
    from agentic_rag.models.contextual import ContextArtifactRef

    query_plan = await plan()
    for data in [
        dict(query_plan_hash="f" * 64),
        dict(
            query_plan_ref=ContextArtifactRef(
                uri="artifact://plan.json", sha256="f" * 64, size_bytes=1
            )
        ),
    ]:
        with pytest.raises(ValueError, match="plan"):
            EvidenceBatch(query="q", parents=(), query_plan=query_plan, **data)
        with pytest.raises(ValueError, match="plan"):
            RankedCandidates(
                query="q",
                hits=(),
                user_id=SCOPE.user_id,
                snapshot_id=SNAPSHOT.snapshot_id,
                index_generation=SNAPSHOT.index_generation,
                query_plan=query_plan,
                **data,
            )


@pytest.mark.asyncio
async def test_supporting_source_checkpoint_rejects_content_hash_forgery(tmp_path):
    from agentic_rag.models.contextual import SupportingContext

    hydrator, _, hit, _ = await built(tmp_path)
    supporting = await hydrator.hydrate([hit], SCOPE, SNAPSHOT, await plan())
    payload = supporting.model_dump(mode="json")
    payload["sources"][0]["content"] = "forged scope"
    with pytest.raises(ValueError, match="hash"):
        SupportingContext.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [4, 5])
async def test_complete_verifier_closure_obeys_four_source_cap(tmp_path, count):
    texts = (
        "本制度适用于星河公司。",
        "本章适用于普通员工，不含外包人员。",
        *(f"术语定义{i}：适用范围按本制度。" for i in range(count - 2)),
        "上海标准500一晚。",
    )
    hydrator, _, hit, parent = await built(tmp_path, texts=texts, all_support=True)
    support = await hydrator.hydrate([hit], SCOPE, SNAPSHOT, await plan())
    assert len(support.sources) == count
    packed = await pack(hydrator, hit, parent, query_plan=await plan())
    if count == 4:
        assert len(packed.items) == 5 and not packed.scope_evidence_missing
    else:
        assert not packed.items and packed.scope_evidence_missing


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["body", "heading", "locator"])
async def test_reauthorization_binds_parent_body_and_metadata(tmp_path, damage):
    from agentic_rag.query.evidence_builder import (
        EvidenceItem,
        EvidenceManifestEntry,
        evidence_projection,
    )
    from agentic_rag.models.contextual import text_sha256
    from agentic_rag.orchestration.authorization import (
        authorize_sources,
        SourceAccessRevoked,
    )

    hydrator, _, hit, parent = await built(tmp_path)
    packed = await pack(hydrator, hit, parent)
    body = packed.items[0]
    changes = (
        {
            "content": "上海标准99999一晚。",
            "content_hash": text_sha256("上海标准99999一晚。"),
            "parent_excerpt_length": len("上海标准99999一晚。"),
        }
        if damage == "body"
        else {"heading_path": ("伪造章节",)}
        if damage == "heading"
        else {"ast_locator": "forged-locator"}
    )
    forged = EvidenceItem(**{**body.model_dump(), **changes})
    entry = EvidenceManifestEntry(**evidence_projection(forged))
    with pytest.raises(ValueError):
        await hydrator.authorize_parent(entry, SCOPE, SNAPSHOT)
    run = dict(
        user_id=SCOPE.user_id,
        data=dict(
            enforce_source_access=False, snapshot=SNAPSHOT.model_dump(mode="json")
        ),
    )
    with pytest.raises(SourceAccessRevoked):
        await authorize_sources(
            None, run, [forged.model_dump(mode="json")], context_sources=hydrator
        )
    from agentic_rag.query.audit import (
        CitationValidator,
        ParentRepositoryAuthorizationResolver,
    )
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment
    from types import SimpleNamespace

    class Parents:
        async def get_many(self, ids, scope):
            return [
                SimpleNamespace(
                    id=parent.parent_id,
                    user_id=SCOPE.user_id,
                    status="active",
                    index_generation=SNAPSHOT.index_generation,
                    document_id=parent.document_id,
                    document_version_id=parent.document_version_id,
                    ast_locator=parent.ast_locator,
                    heading_path=parent.heading_path,
                )
            ]

    forged_pack = packed.model_copy(
        update={
            "items": (forged, *packed.items[1:]),
            "manifest": {**packed.manifest, body.evidence_id: entry},
        }
    )
    draft = AnswerDraft(
        segments=(
            AnswerSegment(
                kind="content", text="99999", evidence_ids=tuple(packed.manifest)
            ),
        )
    )
    validation = await CitationValidator().validate_async(
        draft,
        forged_pack,
        SCOPE,
        SNAPSHOT,
        {},
        authorization_resolver=ParentRepositoryAuthorizationResolver(
            Parents(), context_sources=hydrator
        ),
    )
    assert not validation.passed


@pytest.mark.asyncio
async def test_two_real_parents_share_source_without_foreign_local_relations(tmp_path):
    from agentic_rag.query.evidence_builder import EvidenceBuilder, support_links
    from agentic_rag.retrieval.models import EvidenceBatch

    hydrator, _, hits, parents = await built(tmp_path, split_parents=True)
    supporting = await hydrator.hydrate(hits, SCOPE, SNAPSHOT, await plan())
    packed = EvidenceBuilder().build(
        [EvidenceBatch(query="q", parents=parents, supporting_context=supporting)],
        (),
        SCOPE,
        SNAPSHOT,
    )
    assert len([i for i in packed.items if i.source_kind == "parent_chunk"]) == 2
    for entry in packed.manifest.values():
        if entry.source_kind == "parent_chunk":
            await hydrator.authorize_parent(entry, SCOPE, SNAPSHOT)
        else:
            await hydrator.authorize(entry, SCOPE, SNAPSHOT)
    links = support_links(packed)
    second = next(i for i in packed.items if i.parent_id == "parent-2")
    assert links[second.evidence_id]
    assert all(
        packed.manifest[eid].source_kind == "canonical_excerpt"
        for eid in links[second.evidence_id]
    )
    from agentic_rag.orchestration.authorization import authorize_sources

    run = dict(
        user_id=SCOPE.user_id,
        data=dict(
            enforce_source_access=False, snapshot=SNAPSHOT.model_dump(mode="json")
        ),
    )
    await authorize_sources(
        None,
        run,
        [i.model_dump(mode="json") for i in packed.items],
        context_sources=hydrator,
    )
    small = EvidenceBuilder().build(
        [EvidenceBatch(query="q", parents=parents, supporting_context=supporting)],
        (),
        SCOPE,
        SNAPSHOT,
        max_tokens=600,
    )
    assert all(i.parent_id != "parent-2" for i in small.items)
    assert small.scope_evidence_missing


@pytest.mark.asyncio
async def test_authoritative_parent_reconstruction_accepts_legal_crop_and_rejects_modified_crop(
    tmp_path,
):
    from agentic_rag.models.contextual import text_sha256
    from agentic_rag.query.evidence_builder import (
        EvidenceItem,
        EvidenceManifestEntry,
        evidence_projection,
    )
    from agentic_rag.query.answer_sources import (
        build_answer_sources,
        source_snapshot_matches_verified,
    )
    from tests.unit.query.test_answer_sources import answer

    hydrator, _, hit, parent = await built(
        tmp_path,
        structural=True,
        preserve_parent=True,
        texts=("背景原文。" * 200, "上海标准500一晚。"),
    )
    packed = await pack(hydrator, hit, parent, budget=400)
    assert len(packed.items) == 1
    item = packed.items[0]
    assert hit.content in item.content and len(item.content) < len(parent.content)
    authorized = await hydrator.authorize_parent(
        packed.manifest[item.evidence_id], SCOPE, SNAPSHOT
    )
    assert authorized["verified_content"] == item.content
    public = build_answer_sources(
        answer((item.evidence_id,)),
        packed,
        run_id="r",
        snapshot_id=SNAPSHOT.snapshot_id,
    )
    assert source_snapshot_matches_verified(public.items[0], authorized)
    bad_content = item.content.replace("500", "999")
    bad = EvidenceItem(
        **{
            **item.model_dump(),
            "content": bad_content,
            "content_hash": text_sha256(bad_content),
        }
    )
    with pytest.raises(ValueError, match="original content"):
        await hydrator.authorize_parent(
            EvidenceManifestEntry(**evidence_projection(bad)), SCOPE, SNAPSHOT
        )


@pytest.mark.asyncio
async def test_repeated_child_text_crop_uses_actual_authorized_locator(tmp_path):
    hydrator, _, hit, parent = await built(
        tmp_path,
        structural=True,
        preserve_parent=True,
        texts=("上海标准500一晚。", "背景。" * 250, "上海标准500一晚。"),
    )
    packed = await pack(hydrator, hit, parent, budget=400)
    assert packed.items and packed.items[0].content.startswith("…")
    assert packed.items[0].content.endswith(hit.content)
    verified = await hydrator.authorize_parent(
        packed.manifest[packed.items[0].evidence_id], SCOPE, SNAPSHOT
    )
    assert verified["verified_content"] == packed.items[0].content


@pytest.mark.asyncio
async def test_schema2_legacy_parent_original_bytes_are_reauthorized_without_context_artifact(
    tmp_path,
):
    from agentic_rag.query.audit import ParentRepositoryAuthorizationResolver
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.retrieval.models import EvidenceBatch, ChildHit
    from agentic_rag.models.contextual import text_sha256
    from types import SimpleNamespace

    _, _, contextual_hit, parent = await built(
        tmp_path, structural=True, texts=("星河公司普通员工上海标准500一晚。",)
    )
    legacy = ChildHit(
        **{
            name: getattr(contextual_hit, name)
            for name in (
                "child_id",
                "parent_id",
                "user_id",
                "document_id",
                "document_version_id",
                "content",
                "ast_locator",
                "lane",
                "lane_rank",
                "retrieval_score",
            )
        }
    )
    packed = EvidenceBuilder().build(
        [
            EvidenceBatch(
                query="q",
                query_plan=await plan(),
                parents=(parent.model_copy(update={"child_hits": (legacy,)}),),
            )
        ],
        (),
        SCOPE,
        SNAPSHOT,
    )
    item = packed.items[0]
    assert item.schema_version == 2 and not item.context_refs

    class Parents:
        async def get_many(self, ids, scope):
            return [
                SimpleNamespace(
                    id=parent.parent_id,
                    user_id=SCOPE.user_id,
                    document_id=parent.document_id,
                    document_version_id=parent.document_version_id,
                    status="active",
                    content=parent.content,
                    ast_locator=parent.ast_locator,
                    heading_path=parent.heading_path,
                    index_generation=SNAPSHOT.index_generation,
                )
            ]

    resolver = ParentRepositoryAuthorizationResolver(Parents())
    verified = await resolver.resolve(packed.manifest, SCOPE, SNAPSHOT)
    assert verified[item.evidence_id]["verified_content"] == item.content
    entry = packed.manifest[item.evidence_id].model_copy(
        update={"content_hash": text_sha256("forged")}
    )
    with pytest.raises(ValueError):
        await resolver.resolve({item.evidence_id: entry}, SCOPE, SNAPSHOT)


@pytest.mark.asyncio
async def test_mixed_legacy_and_contextual_sources_survive_reducer_and_public_read(
    tmp_path, monkeypatch
):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from agentic_rag.domain.models import RunStatus
    from agentic_rag.query.answer_sources import build_answer_sources
    from agentic_rag.query.evidence_builder import EvidenceBuilder
    from agentic_rag.query.subagents import EvidenceReducer
    from agentic_rag.retrieval.models import ChildHit, EvidenceBatch
    from agentic_rag.runtime import chat_sources
    from tests.unit.query.test_answer_sources import answer

    hydrator, authority, hits, parents = await built(tmp_path, split_parents=True)
    contextual = await pack(hydrator, hits[0], parents[0])
    legacy_hit = ChildHit(
        **{
            name: getattr(hits[1], name)
            for name in (
                "child_id",
                "parent_id",
                "user_id",
                "document_id",
                "document_version_id",
                "content",
                "ast_locator",
                "lane",
                "lane_rank",
                "retrieval_score",
            )
        }
    )
    legacy_parent = parents[1].model_copy(update={"child_hits": (legacy_hit,)})
    legacy = EvidenceBuilder().build(
        [EvidenceBatch(query="q", parents=(legacy_parent,))],
        (),
        SCOPE,
        SNAPSHOT,
    )
    merged = EvidenceReducer.merge(
        [
            SimpleNamespace(todo_id="context", evidence=contextual),
            SimpleNamespace(todo_id="legacy", evidence=legacy),
        ],
        expected_index_generation=SNAPSHOT.index_generation,
    )
    assert {item.schema_version for item in merged.items} == {1, 2}
    public = answer(tuple(merged.manifest))
    sources = build_answer_sources(
        public, merged, run_id="run", snapshot_id=SNAPSHOT.snapshot_id
    )
    assert sources.schema_version == 2
    assert {s.evidence_id: s.evidence_schema_version for s in sources.items} == {
        item.evidence_id: item.schema_version for item in merged.items
    }
    run = SimpleNamespace(
        id="run",
        thread_id="session",
        status=RunStatus.COMPLETED,
        runtime_config_snapshot=SNAPSHOT.model_dump(mode="json"),
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
        answer=public.model_dump(mode="json"),
        answer_sources=sources.model_dump(mode="json"),
    )

    class Sessions:
        def __init__(self, db):
            pass

        async def get_owned(self, scope, session_id):
            assert scope == SCOPE and session_id == "session"
            return object()

        async def source_documents(self, scope, identities):
            assert scope == SCOPE
            return {
                key: SimpleNamespace(filename="policy.pdf", active_version_id=key[1])
                for key in identities
            }

    class Runs:
        def __init__(self, db):
            pass

        async def get(self, run_id, scope):
            assert run_id == "run" and scope == SCOPE
            return run

    @asynccontextmanager
    async def session_factory():
        yield object()

    monkeypatch.setattr(chat_sources, "SqlAlchemyChatSessionRepository", Sessions)
    monkeypatch.setattr(chat_sources, "SqlAlchemyRunRepository", Runs)
    reader = chat_sources.ChatSourceService(session_factory, context_sources=hydrator)
    view = await reader.get(SCOPE, "session", "run")
    assert view.status == "available" and view.omitted_source_count == 0
    assert tuple(s.evidence_id for s in view.items) == tuple(merged.manifest)
    assert {s.evidence_id: s.excerpt for s in view.items} == {
        s.evidence_id: s.excerpt for s in sources.items
    }
    assert all("evidence_schema_version" not in s.model_dump() for s in view.items)
    # Revoking context authority must still remove only the affected new source.
    authority.revoked = True
    revoked = await reader.get(SCOPE, "session", "run")
    assert revoked.status == "partial"
    assert tuple(s.evidence_id for s in revoked.items) == tuple(legacy.manifest)
    # Genuine historical schema1 snapshots predate both private fields.
    historical_answer = answer(tuple(legacy.manifest))
    historical = build_answer_sources(
        historical_answer, legacy, run_id="run", snapshot_id=SNAPSHOT.snapshot_id
    ).model_dump(mode="json")
    assert historical["schema_version"] == 1
    for source in historical["items"]:
        source.pop("evidence_schema_version")
        source.pop("source_provenance")
    run.answer = historical_answer.model_dump(mode="json")
    run.answer_sources = historical
    old_view = await reader.get(SCOPE, "session", "run")
    assert old_view.status == "available" and old_view.omitted_source_count == 0
    assert tuple(s.evidence_id for s in old_view.items) == tuple(legacy.manifest)
