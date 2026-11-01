"""Offline grounded extraction and durable retry contracts."""

import pytest
from agentic_rag.ingestion.contextualization.service import (
    ContextualizationService,
    ContextBuildConfig,
)
from agentic_rag.ingestion.contextualization.gateway import (
    ExtractionResponse,
    VerificationResponse,
    CandidateClaim,
    Quote,
)
from agentic_rag.ingestion.contextualization.store import MemoryContextBuildStore
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.ingestion.chunker import ParentChunk
from agentic_rag.models.contextual import ContextArtifactRef
from agentic_rag.persistence.artifacts import LocalArtifactStore
from tests.unit.ingestion.test_context_sources import _child
from tests.unit.ingestion.test_chunker import _canonical, _block


async def fence():
    pass


def setup(tmp_path, texts):
    artifacts = LocalArtifactStore(tmp_path)
    ast = _canonical(tuple(_block(i, text=t) for i, t in enumerate(texts)))
    ref = artifacts.put_json("canonical.json", ast.model_dump(mode="json"))
    child = _child(*texts)
    parent = ParentChunk(
        **{
            **child.model_dump(
                exclude={"parent_id", "parent_ordinal", "search_representation"}
            ),
            "id": child.parent_id,
            "children": (child,),
        }
    )
    claim = IngestionJobClaim(
        job_id="job",
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        owner="worker",
        claim_generation=1,
    )
    return artifacts, ast, ContextArtifactRef(**ref.__dict__), parent, claim


class Recorded:
    def __init__(
        self, verdict="supported", field="company", value="星河公司", timeout=False
    ):
        self.calls = []
        self.verdict = verdict
        self.field = field
        self.value = value
        self.timeout = timeout

    async def extract(self, input):
        self.calls.append(("extract", input))
        source = next(s for s in input.sources if "星河公司" in s.text)
        return ExtractionResponse(
            target_source_child_id=input.target_source_child_id,
            claims=(
                CandidateClaim(
                    field=self.field,
                    value=self.value,
                    source_quotes=(
                        Quote(source_id=source.source_id, quote=source.text),
                    ),
                    scope_region_id="document",
                ),
            ),
        )

    async def verify(self, input):
        self.calls.append(("verify", input))
        if self.timeout:
            raise TimeoutError("recorded timeout")
        return VerificationResponse(
            status=self.verdict,
            source_ids=tuple(q.source_id for q in input.candidate.source_quotes),
        )


def config(**kwargs):
    return ContextBuildConfig(
        enabled=True,
        extraction_model="extract-1",
        verification_model="verify-1",
        provider_fingerprint="a" * 64,
        max_requests=100,
        max_tokens=1000000,
        max_cost=10,
        input_price_per_million=1,
        output_price_per_million=2,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts",
    [
        ("本制度适用于星河公司。 上海差旅标准500一晚。",),
        ("本制度适用于星河公司。", "上海差旅标准500一晚。"),
    ],
)
async def test_all_semantic_candidates_independently_verified_and_retry_reused(
    tmp_path, texts
):
    artifacts, ast, ref, parent, claim = setup(tmp_path, texts)
    model = Recorded()
    store = MemoryContextBuildStore()
    service = ContextualizationService(
        artifacts=artifacts,
        store=store,
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    )
    result = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    child = result.parents[0].children[0]
    assert child.content == parent.children[0].content
    assert result.parents[0].content == parent.content
    assert child.search_representation.context_prefix == "company: 星河公司"
    assert [c[0] for c in model.calls] == ["extract", "verify"]
    again = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert again == result and len(model.calls) == 2
    assert artifacts.read_json(ref) == ast.model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict,field,value,timeout",
    [
        ("insufficient", "company", "星河公司", False),
        ("contradicted", "company", "星河公司", False),
        ("supported", "currency", "人民币", False),
        ("supported", "company", "星河公司", True),
    ],
)
async def test_unverified_unsupported_or_invented_claims_fall_back(
    tmp_path, verdict, field, value, timeout
):
    artifacts, ast, ref, parent, claim = setup(
        tmp_path, ("本制度适用于星河公司。", "上海差旅标准500一晚。")
    )
    model = Recorded(verdict, field, value, timeout)
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    ).enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert (
        result.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
    )
    assert len(model.calls) == 2


def test_config_requires_finite_budgets_and_changes_fingerprint():
    with pytest.raises(ValueError):
        ContextBuildConfig(enabled=True, extraction_model="x", verification_model="y")
    with pytest.raises(ValueError):
        config(max_concurrency=0)
    assert config().fingerprint("a" * 64) != config(
        extraction_prompt_hash="b" * 64
    ).fingerprint("a" * 64)


@pytest.mark.asyncio
async def test_missing_scope_is_rejected_after_independent_verification(tmp_path):
    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))

    class WrongScope(Recorded):
        async def extract(self, input):
            result = await super().extract(input)
            return result.model_copy(
                update={
                    "claims": (
                        result.claims[0].model_copy(
                            update={"scope_region_id": "invented-section"}
                        ),
                    )
                }
            )

    model = WrongScope()
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    ).enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert len(model.calls) == 2
    assert (
        result.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
    )


@pytest.mark.asyncio
async def test_known_exclusions_cannot_be_silently_removed(tmp_path):
    artifacts, ast, ref, parent, claim = setup(
        tmp_path, ("本制度适用于星河公司，不含外包人员。",)
    )
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=Recorded(),
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    ).enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert (
        result.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
    )


@pytest.mark.asyncio
async def test_changed_model_and_disabled_build_are_distinct_and_preserve_raw(tmp_path):
    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))
    model = Recorded()
    service = ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    )
    first = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    second = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config().model_copy(update={"extraction_model": "extract-2"}),
        claim=claim,
        before_side_effect=fence,
    )
    assert first.enriched_chunks_ref != second.enriched_chunks_ref
    assert first.parents[0].children[0].id != second.parents[0].children[0].id
    assert len(model.calls) == 4
    third = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=ContextBuildConfig(),
        claim=claim,
        before_side_effect=fence,
    )
    assert (
        third.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
        and len(model.calls) == 4
    )


@pytest.mark.asyncio
async def test_partial_timeout_preserves_already_verified_field(tmp_path):
    artifacts, ast, ref, parent, claim = setup(
        tmp_path, ("本制度适用于星河公司。", "适用于普通员工。")
    )

    class Partial(Recorded):
        async def extract(self, input):
            result = await super().extract(input)
            source = next(s for s in input.sources if "普通员工" in s.text)
            candidate = CandidateClaim(
                field="population",
                value="普通员工",
                source_quotes=(Quote(source_id=source.source_id, quote=source.text),),
                scope_region_id="document",
            )
            return result.model_copy(update={"claims": (*result.claims, candidate)})

        async def verify(self, input):
            if input.candidate.field == "population":
                self.calls.append(("verify", input))
                raise TimeoutError()
            return await super().verify(input)

    model = Partial()
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    ).enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    rep = result.parents[0].children[0].search_representation
    assert (
        rep.enrichment_status == "partial" and rep.context_prefix == "company: 星河公司"
    )
    assert len(model.calls) == 3


@pytest.mark.asyncio
async def test_model_input_uses_separate_model_counter_and_requires_it(tmp_path):
    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))
    model = Recorded()
    service = ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
    )
    with pytest.raises(ValueError, match="model.*token"):
        await service.enrich(
            ast,
            [parent],
            canonical_ref=ref,
            build=config(),
            claim=claim,
            before_side_effect=fence,
        )
    service = ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: 10000,
    )
    result = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert (
        model.calls == []
        and result.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
    )


@pytest.mark.asyncio
async def test_short_quote_cannot_hide_exclusion_in_supporting_source(tmp_path):
    artifacts, ast, ref, parent, claim = setup(
        tmp_path, ("本制度适用于星河公司，不含外包人员。",)
    )

    class ShortQuote(Recorded):
        async def extract(self, input):
            result = await super().extract(input)
            candidate = result.claims[0]
            return result.model_copy(
                update={
                    "claims": (
                        candidate.model_copy(
                            update={
                                "source_quotes": (
                                    Quote(
                                        source_id=candidate.source_quotes[0].source_id,
                                        quote="星河公司",
                                    ),
                                )
                            }
                        ),
                    )
                }
            )

    model = ShortQuote()
    result = await ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    ).enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert (
        result.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
    )


@pytest.mark.asyncio
async def test_new_context_pipeline_rejects_legacy_384_budget_even_without_llm(
    tmp_path,
):
    artifacts, ast, ref, parent, claim = setup(tmp_path, ("x" * 321,))
    service = ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=None,
        count_tokens=len,
    )
    with pytest.raises(ValueError, match="320|base.*budget"):
        await service.enrich(
            ast,
            [parent],
            canonical_ref=ref,
            build=ContextBuildConfig(),
            claim=claim,
            before_side_effect=fence,
        )


@pytest.mark.asyncio
async def test_conflicting_whole_child_populations_are_ambiguous(tmp_path):
    artifacts, ast, ref, parent, claim = setup(
        tmp_path, ("本制度适用于星河公司，普通员工和管理人员。",)
    )

    class Conflicting(Recorded):
        async def extract(self, input):
            result = await super().extract(input)
            return result.model_copy(
                update={
                    "claims": tuple(
                        result.claims[0].model_copy(
                            update={"field": "population", "value": value}
                        )
                        for value in ("普通员工", "管理人员")
                    )
                }
            )

    model = Conflicting()
    service = ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    )
    result = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert (
        result.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
    )
    assert len(model.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["owner", "quote", "source_id"])
async def test_integrity_errors_fail_build_instead_of_fallback(tmp_path, corruption):
    from agentic_rag.ingestion.contextualization.sources import SourceIntegrityError

    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))

    class Forged(Recorded):
        async def extract(self, input):
            result = await super().extract(input)
            candidate = result.claims[0]
            quote = candidate.source_quotes[0].model_copy(
                update={
                    ("quote" if corruption == "quote" else "source_id"): (
                        "not in original" if corruption == "quote" else "forged"
                    )
                }
            )
            return result.model_copy(
                update={
                    "claims": (
                        candidate.model_copy(update={"source_quotes": (quote,)}),
                    )
                }
            )

    if corruption == "owner":
        claim = claim.model_copy(update={"user_id": "user-2"})
    model = Forged()
    service = ContextualizationService(
        artifacts=artifacts,
        store=MemoryContextBuildStore(),
        model=model,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    )
    with pytest.raises(SourceIntegrityError):
        await service.enrich(
            ast,
            [parent],
            canonical_ref=ref,
            build=config(),
            claim=claim,
            before_side_effect=fence,
        )


def test_tokenizer_identity_changes_build_fingerprint():
    first = config()
    second = ContextBuildConfig.model_validate(
        {**first.model_dump(), "embedding_tokenizer_id": "tokenizer-new"}
    )
    assert first.fingerprint("a" * 64) != second.fingerprint("a" * 64)
