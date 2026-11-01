"""Search text integrity across indexing, recall and worker serialization."""

from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_rag.ingestion import chunker
from agentic_rag.models.contextual import ContextArtifactRef
from agentic_rag.orchestration.contracts import CandidateResult
from agentic_rag.persistence.elasticsearch import (
    ChildIndexMappingError,
    ElasticsearchChildIndexStore,
)
from agentic_rag.retrieval.adapters.elasticsearch import (
    ElasticsearchBm25Index,
    ElasticsearchVectorIndex,
)
from agentic_rag.retrieval.fusion import rrf_fuse
from agentic_rag.retrieval.models import ChildHit, RankedCandidates, SearchFilter
from agentic_rag.retrieval.reranker import Reranker
from agentic_rag.retrieval import reranker as reranker_module
from tests.unit.ingestion.test_context_rendering import _claim, _render
from tests.unit.ingestion.test_context_sources import _child
from tests.unit.persistence.test_elasticsearch_staging import (
    _Client,
    _Response,
    _context,
    _embedded_child,
)


def representation(*, enriched: bool = True):
    return _render(claims=(_claim(),) if enriched else ()).with_context_ref(
        ContextArtifactRef(
            uri="artifact://context/abc.json", sha256="a" * 64, size_bytes=12
        )
    )


def hit_payload(**updates: Any) -> dict[str, Any]:
    rep = representation()
    values = {
        "child_id": rep.id,
        "parent_id": "parent-1",
        "user_id": "user-1",
        "document_id": "document-1",
        "document_version_id": "version-1",
        "content": rep.content,
        "ast_locator": "source-locator",
        "lane": "dense",
        "lane_rank": 1,
        "score": 1.0,
        **{
            name: getattr(rep, name)
            for name in (
                "contextualized_content",
                "representation_version",
                "content_hash",
                "source_child_id",
                "source_content_hash",
                "context_ref",
                "context_claim_ids",
                "build_fingerprint",
                "context_prefix",
                "enrichment_status",
            )
        },
    }
    values.update(updates)
    return values


def test_final_chunk_binds_source_and_keeps_original_provenance() -> None:
    raw = _child("policy body")
    rep = representation()
    final = chunker.apply_search_representation(raw, rep)
    assert final.id == rep.id != raw.id
    assert final.content == raw.content
    assert final.ast_locator == raw.ast_locator
    assert final.search_representation.source_child_id == raw.id
    assert final.contextualized_content == "company: 星河公司\n\npolicy body"
    assert chunker.ChildChunk.model_validate_json(final.model_dump_json()) == final
    assert "search_representation" not in raw.model_dump(mode="json")


@pytest.mark.parametrize("mutation", ["draft", "source", "id", "text"])
def test_final_chunk_rejects_unbound_representation(mutation: str) -> None:
    raw = _child("policy body")
    rep = representation()
    changes = {
        "draft": {"context_ref": None},
        "source": {"source_child_id": "another"},
        "id": {"id": "forged-final-id"},
        "text": {"content": "changed original"},
    }
    with pytest.raises(ValueError):
        chunker.apply_search_representation(
            raw, rep.model_copy(update=changes[mutation])
        )


def test_context_hit_and_worker_json_preserve_search_representation() -> None:
    hit = ChildHit(**hit_payload())
    assert hit.search_content == "company: 星河公司\n\npolicy body"
    ranked = RankedCandidates(
        query="q",
        hits=(hit,),
        user_id="user-1",
        snapshot_id="s",
        index_generation="index-v5",
    )
    result = CandidateResult(
        origin="fast_pipeline",
        origin_key="fast",
        criterion_ids=("answer",),
        candidates=(ranked,),
    )
    restored = CandidateResult.model_validate_json(result.model_dump_json())
    assert restored.candidates[0].hits[0] == hit
    assert restored.candidates[0].hits[0].context_ref == representation().context_ref
    legacy = ChildHit(
        **{
            key: value
            for key, value in hit_payload().items()
            if key
            in {
                "child_id",
                "parent_id",
                "user_id",
                "document_id",
                "document_version_id",
                "content",
                "ast_locator",
                "lane",
                "lane_rank",
                "score",
            }
        }
    )
    assert legacy.search_content == "policy body"
    # Old ES records already had these two fields, but their rerank input stays original.
    old_record = legacy.model_dump()
    old_record.update(
        contextualized_content="Heading\npolicy body", content_hash="a" * 64
    )
    assert ChildHit.model_validate(old_record).search_content == "policy body"


@pytest.mark.parametrize(
    "missing",
    [
        "contextualized_content",
        "content_hash",
        "source_child_id",
        "source_content_hash",
        "context_ref",
        "build_fingerprint",
        "context_claim_ids",
        "context_prefix",
        "enrichment_status",
        "representation_version",
    ],
)
def test_context_hit_cannot_downgrade_incomplete_new_record(missing: str) -> None:
    payload = hit_payload()
    payload.pop(missing)
    with pytest.raises(ValidationError):
        ChildHit(**payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"content_hash": "b" * 64},
        {"source_content_hash": "b" * 64},
        {"contextualized_content": ""},
        {"representation_version": "context-v2"},
        {"context_prefix": ""},
        {"enrichment_status": "structural_only"},
        {"context_claim_ids": ()},
    ],
)
def test_context_hit_rejects_corrupt_text_or_unknown_version(
    updates: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ChildHit(**hit_payload(**updates))


@pytest.mark.parametrize(
    "updates",
    [
        {"document_version_id": "other"},
        {"parent_id": "other"},
        {"content": "different"},
    ],
)
def test_rrf_rejects_conflicting_legacy_duplicate_identity(
    updates: dict[str, Any],
) -> None:
    hit = ChildHit(
        child_id="a",
        parent_id="p",
        user_id="u",
        document_id="d",
        document_version_id="v",
        content="original",
        ast_locator="l",
        lane="dense",
        lane_rank=1,
        score=1,
    )
    with pytest.raises(ValueError, match="conflict"):
        rrf_fuse([[hit], [hit.model_copy(update=updates)]])


def test_rrf_rejects_conflicting_search_hash_even_for_same_source() -> None:
    first = ChildHit(**hit_payload())
    second = ChildHit(**hit_payload(lane="bm25"))
    assert rrf_fuse([[first], [second]])[0].context_ref == first.context_ref
    with pytest.raises(ValueError, match="conflict|hash"):
        rrf_fuse([[first], [second.model_copy(update={"content_hash": "b" * 64})]])


class RoundTripClient(_Client):
    async def search(self, **kwargs: Any) -> _Response:
        source = {
            key: value
            for key, value in self.operations[1].items()
            if key in kwargs["source_includes"]
        }
        return _Response({"hits": {"hits": [{"_source": source, "_score": 1.0}]}})


@pytest.mark.parametrize("enriched", [False, True])
async def test_es_new_schema_roundtrip_is_identical_for_both_lanes(
    enriched: bool,
) -> None:
    client = RoundTripClient()
    rep = representation(enriched=enriched)
    final = chunker.apply_search_representation(_child("policy body"), rep)
    embedded = replace(_embedded_child(), chunk=final)
    store = ElasticsearchChildIndexStore(
        client, expected_representation_version="context-v1"
    )
    await store.stage(_context(), [embedded])
    assert client.operations[1]["contextualized_content"] == rep.contextualized_content
    assert client.operations[1]["source_child_id"] == "original-child"
    assert client.indices.mappings["_meta"]["representation_version"] == "context-v1"
    assert set(client.operations[1]) <= set(client.indices.mappings["properties"])
    for adapter in (ElasticsearchVectorIndex, ElasticsearchBm25Index):
        lane = adapter(
            client,
            index_generation="index-v2",
            expected_representation_version="context-v1",
        )
        query = [0.1] * 1024 if adapter is ElasticsearchVectorIndex else "policy"
        hits = await lane.search(
            query, SearchFilter(user_id="user-1", index_generation="index-v2"), 1
        )
        assert hits[0].search_content == rep.contextualized_content
        assert hits[0].context_ref == rep.context_ref
        assert hits[0].context_prefix == rep.context_prefix
        assert hits[0].enrichment_status == rep.enrichment_status
        assert hits[0].content == "policy body"


async def test_new_generation_requires_representation_and_legacy_schema_stays_unchanged() -> (
    None
):
    client = RoundTripClient()
    old_store = ElasticsearchChildIndexStore(client)
    await old_store.stage(_context(), [_embedded_child()])
    assert client.indices.mappings["_meta"]["schema_version"] == 2
    assert "representation_version" not in client.indices.mappings["properties"]
    await old_store.validate_index("index-v2")
    lane = ElasticsearchBm25Index(
        client,
        index_generation="index-v2",
        expected_representation_version="context-v1",
    )
    with pytest.raises(ValueError, match="representation"):
        await lane.search(
            "policy", SearchFilter(user_id="user-1", index_generation="index-v2"), 1
        )
    new_client = RoundTripClient()
    new_store = ElasticsearchChildIndexStore(
        new_client, expected_representation_version="context-v1"
    )
    with pytest.raises(ChildIndexMappingError, match="representation"):
        await new_store.stage(_context(), [_embedded_child()])
    assert new_client.bulk_calls == 0
    with pytest.raises(ChildIndexMappingError):
        await ElasticsearchChildIndexStore(
            client, expected_representation_version="context-v1"
        ).validate_index("index-v2")


async def test_reranker_scores_validated_search_text() -> None:
    class Encoder:
        def predict(self, pairs):
            return [9.0 if "星河公司" in document else 0.0 for _, document in pairs]

    service = Reranker(Encoder(), model_version="test")
    try:
        result = await service.rerank("q", [ChildHit(**hit_payload())])
        assert result.hits[0].rerank_score == 9.0
    finally:
        service.close()


class CharacterTokenizer:
    """Deterministic stand-in for an offline model tokenizer with 3 pair tokens."""

    def __call__(
        self,
        text,
        text_pair=None,
        *,
        add_special_tokens,
        truncation,
        return_offsets_mapping=False,
    ):
        assert truncation is False
        result = {
            "input_ids": list(
                range(
                    len(text)
                    + (len(text_pair) if text_pair is not None else 0)
                    + (3 if add_special_tokens else 0)
                )
            )
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class LengthCheckingEncoder:
    def __init__(self, capacity):
        self.capacity = capacity
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        assert all(
            len(query) + len(document) + 3 <= self.capacity for query, document in pairs
        )
        return [9.0 if query.endswith("z") else 1.0 for query, _ in pairs]


async def test_pair_windows_preserve_document_qualifiers_and_entire_query_with_max_score() -> (
    None
):
    hit = ChildHit(**hit_payload())
    document = hit.search_content
    prefix, body = "company: 星河公司", "abcdefghiz"
    capacity = len(document) + len(prefix + "\n\n") + 5 + 3
    model = LengthCheckingEncoder(capacity)
    tokenizer = reranker_module.TokenizerPairAdapter(
        CharacterTokenizer(), max_pair_tokens=capacity
    )
    service = Reranker(model, model_version="offline", pair_tokenizer=tokenizer)
    try:
        result = await service.rerank(
            prefix + "\n\n" + body, [hit], protected_query_context=prefix
        )
        assert result.degraded is False
        assert result.hits[0].rerank_score == 9.0
        assert result.window_counts == (2,)
        assert model.calls == [
            [(prefix + "\n\nabcde", document), (prefix + "\n\nfghiz", document)]
        ]
    finally:
        service.close()


@pytest.mark.parametrize("body,capacity_extra", [("x", 2), ("abcdefghijk", 5)])
async def test_impossible_pair_or_more_than_two_windows_falls_back_without_inference(
    body, capacity_extra
) -> None:
    hit = ChildHit(**hit_payload())
    capacity = len(hit.search_content) + capacity_extra + 3
    model = LengthCheckingEncoder(capacity)
    tokenizer = reranker_module.TokenizerPairAdapter(
        CharacterTokenizer(), max_pair_tokens=capacity
    )
    service = Reranker(model, model_version="offline", pair_tokenizer=tokenizer)
    # For the single-character case the document itself exceeds model capacity.
    if body == "x":
        tokenizer = reranker_module.TokenizerPairAdapter(
            CharacterTokenizer(), max_pair_tokens=len(hit.search_content) + 2
        )
        service.close()
        service = Reranker(model, model_version="offline", pair_tokenizer=tokenizer)
    try:
        result = await service.rerank(body, [hit])
        assert result.degraded is True
        assert result.degradation_reason == "rerank_input_too_long"
        assert result.hits == [hit]
        assert model.calls == []
    finally:
        service.close()


async def test_normal_pair_scores_once_and_inconsistent_protected_prefix_is_rejected() -> (
    None
):
    hit = ChildHit(**hit_payload())
    model = LengthCheckingEncoder(100)
    service = Reranker(
        model,
        model_version="offline",
        pair_tokenizer=reranker_module.TokenizerPairAdapter(
            CharacterTokenizer(), max_pair_tokens=100
        ),
    )
    try:
        result = await service.rerank(
            "company\n\nq", [hit], protected_query_context="company"
        )
        assert result.window_counts == (1,)
        assert model.calls == [[("company\n\nq", hit.search_content)]]
        with pytest.raises(ValueError, match="prefix"):
            await service.rerank(
                "different\n\nq", [hit], protected_query_context="company"
            )
    finally:
        service.close()


async def test_tokenizer_claiming_complete_but_losing_query_is_rejected() -> None:
    class CorruptAdapter:
        max_pair_tokens = 1

        def count_pair(self, query, document):
            return 2

        def query_windows(self, body, **kwargs):
            return reranker_module.QueryWindows(windows=(body[:1],), complete=True)

    service = Reranker(
        LengthCheckingEncoder(100),
        model_version="offline",
        pair_tokenizer=CorruptAdapter(),
    )
    try:
        with pytest.raises(ValueError, match="coverage"):
            await service.rerank("complete question", [ChildHit(**hit_payload())])
    finally:
        service.close()


async def test_mixed_candidate_window_scores_are_aggregated_per_original_hit() -> None:
    context_hit = ChildHit(**hit_payload())
    short_hit = ChildHit(
        child_id="short",
        parent_id="p",
        user_id="u",
        document_id="d",
        document_version_id="v",
        content="a",
        ast_locator="l",
        lane="dense",
        lane_rank=2,
        score=1,
    )
    capacity = len(context_hit.search_content) + 5 + 3

    class Encoder:
        def predict(self, pairs):
            assert pairs == [
                ("abcde", context_hit.search_content),
                ("fghiz", context_hit.search_content),
                ("abcdefghiz", "a"),
            ]
            return [1, 9, 5]

    service = Reranker(
        Encoder(),
        model_version="offline",
        pair_tokenizer=reranker_module.TokenizerPairAdapter(
            CharacterTokenizer(), max_pair_tokens=capacity
        ),
    )
    try:
        result = await service.rerank("abcdefghiz", [context_hit, short_hit])
        assert result.window_counts == (2, 1)
        assert [(hit.child_id, hit.rerank_score) for hit in result.hits] == [
            (context_hit.child_id, 9),
            ("short", 5),
        ]
    finally:
        service.close()


async def test_explicit_scope_disables_windows_and_keeps_complete_pair():
    hit = ChildHit(**hit_payload())
    query = "星河公司普通员工abcdefghiz"
    for extra, degraded in ((10, True), (len(query), False)):
        capacity = len(hit.search_content) + extra + 3
        model = LengthCheckingEncoder(capacity)
        service = Reranker(model, model_version="offline", pair_tokenizer=reranker_module.TokenizerPairAdapter(CharacterTokenizer(), max_pair_tokens=capacity))
        try:
            result = await service.rerank(query, [hit], allow_query_windows=False)
            assert result.degraded is degraded
            if degraded:
                assert result.degradation_reason == "rerank_input_too_long" and not model.calls
            else:
                assert model.calls == [[(query, hit.search_content)]]
        finally:
            service.close()
