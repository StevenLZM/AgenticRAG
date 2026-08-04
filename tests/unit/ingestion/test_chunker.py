"""Behavioral tests for deterministic Parent-Child chunking."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

import pytest
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast

from agentic_rag.ingestion.assembler import (
    CanonicalAst,
    CanonicalBlock,
    DocumentEnvelope,
    Provenance,
    SourceRegion,
)
from agentic_rag.ingestion.chunker import (
    ChildBuilder,
    ChunkingPipeline,
    ParentBuilder,
)


def _tokenizer(*, max_tokens: int = 384) -> HuggingFaceTokenizer:
    vocabulary = {"[UNK]": 0}
    vocabulary.update({f"w{index}": index + 1 for index in range(10_000)})
    vocabulary.update({"Product": 10_001, "Design": 10_002, "Details": 10_003})
    core = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    return HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=max_tokens)


def _block(
    ordinal: int,
    *,
    kind: Literal[
        "heading", "paragraph", "list_item", "table", "code", "formula", "other"
    ] = "paragraph",
    text: str,
    page_from: int | None = None,
    page_to: int | None = None,
    heading_level: int | None = None,
) -> CanonicalBlock:
    start = page_from or ordinal + 1
    end = page_to or start
    return CanonicalBlock(
        id=f"block-{ordinal}",
        kind=kind,
        text=text,
        provenance=Provenance(
            page_from=start,
            page_to=end,
            regions=tuple(SourceRegion(page_no=page) for page in range(start, end + 1)),
        ),
        source_refs=(f"#/texts/{ordinal}",),
        heading_level=heading_level,
    )


def _canonical(
    blocks: tuple[CanonicalBlock, ...], *, version_id: str = "version-1"
) -> CanonicalAst:
    return CanonicalAst(
        envelope=DocumentEnvelope(
            document_id="document-1",
            document_version_id=version_id,
            user_id="user-1",
            source_type="pdf",
            source_uri="documents/user-1/document-1/source.pdf",
            content_hash="a" * 64,
            parser_version="docling-v1",
            pipeline_version="ingestion-v1",
            created_at=datetime(2026, 8, 5, tzinfo=UTC),
        ),
        docling_document={"schema_name": "DoclingDocument"},
        text_blocks=blocks,
    )


def _words(start: int, count: int) -> str:
    return " ".join(f"w{index}" for index in range(start, start + count))


def test_parent_child_ids_are_deterministic_version_scoped_and_traceable() -> None:
    canonical = _canonical(
        (
            _block(0, kind="heading", text="Product Design", heading_level=1),
            _block(1, text=_words(0, 900), page_from=1, page_to=2),
            _block(2, text=_words(900, 700), page_from=2, page_to=3),
        )
    )
    pipeline = ChunkingPipeline(_tokenizer())

    first = pipeline.build(canonical)
    second = pipeline.build(canonical)
    other_version = pipeline.build(_canonical(canonical.text_blocks, version_id="version-2"))

    assert [(p.id, [c.id for c in p.children]) for p in first] == [
        (p.id, [c.id for c in p.children]) for p in second
    ]
    assert [parent.id for parent in first] != [parent.id for parent in other_version]
    assert max(child.token_count for parent in first for child in parent.children) <= 384
    assert all(parent.document_version_id == "version-1" for parent in first)
    assert all(parent.user_id == "user-1" for parent in first)
    assert all(parent.document_id == "document-1" for parent in first)
    assert all(parent.ast_locator.startswith("#/text_blocks/") for parent in first)
    assert all(parent.heading_ast_locators == ("#/text_blocks/0",) for parent in first)
    assert all(parent.page_from <= parent.page_to for parent in first)
    assert all(
        parent.content_hash == hashlib.sha256(parent.content.encode()).hexdigest()
        for parent in first
    )
    assert all(
        child.parent_id == parent.id
        and child.document_version_id == parent.document_version_id
        and child.user_id == parent.user_id
        and child.page_from == parent.page_from
        and child.page_to == parent.page_to
        and child.ast_locator == parent.ast_locator
        and child.heading_ast_locators == parent.heading_ast_locators
        and child.content_hash
        for parent in first
        for child in parent.children
    )


def test_parent_builder_targets_section_sized_chunks_without_crossing_headings() -> None:
    canonical = _canonical(
        (
            _block(0, kind="heading", text="Product", heading_level=1),
            _block(1, text=_words(0, 700)),
            _block(2, text=_words(700, 700)),
            _block(3, kind="heading", text="Details", heading_level=1),
            _block(4, text=_words(1400, 1300)),
        )
    )

    parents = ParentBuilder(_tokenizer()).build(canonical)

    assert [parent.heading_path for parent in parents] == [
        ("Product",),
        ("Details",),
    ]
    assert [parent.token_count for parent in parents] == [1400, 1300]
    assert all(1200 <= parent.token_count <= 1800 for parent in parents)
    assert parents[0].page_from == 2
    assert parents[0].page_to == 3


def test_parent_builder_uses_row_fallback_for_oversized_structural_atom() -> None:
    table = "\n".join((_words(0, 1300), _words(1300, 1300)))
    canonical = _canonical((_block(0, kind="table", text=table),))

    parents = ParentBuilder(_tokenizer()).build(canonical)

    assert [parent.token_count for parent in parents] == [1300, 1300]
    assert [parent.row_from for parent in parents] == [1, 2]
    assert [parent.row_to for parent in parents] == [1, 2]
    assert all(parent.content_type == "table" for parent in parents)
    assert all(parent.oversize_reason is None for parent in parents)
    assert "\n".join(parent.content for parent in parents) == table


def test_parent_builder_marks_only_an_indivisible_row_over_hard_limit() -> None:
    canonical = _canonical((_block(0, kind="code", text=_words(0, 2500)),))

    parents = ParentBuilder(_tokenizer()).build(canonical)

    assert len(parents) == 1
    assert parents[0].token_count == 2500
    assert parents[0].oversize_reason == "indivisible_row_exceeds_parent_hard_limit"
    assert parents[0].row_from == 1
    assert parents[0].row_to == 1


def test_parent_builder_uses_docling_semantic_split_for_large_paragraph() -> None:
    canonical = _canonical((_block(0, text=_words(0, 4000)),))

    parents = ParentBuilder(_tokenizer()).build(canonical)

    assert [parent.token_count for parent in parents] == [1800, 1800, 400]
    body_tokens = [token for parent in parents for token in parent.content.split()]
    assert body_tokens == [f"w{index}" for index in range(4000)]
    assert len(body_tokens) == len(set(body_tokens))


def test_child_builder_uses_hybrid_chunker_heading_context_without_overlap() -> None:
    canonical = _canonical(
        (
            _block(0, kind="heading", text="Product", heading_level=1),
            _block(1, kind="heading", text="Design", heading_level=2),
            _block(2, text=_words(0, 800)),
        )
    )
    parent = ParentBuilder(_tokenizer()).build(canonical)[0]

    children = ChildBuilder(_tokenizer()).build(parent)

    assert len(children) == 3
    assert all(child.contextualized_content.startswith("Product\nDesign\n") for child in children)
    assert all(child.token_count <= 384 for child in children)
    body_tokens = [
        token
        for child in children
        for token in child.content.split()
    ]
    assert body_tokens == [f"w{index}" for index in range(800)]
    assert len(body_tokens) == len(set(body_tokens))


def test_parent_builder_rejects_an_empty_canonical_document() -> None:
    with pytest.raises(ValueError, match="at least one content block"):
        ParentBuilder(_tokenizer()).build(_canonical(()))
