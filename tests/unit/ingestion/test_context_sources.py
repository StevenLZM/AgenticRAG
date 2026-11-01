"""Grounded source identity and complete scope coverage boundaries."""

from __future__ import annotations

import hashlib
import json

import pytest

from agentic_rag.ingestion.contextualization.sources import (
    SourceCatalog,
    SourceIntegrityError,
)
from tests.unit.ingestion.test_chunker import _block, _canonical
from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk


def _catalog(*texts: str) -> SourceCatalog:
    ast = _canonical(tuple(_block(i, text=text) for i, text in enumerate(texts)))
    encoded = json.dumps(
        ast.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return SourceCatalog.from_ast(ast, hashlib.sha256(encoded).hexdigest())


def _child(*texts: str) -> ChildChunk:
    content = "\n".join(texts)
    spans = []
    offset = 0
    for i, text in enumerate(texts):
        spans.append(
            AstSpan(
                canonical_path=f"#/text_blocks/{i}",
                block_id=f"block-{i}",
                page_from=i + 1,
                page_to=i + 1,
                char_from=0,
                char_to=len(text),
                parent_char_from=offset,
                parent_char_to=offset + len(text),
                separator_before="\n" if i else "",
            )
        )
        offset += len(text) + 1
    return ChildChunk(
        id="original-child",
        parent_id="parent-1",
        parent_ordinal=0,
        document_id="document-1",
        document_version_id="version-1",
        user_id="user-1",
        ordinal=0,
        heading_path=(),
        heading_ast_locators=(),
        content_type="text",
        content=content,
        contextualized_content=content,
        token_count=len(content.split()),
        page_from=1,
        page_to=len(texts),
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        ast_locator=AstLocator(
            spans=tuple(spans),
            segment_ordinal=0,
            parent_char_from=0,
            parent_char_to=len(content),
        ),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_id", "other-owner"),
        ("document_id", "other-document"),
        ("document_version_id", "other-version"),
    ],
)
def test_real_quote_wrong_owner_or_version_is_rejected(field: str, value: str) -> None:
    # Removing identity verification would incorrectly accept a genuine quote.
    catalog = _catalog("适用于星河公司。")
    ref = catalog.resolve_quote(catalog.source_ids[0], "星河公司")
    assert catalog.verify_source(ref) == "星河公司"
    assert (ref.char_from, ref.char_to, ref.page_from, ref.page_to) == (3, 7, 1, 1)
    with pytest.raises(SourceIntegrityError):
        catalog.verify_source(ref.model_copy(update={field: value}))


def test_ambiguous_quote_is_not_silently_first_match() -> None:
    # Using str.find once would silently bind the wrong occurrence.
    catalog = _catalog("甲公司与甲公司")
    with pytest.raises(SourceIntegrityError, match="ambiguous"):
        catalog.resolve_quote(catalog.source_ids[0], "甲公司")
    with pytest.raises(SourceIntegrityError):
        catalog.resolve_quote("forged-source-id", "甲公司")


@pytest.mark.parametrize(
    "update",
    [
        {"char_from": 1},
        {"quote_sha256": "0" * 64},
        {"block_id": "forged"},
        {"canonical_path": "#/text_blocks/99"},
        {"canonical_ast_sha256": "0" * 64},
        {"page_from": 99, "page_to": 99},
    ],
)
def test_forged_coordinates_and_hashes_fail_integrity(update: dict) -> None:
    catalog = _catalog("original source")
    ref = catalog.resolve_quote(catalog.source_ids[0], "original")
    with pytest.raises(SourceIntegrityError):
        catalog.verify_source(ref.model_copy(update=update))


def test_catalog_rejects_wrong_ast_hash() -> None:
    with pytest.raises(SourceIntegrityError):
        SourceCatalog.from_ast(_canonical((_block(0, text="body"),)), "a" * 64)


def test_partial_scope_cannot_label_whole_child() -> None:
    # Checking only the first body span would promote a partial scope to a global label.
    catalog = _catalog("first part", "second part")
    first = catalog.resolve_quote(catalog.source_ids[0], "first part")
    second = catalog.resolve_quote(catalog.source_ids[1], "second part")
    child = _child("first part", "second part")
    assert not catalog.covers_child((first,), child)
    assert catalog.covers_child((first, second), child)
    assert not catalog.covers_child(
        (first, second), child.model_copy(update={"user_id": "other"})
    )


def test_adjacent_scope_ranges_cover_without_gaps() -> None:
    catalog = _catalog("alpha beta")
    left = catalog.resolve_quote(catalog.source_ids[0], "alpha ")
    right = catalog.resolve_quote(catalog.source_ids[0], "beta")
    assert catalog.covers_child((right, left), _child("alpha beta"))
    assert not catalog.covers_child((right,), _child("alpha beta"))
