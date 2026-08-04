"""Unit tests for Docling orchestration around an injected converter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.ingestion.assembler import DocumentEnvelope
from agentic_rag.ingestion.assembler import GlobalAssembler
from agentic_rag.ingestion.parser import DocumentParser
from agentic_rag.persistence.artifacts import LocalArtifactStore


@dataclass
class _Stream:
    name: str
    stream: BytesIO


class _Document:
    def __init__(self, source_pages: list[int], *, rebase_pages: bool = False) -> None:
        self.source_pages = source_pages
        self.rebase_pages = rebase_pages

    def export_to_dict(self, **_kwargs: Any) -> dict[str, Any]:
        page_numbers = (
            list(range(1, len(self.source_pages) + 1))
            if self.rebase_pages
            else self.source_pages
        )
        return {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
            "name": "source",
            "body": {
                "self_ref": "#/body",
                "children": [
                    {"$ref": f"#/texts/{index}"}
                    for index, _page in enumerate(self.source_pages)
                ],
            },
            "texts": [
                {
                    "self_ref": f"#/texts/{index}",
                    "parent": {"$ref": "#/body"},
                    "label": "text",
                    "text": f"page {source_page}",
                    "prov": [{"page_no": output_page}],
                }
                for index, (source_page, output_page) in enumerate(
                    zip(self.source_pages, page_numbers, strict=True)
                )
            ],
            "pages": {str(page): {"page_no": page} for page in page_numbers},
        }

    def filter(self, page_nrs: set[int] | None = None) -> _Document:
        selected = [page for page in self.source_pages if page in (page_nrs or set())]
        return _Document(selected, rebase_pages=True)


@dataclass
class _Result:
    document: _Document


class _Converter:
    def convert(self, source: object, **_kwargs: Any) -> _Result:
        assert isinstance(source, _Stream)
        assert source.name == "report.pdf"
        assert source.stream.read() == b"source bytes"
        return _Result(_Document([1, 2, 3]))


class _FurnitureDocument(_Document):
    def __init__(
        self,
        source_pages: list[int],
        *,
        rebase_pages: bool = False,
        drop_furniture: bool = False,
    ) -> None:
        super().__init__(source_pages, rebase_pages=rebase_pages)
        self.drop_furniture = drop_furniture

    def export_to_dict(self, **kwargs: Any) -> dict[str, Any]:
        exported = super().export_to_dict(**kwargs)
        exported["furniture"] = {"self_ref": "#/furniture", "children": []}
        exported["groups"] = []
        if self.drop_furniture or 3 not in self.source_pages:
            return exported
        header_ref = f"#/texts/{len(self.source_pages)}"
        exported["furniture"]["children"] = [{"$ref": "#/groups/0"}]
        exported["groups"] = [
            {
                "self_ref": "#/groups/0",
                "parent": {"$ref": "#/furniture"},
                "children": [{"$ref": header_ref}],
                "label": "section",
                "name": "headers",
            }
        ]
        exported["texts"].append(
            {
                "self_ref": header_ref,
                "parent": {"$ref": "#/groups/0"},
                "label": "page_header",
                "text": "HEADER",
                "prov": [{"page_no": 3}],
            }
        )
        return exported

    def filter(self, page_nrs: set[int] | None = None) -> _FurnitureDocument:
        selected = [page for page in self.source_pages if page in (page_nrs or set())]
        return _FurnitureDocument(
            selected,
            rebase_pages=True,
            drop_furniture=True,
        )


class _FurnitureConverter:
    def convert(self, source: object, **_kwargs: Any) -> _Result:
        assert isinstance(source, _Stream)
        assert source.stream.read() == b"source bytes"
        return _Result(_FurnitureDocument([1, 2, 3]))


def _envelope(uri: str, sha256: str) -> DocumentEnvelope:
    return DocumentEnvelope(
        document_id="doc-1",
        document_version_id="ver-1",
        user_id="user-1",
        source_type="pdf",
        source_uri=uri,
        content_hash=sha256,
        parser_version="docling-v1",
        pipeline_version="ingestion-v1",
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_parser_filters_docling_document_into_versioned_page_batches(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path)
    original = artifacts.put_bytes("source/report.pdf", b"source bytes")
    parser = DocumentParser(
        source_loader=lambda _ref: b"source bytes",
        artifacts=artifacts,
        converter=_Converter(),
        document_stream_factory=_Stream,
        page_batch_size=2,
    )

    fragments = [
        fragment
        async for fragment in parser.parse_batches(
            original, _envelope(original.uri, original.sha256)
        )
    ]

    assert [(item.batch_no, item.page_from, item.page_to) for item in fragments] == [
        (1, 1, 2),
        (2, 3, 3),
    ]
    assert [
        sorted(int(page) for page in item.docling_document["pages"])
        for item in fragments
    ] == [[1, 2], [3]]
    assert all(
        item.artifact_ref and artifacts.verify(item.artifact_ref) for item in fragments
    )
    assert fragments[0].artifact_ref is not None
    assert fragments[0].artifact_ref.uri.endswith(
        "/fragments/docling-v1/ingestion-v1/batch-000001.json"
    )
    source_refs = [
        item["self_ref"]
        for fragment in fragments
        for item in fragment.docling_document["texts"]
    ]
    assert source_refs == ["#/texts/0", "#/texts/1", "#/texts/2"]
    canonical = GlobalAssembler().assemble(fragments)
    assert [block.text for block in canonical.text_blocks] == [
        "page 1",
        "page 2",
        "page 3",
    ]


@pytest.mark.asyncio
async def test_parser_does_not_double_shift_restored_furniture_provenance(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path)
    original = artifacts.put_bytes("source/report.pdf", b"source bytes")
    parser = DocumentParser(
        source_loader=lambda _ref: b"source bytes",
        artifacts=artifacts,
        converter=_FurnitureConverter(),
        document_stream_factory=_Stream,
        page_batch_size=2,
    )

    fragments = [
        fragment
        async for fragment in parser.parse_batches(
            original, _envelope(original.uri, original.sha256)
        )
    ]

    second = fragments[1]
    assert (second.page_from, second.page_to) == (3, 3)
    assert second.docling_document["pages"] == {"3": {"page_no": 3}}
    assert [
        (item["label"], item["prov"][0]["page_no"])
        for item in second.docling_document["texts"]
    ] == [("text", 3), ("page_header", 3)]
    canonical = GlobalAssembler().assemble(fragments)
    assert [
        item["prov"][0]["page_no"]
        for item in canonical.docling_document["texts"]
        if item["label"] == "page_header"
    ] == [3]


@pytest.mark.asyncio
async def test_parser_rejects_original_ref_that_does_not_match_envelope(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path)
    original = artifacts.put_bytes("source/report.pdf", b"source bytes")
    parser = DocumentParser(
        source_loader=lambda _ref: b"source bytes",
        artifacts=artifacts,
        converter=_Converter(),
        document_stream_factory=_Stream,
    )

    with pytest.raises(ValueError, match="does not match"):
        _ = [
            item
            async for item in parser.parse_batches(
                original, _envelope(original.uri, "b" * 64)
            )
        ]


@pytest.mark.asyncio
async def test_parser_rejects_source_bytes_that_do_not_match_artifact_hash(
    tmp_path: Path,
) -> None:
    artifacts = LocalArtifactStore(tmp_path)
    original = artifacts.put_bytes("source/report.pdf", b"source bytes")
    parser = DocumentParser(
        source_loader=lambda _ref: b"tamper bytes",
        artifacts=artifacts,
        converter=_Converter(),
        document_stream_factory=_Stream,
    )

    with pytest.raises(ValueError, match="integrity"):
        _ = [
            item
            async for item in parser.parse_batches(
                original, _envelope(original.uri, original.sha256)
            )
        ]
