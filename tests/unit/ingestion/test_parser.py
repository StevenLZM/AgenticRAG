"""Unit tests for Docling orchestration around an injected converter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.ingestion.assembler import DocumentEnvelope
from agentic_rag.ingestion.parser import DocumentParser
from agentic_rag.persistence.artifacts import LocalArtifactStore


@dataclass
class _Stream:
    name: str
    stream: BytesIO


class _Document:
    def __init__(self, pages: set[int]) -> None:
        self.pages = pages

    def export_to_dict(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
            "name": "source",
            "texts": [
                {
                    "self_ref": f"#/texts/{page}",
                    "label": "text",
                    "text": f"page {page}",
                    "prov": [{"page_no": page}],
                }
                for page in sorted(self.pages)
            ],
            "pages": {str(page): {"page_no": page} for page in sorted(self.pages)},
        }

    def filter(self, page_nrs: set[int] | None = None) -> _Document:
        return _Document(self.pages & (page_nrs or self.pages))


@dataclass
class _Result:
    document: _Document


class _Converter:
    def convert(self, source: object, **_kwargs: Any) -> _Result:
        assert isinstance(source, _Stream)
        assert source.name == "report.pdf"
        assert source.stream.read() == b"source bytes"
        return _Result(_Document({1, 2, 3}))


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
    assert all(item.artifact_ref and artifacts.verify(item.artifact_ref) for item in fragments)
    assert fragments[0].artifact_ref is not None
    assert fragments[0].artifact_ref.uri.endswith(
        "/fragments/docling-v1/ingestion-v1/batch-000001.json"
    )


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
