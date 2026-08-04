"""Thin asynchronous adapter around Docling's document converter."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Protocol

from agentic_rag.ingestion.assembler import DocumentEnvelope, FragmentAst
from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore


class DoclingUnavailableError(RuntimeError):
    """Raised when the declared Docling runtime dependency cannot be loaded."""


class DoclingConversionError(RuntimeError):
    """Raised when Docling does not return a valid document export."""


class _DoclingDocument(Protocol):
    def export_to_dict(self, **kwargs: Any) -> dict[str, Any]: ...

    def filter(self, page_nrs: set[int] | None = None) -> _DoclingDocument: ...


class _ConversionResult(Protocol):
    document: _DoclingDocument


class _Converter(Protocol):
    def convert(self, source: object, **kwargs: Any) -> _ConversionResult: ...


class DocumentParser:
    """Convert untrusted source bytes with Docling and persist page-batch fragments."""

    def __init__(
        self,
        *,
        source_loader: Callable[[ArtifactRef], bytes],
        artifacts: ArtifactStore,
        converter: _Converter | None = None,
        document_stream_factory: Callable[..., object] | None = None,
        page_batch_size: int = 20,
        max_num_pages: int = 2_000,
        max_file_size: int = 50 * 1024 * 1024,
    ) -> None:
        if page_batch_size <= 0 or max_num_pages <= 0 or max_file_size <= 0:
            raise ValueError("Docling parser limits must be positive")
        self._source_loader = source_loader
        self._artifacts = artifacts
        self._converter = converter
        self._document_stream_factory = document_stream_factory
        self._page_batch_size = page_batch_size
        self._max_num_pages = max_num_pages
        self._max_file_size = max_file_size

    async def parse_batches(
        self, original: ArtifactRef, envelope: DocumentEnvelope
    ) -> AsyncIterator[FragmentAst]:
        if original.uri != envelope.source_uri or original.sha256 != envelope.content_hash:
            raise ValueError("original ArtifactRef does not match the document envelope")
        content = await asyncio.to_thread(self._source_loader, original)
        if (
            len(content) != original.size_bytes
            or hashlib.sha256(content).hexdigest() != original.sha256
        ):
            raise ValueError("source Artifact failed integrity verification")
        converter, document_stream = self._runtime()
        source_name = PurePosixPath(original.uri.removeprefix("artifact://")).name
        stream = document_stream(name=source_name, stream=BytesIO(content))
        result = await asyncio.to_thread(
            converter.convert,
            stream,
            raises_on_error=True,
            max_num_pages=self._max_num_pages,
            max_file_size=self._max_file_size,
        )
        document = getattr(result, "document", None)
        if document is None or not hasattr(document, "export_to_dict"):
            raise DoclingConversionError("Docling conversion returned no document")

        exported = _export_docling(document)
        page_from, page_to = _page_bounds(exported)
        batch_ranges = list(
            _page_ranges(page_from, page_to, batch_size=self._page_batch_size)
        )
        for batch_no, (batch_from, batch_to) in enumerate(batch_ranges, start=1):
            batch_document = document
            if len(batch_ranges) > 1:
                batch_document = document.filter(set(range(batch_from, batch_to + 1)))
            fragment = FragmentAst(
                envelope=envelope,
                batch_no=batch_no,
                page_from=batch_from,
                page_to=batch_to,
                docling_document=_export_docling(batch_document),
            )
            relative_path = (
                f"documents/{envelope.user_id}/{envelope.document_id}/"
                f"{envelope.document_version_id}/fragments/{envelope.parser_version}/"
                f"{envelope.pipeline_version}/batch-{batch_no:06d}.json"
            )
            artifact_ref = await asyncio.to_thread(
                self._artifacts.put_json,
                relative_path,
                fragment.model_dump(mode="json"),
            )
            yield fragment.model_copy(update={"artifact_ref": artifact_ref})

    def _runtime(self) -> tuple[_Converter, Callable[..., object]]:
        if self._converter is not None and self._document_stream_factory is not None:
            return self._converter, self._document_stream_factory
        try:
            from docling.datamodel.base_models import DocumentStream  # type: ignore[import-not-found]
            from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
        except ImportError as error:
            raise DoclingUnavailableError(
                "Docling is required for document parsing; install project dependencies"
            ) from error
        converter: _Converter = self._converter or DocumentConverter()
        document_stream: Callable[..., object] = (
            self._document_stream_factory or DocumentStream
        )
        return converter, document_stream


def _export_docling(document: _DoclingDocument) -> dict[str, Any]:
    exported = document.export_to_dict(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    if not isinstance(exported, dict):
        raise DoclingConversionError("Docling export must be a JSON object")
    schema_name = exported.get("schema_name")
    if schema_name not in {None, "DoclingDocument"}:
        raise DoclingConversionError("Docling export has an unexpected schema")
    if schema_name is None:
        exported["schema_name"] = "DoclingDocument"
    return exported


def _page_bounds(document: Mapping[str, Any]) -> tuple[int, int]:
    pages: set[int] = set()
    raw_pages = document.get("pages")
    if isinstance(raw_pages, Mapping):
        for key, value in raw_pages.items():
            if isinstance(value, Mapping) and isinstance(value.get("page_no"), int):
                pages.add(int(value["page_no"]))
            else:
                try:
                    pages.add(int(key))
                except (TypeError, ValueError):
                    pass
    _collect_provenance_pages(document, pages)
    if not pages:
        return 1, 1
    if min(pages) < 1:
        raise DoclingConversionError("Docling page numbers must be one-based")
    return min(pages), max(pages)


def _collect_provenance_pages(value: Any, pages: set[int]) -> None:
    if isinstance(value, Mapping):
        if isinstance(value.get("page_no"), int):
            pages.add(int(value["page_no"]))
        for child in value.values():
            _collect_provenance_pages(child, pages)
    elif isinstance(value, list):
        for child in value:
            _collect_provenance_pages(child, pages)


def _page_ranges(
    page_from: int, page_to: int, *, batch_size: int
) -> Iterator[tuple[int, int]]:
    for start in range(page_from, page_to + 1, batch_size):
        yield start, min(start + batch_size - 1, page_to)
