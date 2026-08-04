"""Thin asynchronous adapter around Docling's document converter."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Protocol, cast

from agentic_rag.ingestion.assembler import DocumentEnvelope, FragmentAst
from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore


_DOCLING_COLLECTIONS = (
    "groups",
    "texts",
    "pictures",
    "tables",
    "key_value_items",
    "form_items",
    "field_regions",
    "field_items",
)


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
        if (
            original.uri != envelope.source_uri
            or original.sha256 != envelope.content_hash
        ):
            raise ValueError(
                "original ArtifactRef does not match the document envelope"
            )
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
        collection_offsets = {collection: 0 for collection in _DOCLING_COLLECTIONS}
        for batch_no, (batch_from, batch_to) in enumerate(batch_ranges, start=1):
            batch_document = document
            if len(batch_ranges) > 1:
                batch_document = document.filter(set(range(batch_from, batch_to + 1)))
            batch_export = _restore_page_furniture(
                _export_docling(batch_document),
                exported,
                page_from=batch_from,
                page_to=batch_to,
            )
            batch_export = _globalize_batch_export(
                batch_export,
                page_from=batch_from,
                collection_offsets=collection_offsets,
            )
            actual_page_from, actual_page_to = _page_bounds(batch_export)
            fragment = FragmentAst(
                envelope=envelope,
                batch_no=batch_no,
                page_from=min(batch_from, actual_page_from),
                page_to=max(batch_to, actual_page_to),
                docling_document=batch_export,
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
        converter = self._converter or cast(_Converter, DocumentConverter())
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


def _restore_page_furniture(
    filtered: dict[str, Any],
    original: Mapping[str, Any],
    *,
    page_from: int,
    page_to: int,
) -> dict[str, Any]:
    """Restore only furniture descendants belonging to one filtered page batch.

    Docling 2.118 filters the body tree but does not traverse the furniture tree.
    This adapter selects page-local furniture leaves from the full export, retains
    their structural ancestors, and appends them to the filtered collections using
    a fresh local namespace. The normal globalization pass then assigns references
    shared by all persisted fragments.
    """
    original_root = original.get("furniture")
    filtered_root = filtered.get("furniture")
    if not isinstance(original_root, Mapping):
        return filtered
    if not isinstance(filtered_root, Mapping):
        raise DoclingConversionError("Docling furniture root must be an object")
    filtered_children = filtered_root.get("children")
    if not isinstance(filtered_children, list):
        raise DoclingConversionError("Docling furniture children must be a list")
    if filtered_children:
        return filtered
    local_page_from, _ = _declared_page_bounds(filtered)
    furniture_page_delta = local_page_from - page_from

    items_by_ref: dict[str, Mapping[str, Any]] = {}
    collection_by_ref: dict[str, str] = {}
    for collection in _DOCLING_COLLECTIONS:
        items = original.get(collection, [])
        if not isinstance(items, list):
            raise DoclingConversionError(
                f"Docling collection {collection!r} must be a list"
            )
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(
                item.get("self_ref"), str
            ):
                raise DoclingConversionError(
                    f"Docling collection {collection!r} contains an invalid item"
                )
            self_ref = str(item["self_ref"])
            items_by_ref[self_ref] = item
            collection_by_ref[self_ref] = collection

    selected: set[str] = set()
    selected_children: dict[str, list[str]] = {}

    def select(reference: str, visiting: set[str]) -> bool:
        item = items_by_ref.get(reference)
        if item is None:
            raise DoclingConversionError(
                f"Docling furniture reference {reference!r} does not resolve"
            )
        if reference in visiting:
            raise DoclingConversionError("Docling furniture tree contains a cycle")
        child_refs = _child_references(item)
        next_visiting = {*visiting, reference}
        kept_children = [
            child_ref for child_ref in child_refs if select(child_ref, next_visiting)
        ]
        relevant = bool(kept_children) or any(
            page_from <= page_no <= page_to for page_no in _provenance_pages(item)
        )
        if relevant:
            selected.add(reference)
            selected_children[reference] = kept_children
        return relevant

    root_children = _child_references(original_root)
    kept_root_children = [
        reference for reference in root_children if select(reference, set())
    ]
    if not kept_root_children:
        return filtered

    restored = copy.deepcopy(filtered)
    reference_map: dict[str, str] = {}
    for collection in _DOCLING_COLLECTIONS:
        existing = restored.get(collection, [])
        if not isinstance(existing, list):
            raise DoclingConversionError(
                f"Docling collection {collection!r} must be a list"
            )
        original_items = original.get(collection, [])
        if not isinstance(original_items, list):
            raise DoclingConversionError(
                f"Docling collection {collection!r} must be a list"
            )
        for item in original_items:
            if not isinstance(item, Mapping) or not isinstance(
                item.get("self_ref"), str
            ):
                raise DoclingConversionError(
                    f"Docling collection {collection!r} contains an invalid item"
                )
            old_ref = str(item["self_ref"])
            if old_ref in selected:
                reference_map[old_ref] = f"#/{collection}/{len(existing)}"
                existing.append(copy.deepcopy(dict(item)))
        if existing or collection in restored:
            restored[collection] = existing

    restored_root = copy.deepcopy(dict(original_root))
    restored_root["children"] = [{"$ref": ref} for ref in kept_root_children]
    restored["furniture"] = restored_root
    for old_ref in selected:
        collection = collection_by_ref[old_ref]
        new_ref = reference_map[old_ref]
        index = int(new_ref.rsplit("/", maxsplit=1)[1])
        item = restored[collection][index]
        child_refs = selected_children[old_ref]
        if "children" in item:
            item["children"] = [{"$ref": ref} for ref in child_refs]
        if furniture_page_delta:
            _shift_nested_page_numbers(item, furniture_page_delta)
        _rewrite_docling_references(item, reference_map)
    _rewrite_docling_references(restored_root, reference_map)
    return restored


def _child_references(item: Mapping[str, Any]) -> list[str]:
    children = item.get("children", [])
    if not isinstance(children, list):
        raise DoclingConversionError("Docling furniture children must be a list")
    references: list[str] = []
    for child in children:
        if not isinstance(child, Mapping) or not isinstance(child.get("$ref"), str):
            raise DoclingConversionError(
                "Docling furniture children require string references"
            )
        references.append(str(child["$ref"]))
    return references


def _provenance_pages(item: Mapping[str, Any]) -> set[int]:
    provenance = item.get("prov", [])
    if not isinstance(provenance, list):
        raise DoclingConversionError("Docling provenance must be a list")
    return {
        int(entry["page_no"])
        for entry in provenance
        if isinstance(entry, Mapping) and isinstance(entry.get("page_no"), int)
    }


def _globalize_batch_export(
    document: dict[str, Any],
    *,
    page_from: int,
    collection_offsets: dict[str, int],
) -> dict[str, Any]:
    """Restore document-global pages and references after Docling.filter().

    Docling deliberately creates a standalone document for every filtered page set,
    rebasing both collection indices and page numbers. Fragment artifacts instead
    share one document namespace, so each local collection is shifted by the counts
    already emitted and every reference is rewritten before persistence.
    """
    globalized = copy.deepcopy(document)
    reference_map: dict[str, str] = {}
    for collection in _DOCLING_COLLECTIONS:
        items = globalized.get(collection, [])
        if not isinstance(items, list):
            raise DoclingConversionError(
                f"Docling collection {collection!r} must be a list"
            )
        offset = collection_offsets[collection]
        for local_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise DoclingConversionError(
                    f"Docling collection {collection!r} contains a non-object"
                )
            old_ref = item.get("self_ref")
            if not isinstance(old_ref, str):
                raise DoclingConversionError("Docling items require self_ref")
            reference_map[old_ref] = f"#/{collection}/{offset + local_index}"
        collection_offsets[collection] += len(items)

    _rewrite_docling_references(globalized, reference_map)
    local_page_from, _ = _declared_page_bounds(globalized)
    page_delta = page_from - local_page_from
    if page_delta:
        _shift_docling_page_numbers(globalized, page_delta)
    return globalized


def _declared_page_bounds(document: Mapping[str, Any]) -> tuple[int, int]:
    """Read physical pages without letting cross-page item provenance skew rebasing."""
    pages: set[int] = set()
    raw_pages = document.get("pages")
    if isinstance(raw_pages, Mapping):
        for key, value in raw_pages.items():
            if isinstance(value, Mapping) and isinstance(value.get("page_no"), int):
                pages.add(int(value["page_no"]))
                continue
            try:
                pages.add(int(key))
            except (TypeError, ValueError):
                pass
    return (min(pages), max(pages)) if pages else _page_bounds(document)


def _rewrite_docling_references(value: Any, reference_map: Mapping[str, str]) -> None:
    if isinstance(value, dict):
        self_ref = value.get("self_ref")
        if isinstance(self_ref, str) and self_ref in reference_map:
            value["self_ref"] = reference_map[self_ref]
        reference = value.get("$ref")
        if isinstance(reference, str) and reference in reference_map:
            value["$ref"] = reference_map[reference]
        for child in value.values():
            _rewrite_docling_references(child, reference_map)
    elif isinstance(value, list):
        for child in value:
            _rewrite_docling_references(child, reference_map)


def _shift_docling_page_numbers(document: dict[str, Any], delta: int) -> None:
    pages = document.get("pages")
    if isinstance(pages, Mapping):
        shifted_pages: dict[str, Any] = {}
        for key, page in pages.items():
            try:
                shifted_key = str(int(key) + delta)
            except (TypeError, ValueError) as error:
                raise DoclingConversionError(
                    "Docling pages require numeric keys"
                ) from error
            shifted_pages[shifted_key] = page
        document["pages"] = shifted_pages

    _shift_nested_page_numbers(document, delta)


def _shift_nested_page_numbers(value: Any, delta: int) -> None:
    """Shift every declared page number without changing page-map keys."""
    if isinstance(value, dict):
        page_no = value.get("page_no")
        if isinstance(page_no, int):
            value["page_no"] = page_no + delta
        for child in value.values():
            _shift_nested_page_numbers(child, delta)
    elif isinstance(value, list):
        for child in value:
            _shift_nested_page_numbers(child, delta)
