"""Deterministic assembly of Docling fragments into a canonical document AST."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore


SourceType = Literal["pdf", "scanned_pdf", "excel", "text"]
BlockKind = Literal[
    "heading", "paragraph", "list_item", "table", "code", "formula", "other"
]
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


class AstAssemblyError(ValueError):
    """Raised when source provenance or Docling references fail closed."""


class DocumentEnvelope(BaseModel):
    """Stable identity and pipeline metadata surrounding untrusted Docling output."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_version_id: str
    user_id: str
    source_type: SourceType
    source_uri: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_version: str
    pipeline_version: str
    created_at: datetime

    @field_validator(
        "document_id",
        "document_version_id",
        "user_id",
        "source_uri",
        "parser_version",
        "pipeline_version",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("envelope values must not be blank")
        return value

    @field_validator("created_at")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class FragmentAst(BaseModel):
    """One page batch exported by Docling and persisted as immutable JSON."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    envelope: DocumentEnvelope
    batch_no: int = Field(ge=1)
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    docling_document: dict[str, Any]
    schema_version: str = "fragment-ast-v1"
    artifact_ref: ArtifactRef | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _valid_page_range(self) -> FragmentAst:
        if self.page_to < self.page_from:
            raise ValueError("fragment page range is reversed")
        if not self.docling_document:
            raise ValueError("fragment requires a Docling document")
        return self


class SourceRegion(BaseModel):
    """Traceable page region from Docling provenance."""

    model_config = ConfigDict(frozen=True)

    page_no: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None

    @field_validator("bbox")
    @classmethod
    def _finite_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is not None and not all(
            math.isfinite(coordinate) for coordinate in value
        ):
            raise ValueError("bounding box coordinates must be finite")
        return value


class Provenance(BaseModel):
    """Combined page range and all contributing source regions."""

    model_config = ConfigDict(frozen=True)

    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    regions: tuple[SourceRegion, ...]

    @model_validator(mode="after")
    def _valid_range(self) -> Provenance:
        if not self.regions:
            raise ValueError("provenance requires at least one source region")
        if self.page_to < self.page_from:
            raise ValueError("provenance page range is reversed")
        pages = [region.page_no for region in self.regions]
        if self.page_from != min(pages) or self.page_to != max(pages):
            raise ValueError("provenance range must cover all source regions")
        return self


class CanonicalBlock(BaseModel):
    """Normalized retrieval atom before Parent/Child chunking."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: BlockKind
    text: str
    provenance: Provenance
    source_refs: tuple[str, ...]
    heading_level: int | None = Field(default=None, ge=1)

    @field_validator("id", "text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("canonical block identifiers and text must not be blank")
        return value

    @field_validator("source_refs")
    @classmethod
    def _requires_source_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("canonical blocks require source references")
        return value


class CanonicalAst(BaseModel):
    """Canonical, validated document view consumed by later chunking."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    envelope: DocumentEnvelope
    schema_version: str = "canonical-ast-v1"
    docling_document: dict[str, Any]
    text_blocks: tuple[CanonicalBlock, ...]
    artifact_ref: ArtifactRef | None = Field(default=None, exclude=True)


@dataclass(frozen=True, slots=True)
class _Candidate:
    id: str
    kind: BlockKind
    text: str
    provenance: Provenance
    source_refs: tuple[str, ...]
    heading_level: int | None
    source_order: int
    reading_order: int | None
    role: Literal["header", "footer"] | None
    continued_from_previous: bool
    continues_on_next: bool
    continuation_id: str | None


class GlobalAssembler:
    """Run the fixed production assembly passes and optionally persist the result."""

    def __init__(self, *, artifacts: ArtifactStore | None = None) -> None:
        self._artifacts = artifacts

    def assemble(self, fragments: Sequence[FragmentAst]) -> CanonicalAst:
        if not fragments:
            raise AstAssemblyError("at least one Fragment AST is required")
        envelope = fragments[0].envelope
        if any(fragment.envelope != envelope for fragment in fragments[1:]):
            raise AstAssemblyError("all fragments must have the same document envelope")
        ordered_fragments = sorted(fragments, key=lambda item: item.batch_no)
        if len({fragment.batch_no for fragment in ordered_fragments}) != len(
            ordered_fragments
        ):
            raise AstAssemblyError("fragment batch numbers must be unique")

        _validate_fragment_graphs(ordered_fragments)
        known_refs = _known_references(ordered_fragments)
        candidates = _normalize_reading_order(ordered_fragments, known_refs)
        candidates = _join_cross_page_content(
            candidates, namespace=envelope.document_version_id
        )
        candidates = _remove_repeated_furniture_and_ocr_duplicates(candidates)
        candidates = _repair_heading_hierarchy(candidates)
        blocks = _validate_and_materialize(candidates)

        merged_docling_document = _merge_docling_documents(ordered_fragments)
        _validate_complete_docling_graph(merged_docling_document)
        canonical = CanonicalAst(
            envelope=envelope,
            docling_document=merged_docling_document,
            text_blocks=blocks,
        )
        if self._artifacts is None:
            return canonical

        relative_path = (
            f"documents/{envelope.user_id}/{envelope.document_id}/"
            f"{envelope.document_version_id}/canonical/{envelope.parser_version}/"
            f"{envelope.pipeline_version}/{canonical.schema_version}.json"
        )
        artifact_ref = self._artifacts.put_json(
            relative_path,
            canonical.model_dump(mode="json"),
        )
        return canonical.model_copy(update={"artifact_ref": artifact_ref})


def _normalize_reading_order(
    fragments: Sequence[FragmentAst], known_refs: set[str]
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    source_order = 0
    for fragment in fragments:
        doc = fragment.docling_document
        body_order = _body_reading_order(doc)
        raw_blocks: list[Mapping[str, Any]] = []
        custom_blocks = doc.get("blocks")
        if isinstance(custom_blocks, list):
            raw_blocks.extend(
                item for item in custom_blocks if isinstance(item, Mapping)
            )
        else:
            for collection in ("texts", "tables", "key_value_items"):
                items = doc.get(collection, [])
                if isinstance(items, list):
                    raw_blocks.extend(
                        item for item in items if isinstance(item, Mapping)
                    )

        for raw in raw_blocks:
            text = _block_text(raw)
            if not text.strip():
                continue
            self_ref = str(
                raw.get("self_ref") or f"#/fragments/{fragment.batch_no}/{source_order}"
            )
            targets = _reference_targets(raw)
            missing = sorted(target for target in targets if target not in known_refs)
            if missing:
                raise AstAssemblyError(
                    f"Docling reference {missing[0]!r} does not resolve in this document"
                )
            provenance = _provenance(raw, fragment)
            label = str(raw.get("label", "text")).lower()
            kind, role = _block_kind(label)
            explicit_order = raw.get("reading_order")
            if not isinstance(explicit_order, int):
                explicit_order = body_order.get(self_ref)
            heading_level = raw.get("level")
            candidates.append(
                _Candidate(
                    id=_block_id(fragment.envelope.document_version_id, (self_ref,)),
                    kind=kind,
                    text=text.strip(),
                    provenance=provenance,
                    source_refs=(self_ref,),
                    heading_level=(
                        max(1, int(heading_level))
                        if kind == "heading" and isinstance(heading_level, int)
                        else (1 if label == "title" else None)
                    ),
                    source_order=source_order,
                    reading_order=explicit_order,
                    role=role,
                    continued_from_previous=(
                        bool(raw.get("continued_from_previous"))
                        or (
                            kind == "paragraph"
                            and _near_page_edge(raw, doc, edge="start")
                        )
                    ),
                    continues_on_next=(
                        bool(raw.get("continues_on_next"))
                        or (
                            kind == "paragraph"
                            and _near_page_edge(raw, doc, edge="end")
                            and _text_may_continue(text)
                        )
                    ),
                    continuation_id=(
                        str(raw["continuation_id"])
                        if raw.get("continuation_id") is not None
                        else None
                    ),
                )
            )
            source_order += 1

    return sorted(
        candidates,
        key=lambda item: (
            item.provenance.page_from,
            item.reading_order if item.reading_order is not None else item.source_order,
            item.source_order,
        ),
    )


def _join_cross_page_content(
    candidates: Sequence[_Candidate], *, namespace: str
) -> list[_Candidate]:
    joined: list[_Candidate] = []
    for candidate in candidates:
        previous_index = len(joined) - 1
        while previous_index >= 0 and joined[previous_index].role is not None:
            previous_index -= 1
        if previous_index >= 0 and _is_continuation(joined[previous_index], candidate):
            previous = joined[previous_index]
            separator = (
                "\n"
                if candidate.kind == "table"
                else _word_separator(previous.text, candidate.text)
            )
            refs = tuple(dict.fromkeys((*previous.source_refs, *candidate.source_refs)))
            regions = (*previous.provenance.regions, *candidate.provenance.regions)
            joined[previous_index] = replace(
                previous,
                id=_block_id(namespace, refs),
                text=f"{previous.text}{separator}{candidate.text}",
                provenance=Provenance(
                    page_from=min(region.page_no for region in regions),
                    page_to=max(region.page_no for region in regions),
                    regions=regions,
                ),
                source_refs=refs,
                continues_on_next=candidate.continues_on_next,
            )
        else:
            joined.append(candidate)
    return joined


def _is_continuation(previous: _Candidate, current: _Candidate) -> bool:
    if previous.kind != current.kind or current.provenance.page_from not in {
        previous.provenance.page_to,
        previous.provenance.page_to + 1,
    }:
        return False
    if previous.kind not in {"paragraph", "list_item", "table"}:
        return False
    if previous.continues_on_next and current.continued_from_previous:
        return True
    return (
        previous.kind == "table"
        and previous.continuation_id is not None
        and previous.continuation_id == current.continuation_id
    )


def _remove_repeated_furniture_and_ocr_duplicates(
    candidates: Sequence[_Candidate],
) -> list[_Candidate]:
    furniture_pages: dict[tuple[str, str], set[int]] = {}
    for candidate in candidates:
        if candidate.role is not None:
            key = (candidate.role, _normalized_hash_text(candidate.text))
            furniture_pages.setdefault(key, set()).update(
                region.page_no for region in candidate.provenance.regions
            )

    repeated_furniture = {
        key for key, pages in furniture_pages.items() if len(pages) >= 2
    }
    kept: list[_Candidate] = []
    for candidate in candidates:
        if (
            candidate.role is not None
            and (
                candidate.role,
                _normalized_hash_text(candidate.text),
            )
            in repeated_furniture
        ):
            continue
        if any(_is_ocr_duplicate(existing, candidate) for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _is_ocr_duplicate(left: _Candidate, right: _Candidate) -> bool:
    if _normalized_hash_text(left.text) != _normalized_hash_text(right.text):
        return False
    for left_region in left.provenance.regions:
        for right_region in right.provenance.regions:
            if (
                left_region.page_no == right_region.page_no
                and left_region.bbox is not None
                and right_region.bbox is not None
                and _bbox_iou(left_region.bbox, right_region.bbox) >= 0.8
            ):
                return True
    return False


def _repair_heading_hierarchy(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    repaired: list[_Candidate] = []
    previous_level = 0
    for candidate in candidates:
        if candidate.kind != "heading":
            repaired.append(candidate)
            continue
        requested = candidate.heading_level or (
            previous_level + 1 if previous_level else 1
        )
        level = min(requested, previous_level + 1) if previous_level else 1
        repaired.append(replace(candidate, heading_level=max(1, level)))
        previous_level = max(1, level)
    return repaired


def _validate_and_materialize(
    candidates: Sequence[_Candidate],
) -> tuple[CanonicalBlock, ...]:
    identifiers: set[str] = set()
    blocks: list[CanonicalBlock] = []
    for candidate in candidates:
        if candidate.id in identifiers:
            raise AstAssemblyError(f"duplicate canonical block id {candidate.id!r}")
        identifiers.add(candidate.id)
        try:
            block = CanonicalBlock(
                id=candidate.id,
                kind=candidate.kind,
                text=candidate.text,
                provenance=candidate.provenance,
                source_refs=candidate.source_refs,
                heading_level=candidate.heading_level,
            )
        except ValueError as error:
            raise AstAssemblyError(f"invalid canonical provenance: {error}") from error
        blocks.append(block)
    return tuple(blocks)


def _provenance(raw: Mapping[str, Any], fragment: FragmentAst) -> Provenance:
    raw_provenance = raw.get("prov")
    if not isinstance(raw_provenance, list) or not raw_provenance:
        if fragment.envelope.source_type in {"text", "excel"}:
            return Provenance(
                page_from=fragment.page_from,
                page_to=fragment.page_from,
                regions=(SourceRegion(page_no=fragment.page_from),),
            )
        raise AstAssemblyError("every extracted block requires Docling provenance")
    regions: list[SourceRegion] = []
    for item in raw_provenance:
        if not isinstance(item, Mapping) or not isinstance(item.get("page_no"), int):
            raise AstAssemblyError("Docling provenance requires an integer page_no")
        page_no = int(item["page_no"])
        if page_no < fragment.page_from or page_no > fragment.page_to:
            raise AstAssemblyError(
                "block provenance lies outside its fragment page range"
            )
        regions.append(SourceRegion(page_no=page_no, bbox=_bbox(item.get("bbox"))))
    return Provenance(
        page_from=min(region.page_no for region in regions),
        page_to=max(region.page_no for region in regions),
        regions=tuple(regions),
    )


def _bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, Mapping):
        return None
    aliases = (("l", "left"), ("t", "top"), ("r", "right"), ("b", "bottom"))
    values: list[float] = []
    for short, long in aliases:
        value = raw.get(short, raw.get(long))
        if not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return (values[0], values[1], values[2], values[3])


def _near_page_edge(
    raw: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    edge: Literal["start", "end"],
) -> bool:
    provenance = raw.get("prov")
    if not isinstance(provenance, list) or not provenance:
        return False
    item = provenance[0] if edge == "start" else provenance[-1]
    if not isinstance(item, Mapping) or not isinstance(item.get("page_no"), int):
        return False
    bbox = item.get("bbox")
    if not isinstance(bbox, Mapping):
        return False
    coordinates = _bbox(bbox)
    height = _page_height(document, int(item["page_no"]))
    if coordinates is None or height is None or height <= 0:
        return False
    top = coordinates[1]
    bottom = coordinates[3]
    origin = str(bbox.get("coord_origin", "")).upper()
    bottom_left = origin == "BOTTOMLEFT" or (not origin and top >= bottom)
    y_min, y_max = sorted((top, bottom))
    if bottom_left:
        return y_max >= 0.85 * height if edge == "start" else y_min <= 0.15 * height
    return y_min <= 0.15 * height if edge == "start" else y_max >= 0.85 * height


def _page_height(document: Mapping[str, Any], page_no: int) -> float | None:
    pages = document.get("pages")
    if not isinstance(pages, Mapping):
        return None
    page = pages.get(str(page_no), pages.get(page_no))
    if not isinstance(page, Mapping):
        return None
    size = page.get("size")
    if not isinstance(size, Mapping):
        return None
    height = size.get("height")
    return float(height) if isinstance(height, (int, float)) else None


def _text_may_continue(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] not in ".!?;:。！？；："


def _block_kind(
    label: str,
) -> tuple[BlockKind, Literal["header", "footer"] | None]:
    if label in {"title", "section_header"}:
        return "heading", None
    if label == "list_item":
        return "list_item", None
    if label == "table":
        return "table", None
    if label == "code":
        return "code", None
    if label == "formula":
        return "formula", None
    if label == "page_header":
        return "paragraph", "header"
    if label == "page_footer":
        return "paragraph", "footer"
    if label in {"text", "paragraph", "caption", "footnote", "reference"}:
        return "paragraph", None
    return "other", None


def _block_text(raw: Mapping[str, Any]) -> str:
    text = raw.get("text")
    if isinstance(text, str):
        return text
    data = raw.get("data")
    if not isinstance(data, Mapping):
        return ""
    cells = data.get("table_cells")
    if not isinstance(cells, list):
        return ""
    rows: dict[int, list[tuple[int, str]]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("text"), str):
            continue
        row = int(cell.get("start_row_offset_idx", 0))
        column = int(cell.get("start_col_offset_idx", 0))
        rows.setdefault(row, []).append((column, str(cell["text"])))
    return "\n".join(
        " | ".join(text for _, text in sorted(row_cells))
        for _, row_cells in sorted(rows.items())
    )


def _known_references(fragments: Sequence[FragmentAst]) -> set[str]:
    references: set[str] = set()
    for fragment in fragments:
        references.update(_all_self_refs(fragment.docling_document))
    return references


def _validate_fragment_graphs(fragments: Sequence[FragmentAst]) -> None:
    global_locations: dict[str, int] = {}
    collection_offsets = {collection: 0 for collection in _DOCLING_COLLECTIONS}
    for fragment in fragments:
        document = fragment.docling_document
        _validate_fragment_graph_shape(document)
        references = _self_ref_counts(document)
        for reference, count in references.items():
            if count > 1:
                raise AstAssemblyError(f"duplicate Docling self_ref {reference!r}")
            if reference not in {"#/body", "#/furniture"}:
                previous_batch = global_locations.get(reference)
                if previous_batch is not None:
                    raise AstAssemblyError(f"duplicate Docling self_ref {reference!r}")
                global_locations[reference] = fragment.batch_no
        targets = _reference_targets(document)
        missing = sorted(target for target in targets if target not in references)
        if missing:
            raise AstAssemblyError(
                f"complete Docling graph has unresolved reference {missing[0]!r}"
            )
        for collection in _DOCLING_COLLECTIONS:
            items = document.get(collection, [])
            if not isinstance(items, list):
                raise AstAssemblyError(
                    f"fragment Docling graph collection {collection!r} must be a list"
                )
            offset = collection_offsets[collection]
            for local_index, item in enumerate(items):
                if not isinstance(item, Mapping):
                    raise AstAssemblyError(
                        f"fragment Docling graph collection {collection!r} "
                        "has a non-object"
                    )
                if item.get("self_ref") != f"#/{collection}/{offset + local_index}":
                    raise AstAssemblyError(
                        "fragment Docling graph has invalid collection references"
                    )
            collection_offsets[collection] += len(items)


def _validate_fragment_graph_shape(document: Mapping[str, Any]) -> None:
    for root_name in ("body", "furniture"):
        root = document.get(root_name)
        if root is None:
            continue
        if not isinstance(root, Mapping) or root.get("self_ref") != f"#/{root_name}":
            raise AstAssemblyError(
                f"fragment Docling graph has invalid {root_name} root"
            )
        children = root.get("children")
        if not isinstance(children, list):
            raise AstAssemblyError(
                f"fragment Docling graph {root_name} children must be a list"
            )
        if any(
            not isinstance(child, Mapping)
            or not isinstance(child.get("$ref"), str)
            or not child["$ref"]
            for child in children
        ):
            raise AstAssemblyError(
                f"fragment Docling graph {root_name} child must be a reference"
            )

    for collection in _DOCLING_COLLECTIONS:
        items = document.get(collection, [])
        if not isinstance(items, list):
            raise AstAssemblyError(
                f"fragment Docling graph collection {collection!r} must be a list"
            )
        for item in items:
            if not isinstance(item, Mapping):
                raise AstAssemblyError(
                    f"fragment Docling graph collection {collection!r} has a non-object"
                )
            self_ref = item.get("self_ref")
            if (
                not isinstance(self_ref, str)
                or re.fullmatch(rf"#/{re.escape(collection)}/\d+", self_ref) is None
            ):
                raise AstAssemblyError(
                    f"fragment Docling graph collection {collection!r} "
                    "has an invalid item reference"
                )


def _self_ref_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            self_ref = item.get("self_ref")
            if isinstance(self_ref, str):
                counts[self_ref] = counts.get(self_ref, 0) + 1
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    return counts


def _validate_complete_docling_graph(document: Mapping[str, Any]) -> None:
    references = _self_ref_counts(document)
    duplicates = sorted(
        reference for reference, count in references.items() if count > 1
    )
    if duplicates:
        raise AstAssemblyError(f"duplicate Docling self_ref {duplicates[0]!r}")
    missing = sorted(
        target for target in _reference_targets(document) if target not in references
    )
    if missing:
        raise AstAssemblyError(
            f"complete Docling graph has unresolved reference {missing[0]!r}"
        )
    for collection in _DOCLING_COLLECTIONS:
        items = document.get(collection)
        if items is None:
            continue
        if not isinstance(items, list):
            raise AstAssemblyError(f"Docling collection {collection!r} must be a list")
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or item.get("self_ref") != (
                f"#/{collection}/{index}"
            ):
                raise AstAssemblyError(
                    f"Docling collection {collection!r} has invalid global references"
                )


def _all_self_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, Mapping):
        self_ref = value.get("self_ref")
        if isinstance(self_ref, str):
            refs.add(self_ref)
        for child in value.values():
            refs.update(_all_self_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_all_self_refs(child))
    return refs


def _reference_targets(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, Mapping):
        ref = value.get("$ref")
        if isinstance(ref, str):
            refs.add(ref)
        for key, child in value.items():
            if key != "self_ref":
                refs.update(_reference_targets(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_reference_targets(child))
    return refs


def _body_reading_order(doc: Mapping[str, Any]) -> dict[str, int]:
    lookup: dict[str, Mapping[str, Any]] = {}

    def index(value: Any) -> None:
        if isinstance(value, Mapping):
            self_ref = value.get("self_ref")
            if isinstance(self_ref, str):
                lookup[self_ref] = value
            for child in value.values():
                index(child)
        elif isinstance(value, list):
            for child in value:
                index(child)

    index(doc)
    order: dict[str, int] = {}
    visiting: set[str] = set()

    def visit(node: Mapping[str, Any]) -> None:
        node_ref = node.get("self_ref")
        if isinstance(node_ref, str):
            if node_ref in visiting:
                raise AstAssemblyError(
                    "cycle detected in Docling reading-order references"
                )
            visiting.add(node_ref)
            if node_ref not in order and node_ref not in {"#/body", "#/furniture"}:
                order[node_ref] = len(order)
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping) and isinstance(child.get("$ref"), str):
                    target = lookup.get(str(child["$ref"]))
                    if target is not None:
                        visit(target)
        if isinstance(node_ref, str):
            visiting.discard(node_ref)

    body = doc.get("body")
    if isinstance(body, Mapping):
        visit(body)
    return order


def _merge_docling_documents(fragments: Sequence[FragmentAst]) -> dict[str, Any]:
    if len(fragments) == 1:
        return json.loads(json.dumps(fragments[0].docling_document))
    first = fragments[0].docling_document
    excluded = {*_DOCLING_COLLECTIONS, "pages", "blocks", "body", "furniture"}
    merged: dict[str, Any] = {
        key: json.loads(json.dumps(value))
        for key, value in first.items()
        if key not in excluded
    }
    for root_name in ("body", "furniture"):
        valid_roots: list[Mapping[str, Any]] = []
        for fragment in fragments:
            root = fragment.docling_document.get(root_name)
            if isinstance(root, Mapping):
                valid_roots.append(root)
        if valid_roots:
            merged_root = json.loads(json.dumps(valid_roots[0]))
            merged_root["children"] = [
                json.loads(json.dumps(child))
                for root in valid_roots
                for child in root.get("children", [])
                if isinstance(child, Mapping)
            ]
            merged[root_name] = merged_root

    for collection in (*_DOCLING_COLLECTIONS, "blocks"):
        combined: list[Any] = []
        for fragment in fragments:
            items = fragment.docling_document.get(collection, [])
            if not isinstance(items, list):
                continue
            for item in items:
                combined.append(json.loads(json.dumps(item)))
        if combined:
            merged[collection] = combined
    pages: dict[str, Any] = {}
    for fragment in fragments:
        raw_pages = fragment.docling_document.get("pages")
        if isinstance(raw_pages, Mapping):
            for page_no, page in raw_pages.items():
                page_key = str(page_no)
                if page_key in pages:
                    raise AstAssemblyError(f"duplicate Docling page {page_key}")
                pages[page_key] = json.loads(json.dumps(page))
    if pages:
        merged["pages"] = pages
    return merged


def _block_id(namespace: str, source_refs: Sequence[str]) -> str:
    joined_refs = "\x1f".join(source_refs)
    digest = hashlib.sha256(f"{namespace}\x1f{joined_refs}".encode("utf-8")).hexdigest()
    return f"blk_{digest[:24]}"


def _normalized_hash_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _word_separator(left: str, right: str) -> str:
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    if (
        left[-1].isascii()
        and right[0].isascii()
        and left[-1].isalnum()
        and right[0].isalnum()
    ):
        return " "
    return ""


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_x2 = sorted((left[0], left[2]))
    left_y1, left_y2 = sorted((left[1], left[3]))
    right_x1, right_x2 = sorted((right[0], right[2]))
    right_y1, right_y2 = sorted((right[1], right[3]))
    intersection = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1)) * max(
        0.0, min(left_y2, right_y2) - max(left_y1, right_y1)
    )
    left_area = (left_x2 - left_x1) * (left_y2 - left_y1)
    right_area = (right_x2 - right_x1) * (right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
