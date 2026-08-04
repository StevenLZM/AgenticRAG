"""Deterministic structure-semantic Parent/Child chunking.

Parent chunks retain canonical-AST provenance and favor complete sections.  Child
chunks are always produced by Docling's ``HybridChunker``; the small tokenizer
adapter below only narrows the configured tokenizer's limit and never provides a
second tokenization implementation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from docling.chunking import HybridChunker  # type: ignore[import-not-found]
from docling_core.transforms.chunker.tokenizer.base import (  # type: ignore[import-not-found]
    BaseTokenizer,
)
from docling_core.transforms.chunker.hierarchical_chunker import (  # type: ignore[import-not-found]
    ChunkingDocSerializer,
)
from docling_core.transforms.serializer.base import (  # type: ignore[import-not-found]
    BaseDocSerializer,
    BaseSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (  # type: ignore[import-not-found]
    MarkdownTableSerializer,
)
from docling_core.types.doc import (  # type: ignore[import-not-found]
    DocItemLabel,
    DoclingDocument,
    TableCell,
    TableData,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.ingestion.assembler import CanonicalAst, CanonicalBlock
from agentic_rag.runtime.ids import content_id


PARENT_TARGET_MIN_TOKENS = 1200
PARENT_TARGET_MAX_TOKENS = 1800
PARENT_HARD_MAX_TOKENS = 2400
CHILD_MAX_TOKENS = 384


class ChunkingError(ValueError):
    """Raised when chunking cannot preserve a required production invariant."""


class AstSpan(BaseModel):
    """Resolvable half-open range within one Canonical AST text block."""

    model_config = ConfigDict(frozen=True)

    canonical_path: str = Field(pattern=r"^#/text_blocks/(0|[1-9][0-9]*)$")
    block_id: str
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    char_from: int = Field(ge=0)
    char_to: int = Field(gt=0)
    row_from: int | None = Field(default=None, ge=0)
    row_to: int | None = Field(default=None, ge=1)
    table_header_row_to: int = Field(default=0, ge=0)
    parent_char_from: int = Field(ge=0)
    parent_char_to: int = Field(gt=0)
    separator_before: str = ""

    @model_validator(mode="after")
    def _valid_span(self) -> AstSpan:
        if self.char_to <= self.char_from:
            raise ValueError("AST character range is empty or reversed")
        if self.page_to < self.page_from:
            raise ValueError("AST span page range is reversed")
        if self.parent_char_to <= self.parent_char_from:
            raise ValueError("Parent character range is empty or reversed")
        if (self.row_from is None) != (self.row_to is None):
            raise ValueError("AST row range must be complete")
        if self.row_from is not None and self.row_to is not None:
            if self.row_to <= self.row_from:
                raise ValueError("AST row range is empty or reversed")
            if self.table_header_row_to > self.row_from:
                raise ValueError("table header must precede the selected body rows")
        elif self.table_header_row_to:
            raise ValueError("table header metadata requires a row range")
        return self


class AstLocator(BaseModel):
    """Structured provenance locator persisted with a Parent or Child."""

    model_config = ConfigDict(frozen=True)

    spans: tuple[AstSpan, ...]
    segment_ordinal: int = Field(ge=0)
    parent_char_from: int = Field(ge=0)
    parent_char_to: int = Field(gt=0)

    @model_validator(mode="after")
    def _non_empty(self) -> AstLocator:
        if not self.spans:
            raise ValueError("AST locator requires at least one span")
        if self.parent_char_to <= self.parent_char_from:
            raise ValueError("locator Parent range is empty or reversed")
        return self


class ChildChunk(BaseModel):
    """Embedding/search unit that retains its Parent and source provenance."""

    model_config = ConfigDict(frozen=True)

    id: str
    parent_id: str
    parent_ordinal: int = Field(ge=0)
    document_id: str
    document_version_id: str
    user_id: str
    ordinal: int = Field(ge=0)
    heading_path: tuple[str, ...]
    heading_ast_locators: tuple[str, ...]
    content_type: str
    content: str
    contextualized_content: str
    token_count: int = Field(ge=1, le=CHILD_MAX_TOKENS)
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    ast_locator: AstLocator
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _valid_range(self) -> ChildChunk:
        if self.page_to < self.page_from:
            raise ValueError("child page range is reversed")
        if not self.content.strip() or not self.contextualized_content.strip():
            raise ValueError("child content must not be blank")
        return self


class ParentChunk(BaseModel):
    """Reasoning unit stored with full version-scoped source metadata."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    document_version_id: str
    user_id: str
    ordinal: int = Field(ge=0)
    heading_path: tuple[str, ...]
    heading_ast_locators: tuple[str, ...]
    content_type: str
    content: str
    token_count: int = Field(ge=1)
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    ast_locator: AstLocator
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_from: int | None = Field(default=None, ge=1)
    row_to: int | None = Field(default=None, ge=1)
    oversize_reason: Literal["indivisible_row_exceeds_parent_hard_limit"] | None = None
    children: tuple[ChildChunk, ...] = ()

    @model_validator(mode="after")
    def _valid_invariants(self) -> ParentChunk:
        if self.page_to < self.page_from:
            raise ValueError("parent page range is reversed")
        if not self.content.strip():
            raise ValueError("parent content must not be blank")
        if (self.row_from is None) != (self.row_to is None):
            raise ValueError("parent row range must be complete")
        if self.row_from is not None and self.row_to is not None:
            if self.row_to < self.row_from:
                raise ValueError("parent row range is reversed")
        if self.token_count > PARENT_HARD_MAX_TOKENS and self.oversize_reason is None:
            raise ValueError("oversized parent requires an explicit indivisible-row reason")
        if self.oversize_reason is not None and self.content_type not in {"table", "code", "log"}:
            raise ValueError("only structural row content may exceed the parent hard limit")
        return self


@dataclass(frozen=True, slots=True)
class _ParentPiece:
    content: str
    content_type: str
    page_from: int
    page_to: int
    canonical_path: str
    block_id: str
    char_from: int
    char_to: int
    row_from: int | None = None
    row_to: int | None = None
    table_header_row_to: int = 0
    oversize_reason: Literal["indivisible_row_exceeds_parent_hard_limit"] | None = None


@dataclass(frozen=True, slots=True)
class _ParsedTableRow:
    cells: tuple[str, ...]
    raw_index: int
    char_from: int
    char_to: int


class _LimitedTokenizer(BaseTokenizer):
    """Docling tokenizer view with a narrower maximum token window."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delegate: BaseTokenizer
    max_tokens: int = Field(gt=0)

    def count_tokens(self, text: str) -> int:
        return int(self.delegate.count_tokens(text))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> object:
        return self.delegate.get_tokenizer()


class _StructuredSerializerProvider(BaseSerializerProvider):
    """Use Docling's Markdown table serializer so row/header splitting is retained."""

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class ParentBuilder:
    """Build section-aware Parent chunks with a bounded row fallback."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        target_min_tokens: int = PARENT_TARGET_MIN_TOKENS,
        target_max_tokens: int = PARENT_TARGET_MAX_TOKENS,
        hard_max_tokens: int = PARENT_HARD_MAX_TOKENS,
    ) -> None:
        if not 0 < target_min_tokens <= target_max_tokens <= hard_max_tokens:
            raise ValueError("parent token thresholds must be positive and ordered")
        self._tokenizer = tokenizer
        self._target_min = target_min_tokens
        self._target_max = target_max_tokens
        self._hard_max = hard_max_tokens

    def build(self, canonical_ast: CanonicalAst) -> list[ParentChunk]:
        if not canonical_ast.text_blocks:
            raise ValueError("canonical AST requires at least one content block")

        heading_path: list[str] = []
        heading_ast_locators: list[str] = []
        pending: list[_ParentPiece] = []
        drafts: list[
            tuple[tuple[str, ...], tuple[str, ...], list[_ParentPiece]]
        ] = []

        def flush() -> None:
            if pending:
                drafts.append(
                    (
                        tuple(heading_path),
                        tuple(heading_ast_locators),
                        list(pending),
                    )
                )
                pending.clear()

        for block_index, block in enumerate(canonical_ast.text_blocks):
            if block.kind == "heading":
                flush()
                _update_heading_path(
                    heading_path,
                    heading_ast_locators,
                    block,
                    block_index,
                )
                continue

            pieces = self._pieces_for_block(block, block_index)
            for piece in pieces:
                if piece.content_type in {"table", "code", "log"}:
                    flush()
                    drafts.append(
                        (
                            tuple(heading_path),
                            tuple(heading_ast_locators),
                            [piece],
                        )
                    )
                    continue
                self._append_regular_piece(
                    pending,
                    drafts,
                    tuple(heading_path),
                    tuple(heading_ast_locators),
                    piece,
                )
        flush()

        if not drafts:
            raise ValueError("canonical AST requires at least one non-heading content block")
        return [
            self._materialize(
                canonical_ast,
                ordinal,
                path,
                path_locators,
                pieces,
            )
            for ordinal, (path, path_locators, pieces) in enumerate(drafts)
        ]

    def _append_regular_piece(
        self,
        pending: list[_ParentPiece],
        drafts: list[
            tuple[tuple[str, ...], tuple[str, ...], list[_ParentPiece]]
        ],
        heading_path: tuple[str, ...],
        heading_ast_locators: tuple[str, ...],
        piece: _ParentPiece,
    ) -> None:
        if not pending:
            pending.append(piece)
            return
        if pending[-1].content_type != piece.content_type:
            drafts.append((heading_path, heading_ast_locators, list(pending)))
            pending.clear()
            pending.append(piece)
            return

        candidate_content = _join_piece_content((*pending, piece))
        candidate_tokens = self._count(candidate_content)
        current_tokens = self._count(_join_piece_content(pending))
        if candidate_tokens <= self._target_max:
            pending.append(piece)
            return
        if current_tokens < self._target_min and candidate_tokens <= self._hard_max:
            pending.append(piece)
            drafts.append((heading_path, heading_ast_locators, list(pending)))
            pending.clear()
            return
        drafts.append((heading_path, heading_ast_locators, list(pending)))
        pending.clear()
        pending.append(piece)

    def _pieces_for_block(
        self, block: CanonicalBlock, block_index: int
    ) -> list[_ParentPiece]:
        content_type = _content_type(block)
        token_count = self._count(block.text)
        if content_type == "table":
            rows = block.text.splitlines() or [block.text]
            header_row_to = _table_header_row_to(rows)
            if header_row_to >= len(rows):
                if token_count > self._hard_max:
                    raise ChunkingError(
                        f"header-only table exceeds Parent hard limit for {block.id}"
                    )
                if token_count > CHILD_MAX_TOKENS:
                    raise ChunkingError(
                        f"header-only table exceeds Child limit for {block.id}"
                    )
                return [
                    _piece_from_block(
                        block,
                        block_index,
                        content_type=content_type,
                    )
                ]
        if token_count <= self._hard_max:
            if content_type == "table":
                rows = block.text.splitlines() or [block.text]
                header_row_to = _table_header_row_to(rows)
                if header_row_to < len(rows):
                    char_from, char_to = _raw_row_char_range(
                        block.text,
                        header_row_to,
                        len(rows),
                    )
                    return [
                        _piece_from_block(
                            block,
                            block_index,
                            content_type=content_type,
                            char_from=char_from,
                            char_to=char_to,
                            row_from=header_row_to,
                            row_to=len(rows),
                            table_header_row_to=header_row_to,
                        )
                    ]
            return [_piece_from_block(block, block_index, content_type=content_type)]
        if content_type in {"table", "code", "log"}:
            return self._row_fallback(block, block_index, content_type=content_type)
        return self._semantic_fallback(block, block_index, content_type=content_type)

    def _semantic_fallback(
        self, block: CanonicalBlock, block_index: int, *, content_type: str
    ) -> list[_ParentPiece]:
        doc = DoclingDocument(name=f"parent-block-{block.id}")
        doc.add_text(label=DocItemLabel.TEXT, text=block.text)
        chunker = HybridChunker(
            tokenizer=_LimitedTokenizer(
                delegate=self._tokenizer,
                max_tokens=self._target_max,
            ),
            merge_peers=False,
        )
        chunks = list(chunker.chunk(doc))
        if not chunks:
            raise ChunkingError(f"Docling emitted no Parent content for {block.id}")
        pieces: list[_ParentPiece] = []
        source_cursor = 0
        for chunk in chunks:
            if self._count(chunk.text) > self._hard_max:
                raise ChunkingError(
                    f"Docling exceeded the Parent hard limit for {block.id}"
                )
            char_from = block.text.find(chunk.text, source_cursor)
            if char_from < 0:
                raise ChunkingError(
                    f"Docling Parent segment cannot be mapped to {block.id}"
                )
            char_to = char_from + len(chunk.text)
            source_cursor = char_to
            pieces.append(
                _piece_from_block(
                    block,
                    block_index,
                    content_type=content_type,
                    content=chunk.text,
                    char_from=char_from,
                    char_to=char_to,
                )
            )
        return pieces

    def _row_fallback(
        self, block: CanonicalBlock, block_index: int, *, content_type: str
    ) -> list[_ParentPiece]:
        rows = block.text.splitlines() or [block.text]
        if content_type == "table":
            return self._table_row_fallback(block, block_index, rows)
        output: list[_ParentPiece] = []
        current: list[str] = []
        current_from = 0

        def emit(row_to: int) -> None:
            nonlocal current
            if not current:
                return
            content = "\n".join(current)
            char_from, char_to = _raw_row_char_range(
                block.text,
                current_from,
                row_to,
            )
            output.append(
                _piece_from_block(
                    block,
                    block_index,
                    content_type=content_type,
                    content=content,
                    char_from=char_from,
                    char_to=char_to,
                    row_from=current_from,
                    row_to=row_to,
                    oversize_reason=(
                        "indivisible_row_exceeds_parent_hard_limit"
                        if self._count(content) > self._hard_max
                        else None
                    ),
                )
            )
            current = []

        for row_index, row in enumerate(rows):
            if not current:
                current = [row]
                current_from = row_index
                if self._count(row) > self._hard_max:
                    emit(row_index + 1)
                continue
            candidate = "\n".join((*current, row))
            if self._count(candidate) <= self._hard_max:
                current.append(row)
                continue
            emit(row_index)
            current = [row]
            current_from = row_index
            if self._count(row) > self._hard_max:
                emit(row_index + 1)
        emit(len(rows))
        return output

    def _table_row_fallback(
        self,
        block: CanonicalBlock,
        block_index: int,
        rows: Sequence[str],
    ) -> list[_ParentPiece]:
        header_row_to = _table_header_row_to(rows)
        if header_row_to >= len(rows):
            raise ChunkingError(
                f"header-only table cannot use row fallback for {block.id}"
            )
        header = "\n".join(rows[:header_row_to])
        output: list[_ParentPiece] = []
        current: list[str] = []
        current_from = header_row_to

        def emit(row_to: int) -> None:
            nonlocal current
            if not current:
                return
            content = "\n".join((header, *current))
            char_from, char_to = _raw_row_char_range(
                block.text,
                current_from,
                row_to,
            )
            output.append(
                _piece_from_block(
                    block,
                    block_index,
                    content_type="table",
                    content=content,
                    char_from=char_from,
                    char_to=char_to,
                    row_from=current_from,
                    row_to=row_to,
                    table_header_row_to=header_row_to,
                    oversize_reason=(
                        "indivisible_row_exceeds_parent_hard_limit"
                        if self._count(content) > self._hard_max
                        else None
                    ),
                )
            )
            current = []

        for row_index in range(header_row_to, len(rows)):
            row = rows[row_index]
            if not current:
                current = [row]
                current_from = row_index
                if self._count("\n".join((header, row))) > self._hard_max:
                    emit(row_index + 1)
                continue
            candidate = "\n".join((header, *current, row))
            if self._count(candidate) <= self._hard_max:
                current.append(row)
                continue
            emit(row_index)
            current = [row]
            current_from = row_index
            if self._count("\n".join((header, row))) > self._hard_max:
                emit(row_index + 1)
        emit(len(rows))
        return output

    def _materialize(
        self,
        canonical_ast: CanonicalAst,
        ordinal: int,
        heading_path: tuple[str, ...],
        heading_ast_locators: tuple[str, ...],
        pieces: Sequence[_ParentPiece],
    ) -> ParentChunk:
        content = _join_piece_content(pieces)
        digest = _sha256(content)
        locator = _locator_for_pieces(pieces, segment_ordinal=ordinal)
        envelope = canonical_ast.envelope
        chunk_id = (
            "par_"
            f"{content_id(envelope.document_version_id, str(ordinal), locator.model_dump_json(), digest)[:32]}"
        )
        content_types = {piece.content_type for piece in pieces}
        return ParentChunk(
            id=chunk_id,
            document_id=envelope.document_id,
            document_version_id=envelope.document_version_id,
            user_id=envelope.user_id,
            ordinal=ordinal,
            heading_path=heading_path,
            heading_ast_locators=heading_ast_locators,
            content_type=(content_types.pop() if len(content_types) == 1 else "mixed"),
            content=content,
            token_count=self._count(content),
            page_from=min(piece.page_from for piece in pieces),
            page_to=max(piece.page_to for piece in pieces),
            ast_locator=locator,
            content_hash=digest,
            row_from=(
                pieces[0].row_from + 1
                if len(pieces) == 1 and pieces[0].row_from is not None
                else None
            ),
            row_to=pieces[0].row_to if len(pieces) == 1 else None,
            oversize_reason=(pieces[0].oversize_reason if len(pieces) == 1 else None),
        )

    def _count(self, text: str) -> int:
        return int(self._tokenizer.count_tokens(text))


class ChildBuilder:
    """Build searchable children through Docling's production HybridChunker."""

    def __init__(self, tokenizer: BaseTokenizer) -> None:
        configured_max = int(tokenizer.get_max_tokens())
        if configured_max <= 0:
            raise ValueError("tokenizer maximum must be positive")
        self._tokenizer = _LimitedTokenizer(
            delegate=tokenizer,
            max_tokens=min(CHILD_MAX_TOKENS, configured_max),
        )

    def build(self, parent: ParentChunk) -> list[ChildChunk]:
        heading_context = "\n".join(parent.heading_path)
        if heading_context and self._tokenizer.count_tokens(heading_context) >= (
            self._tokenizer.get_max_tokens()
        ):
            raise ChunkingError(
                f"heading context exhausts the Child token limit for {parent.id}"
            )
        doc = _parent_docling_document(parent)
        chunker = HybridChunker(
            tokenizer=self._tokenizer,
            delim="\n\n",
            merge_peers=True,
            repeat_table_header=True,
            serializer_provider=_StructuredSerializerProvider(),
        )
        docling_chunks = list(chunker.chunk(doc))
        if parent.content_type == "list_item":
            # Semchunk may detach Docling's synthetic Markdown marker from a long
            # ListItem. It has no Canonical source range; structure remains in the
            # server-owned content_type while every source token stays in a Child.
            docling_chunks = [
                chunk
                for chunk in docling_chunks
                if chunk.text.strip() not in {"-", "*", "+"}
            ]
        elif parent.content_type == "code":
            # Markdown code fences are serializer syntax, not Canonical source.
            docling_chunks = [
                chunk for chunk in docling_chunks if chunk.text.strip() != "```"
            ]
        if (
            parent.content_type == "table"
            and parent.ast_locator.spans[0].row_from is None
            and len(docling_chunks) > 1
        ):
            raise ChunkingError(
                f"header-only table cannot be mapped across multiple Children for {parent.id}"
            )
        if not docling_chunks:
            raise ChunkingError(f"Docling emitted no Child chunks for {parent.id}")

        children: list[ChildChunk] = []
        source_cursor = 0
        table_row_cursor = 0
        list_char_cursor = 0
        for ordinal, chunk in enumerate(docling_chunks):
            contextualized = chunker.contextualize(chunk=chunk)
            token_count = self._tokenizer.count_tokens(contextualized)
            if token_count > CHILD_MAX_TOKENS:
                raise ChunkingError(
                    f"Docling emitted {token_count} tokens for Child of {parent.id}"
                )
            emitted_headings = tuple(getattr(chunk.meta, "headings", None) or ())
            if parent.heading_path and emitted_headings != parent.heading_path:
                raise ChunkingError(
                    f"Docling dropped required heading context for {parent.id}"
                )
            if parent.content_type == "table":
                locator, table_row_cursor = _table_child_locator(
                    parent,
                    chunk.text,
                    ordinal=ordinal,
                    row_cursor=table_row_cursor,
                )
            elif parent.content_type == "list_item":
                locator, list_char_cursor = _list_child_locator(
                    parent,
                    chunk.text,
                    ordinal=ordinal,
                    char_cursor=list_char_cursor,
                )
            else:
                char_from = parent.content.find(chunk.text, source_cursor)
                if char_from < 0:
                    locator, source_cursor = _formatted_child_locator(
                        parent,
                        chunk.text,
                        ordinal=ordinal,
                        char_cursor=source_cursor,
                    )
                else:
                    char_to = char_from + len(chunk.text)
                    locator = _clip_locator(
                        parent.ast_locator,
                        char_from,
                        char_to,
                        segment_ordinal=ordinal,
                        parent_content=parent.content,
                    )
                    source_cursor = char_to
            digest = _sha256(contextualized)
            child_id = (
                "chi_"
                f"{content_id(parent.document_version_id, parent.id, str(ordinal), locator.model_dump_json(), digest)[:32]}"
            )
            children.append(
                ChildChunk(
                    id=child_id,
                    parent_id=parent.id,
                    parent_ordinal=parent.ordinal,
                    document_id=parent.document_id,
                    document_version_id=parent.document_version_id,
                    user_id=parent.user_id,
                    ordinal=ordinal,
                    heading_path=parent.heading_path,
                    heading_ast_locators=parent.heading_ast_locators,
                    content_type=parent.content_type,
                    content=chunk.text,
                    contextualized_content=contextualized,
                    token_count=token_count,
                    page_from=min(span.page_from for span in locator.spans),
                    page_to=max(span.page_to for span in locator.spans),
                    ast_locator=locator,
                    content_hash=digest,
                )
            )
        return children


class ChunkingPipeline:
    """Compose deterministic Parent and Docling Child construction."""

    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self._parents = ParentBuilder(tokenizer)
        self._children = ChildBuilder(tokenizer)

    def build(self, canonical_ast: CanonicalAst) -> list[ParentChunk]:
        return [
            parent.model_copy(update={"children": tuple(self._children.build(parent))})
            for parent in self._parents.build(canonical_ast)
        ]


def resolve_ast_locator(canonical_ast: CanonicalAst, locator: AstLocator) -> str:
    """Resolve a persisted locator against the Canonical AST, failing closed."""

    resolved: list[str] = []
    for span in locator.spans:
        match = re.fullmatch(r"#/text_blocks/(0|[1-9][0-9]*)", span.canonical_path)
        if match is None:
            raise ChunkingError(f"invalid Canonical AST path {span.canonical_path!r}")
        block_index = int(match.group(1))
        try:
            block = canonical_ast.text_blocks[block_index]
        except IndexError as error:
            raise ChunkingError(
                f"Canonical AST path is out of range: {span.canonical_path}"
            ) from error
        if block.id != span.block_id:
            raise ChunkingError(
                f"Canonical AST block identity changed at {span.canonical_path}"
            )
        if span.char_to > len(block.text):
            raise ChunkingError(
                f"Canonical AST character range exceeds {span.canonical_path}"
            )
        if (
            span.row_from is None
            or span.row_to is None
            or span.table_header_row_to == 0
        ):
            text = block.text[span.char_from : span.char_to]
        else:
            rows = block.text.splitlines() or [block.text]
            if span.row_to > len(rows):
                raise ChunkingError(
                    f"Canonical AST row range exceeds {span.canonical_path}"
                )
            selected = list(rows[span.row_from : span.row_to])
            if span.table_header_row_to:
                selected = list(rows[: span.table_header_row_to]) + selected
            text = "\n".join(selected)
        resolved.append(f"{span.separator_before}{text}")
    return "".join(resolved)


def _parent_docling_document(parent: ParentChunk) -> DoclingDocument:
    doc = DoclingDocument(name=parent.id)
    for level, heading in enumerate(parent.heading_path, start=1):
        doc.add_heading(text=heading, level=level)
    if parent.content_type == "table":
        doc.add_table(data=_table_data(parent.content))
    elif parent.content_type == "list_item":
        list_group = doc.add_list_group()
        for span in parent.ast_locator.spans:
            item = parent.content[span.parent_char_from : span.parent_char_to]
            doc.add_list_item(text=item, parent=list_group)
    elif parent.content_type == "code":
        doc.add_code(text=parent.content)
    elif parent.content_type == "formula":
        doc.add_formula(text=parent.content)
    else:
        for span in parent.ast_locator.spans:
            item = parent.content[span.parent_char_from : span.parent_char_to]
            doc.add_text(label=DocItemLabel.TEXT, text=item)
    return doc


def _table_data(content: str) -> TableData:
    rows = _table_cells(content)
    if not rows:
        raise ChunkingError("table Parent has no rows")
    column_count = max(len(row) for row in rows)
    cells: list[TableCell] = []
    for row_index, row in enumerate(rows):
        padded = [*row, *([""] * (column_count - len(row)))]
        for column_index, text in enumerate(padded):
            cells.append(
                TableCell(
                    text=text,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + 1,
                    start_col_offset_idx=column_index,
                    end_col_offset_idx=column_index + 1,
                    column_header=row_index == 0,
                )
            )
    return TableData(
        table_cells=cells,
        num_rows=len(rows),
        num_cols=column_count,
    )


def _table_cells(content: str) -> list[list[str]]:
    return [list(row.cells) for row in _table_layout(content)]


def _table_layout(content: str) -> list[_ParsedTableRow]:
    parsed: list[_ParsedTableRow] = []
    lines = content.splitlines(keepends=True) or [content]
    cursor = 0
    for raw_index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped or _is_markdown_separator(stripped):
            cursor += len(raw_line)
            continue
        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        elif "\t" in stripped:
            cells = [cell.strip() for cell in stripped.split("\t")]
        else:
            cells = [stripped]
        leading = len(line) - len(line.lstrip())
        trailing = len(line.rstrip())
        parsed.append(
            _ParsedTableRow(
                cells=tuple(cells),
                raw_index=raw_index,
                char_from=cursor + leading,
                char_to=cursor + trailing,
            )
        )
        cursor += len(raw_line)
    return parsed


def _is_markdown_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _table_header_row_to(rows: Sequence[str]) -> int:
    if not rows:
        return 0
    if len(rows) > 1 and _is_markdown_separator(rows[1].strip()):
        return 2
    return 1


def _clip_locator(
    parent: AstLocator,
    char_from: int,
    char_to: int,
    *,
    segment_ordinal: int,
    parent_content: str,
) -> AstLocator:
    if not 0 <= char_from < char_to <= parent.parent_char_to:
        raise ChunkingError("Child character range is outside its Parent")
    spans: list[AstSpan] = []
    for source in parent.spans:
        overlap_from = max(char_from, source.parent_char_from)
        overlap_to = min(char_to, source.parent_char_to)
        if overlap_from >= overlap_to:
            continue
        separator = ""
        separator_start = source.parent_char_from - len(source.separator_before)
        if spans and char_from <= separator_start:
            separator = source.separator_before
        canonical_from = source.char_from + overlap_from - source.parent_char_from
        canonical_to = source.char_from + overlap_to - source.parent_char_from
        row_from = source.row_from
        row_to = source.row_to
        if (
            row_from is not None
            and row_to is not None
            and source.table_header_row_to == 0
        ):
            row_from, row_to = _intersecting_row_range(
                parent_content[
                    source.parent_char_from : source.parent_char_to
                ],
                overlap_from - source.parent_char_from,
                overlap_to - source.parent_char_from,
                base_row=row_from,
            )
        spans.append(
            source.model_copy(
                update={
                    "char_from": canonical_from,
                    "char_to": canonical_to,
                    "row_from": row_from,
                    "row_to": row_to,
                    "parent_char_from": overlap_from,
                    "parent_char_to": overlap_to,
                    "separator_before": separator,
                }
            )
        )
    if not spans:
        raise ChunkingError("Child range has no Canonical AST provenance")
    return AstLocator(
        spans=tuple(spans),
        segment_ordinal=segment_ordinal,
        parent_char_from=char_from,
        parent_char_to=char_to,
    )


def _table_child_locator(
    parent: ParentChunk,
    child_content: str,
    *,
    ordinal: int,
    row_cursor: int,
) -> tuple[AstLocator, int]:
    if len(parent.ast_locator.spans) != 1:
        raise ChunkingError("table Parent must have exactly one Canonical AST span")
    source = parent.ast_locator.spans[0]
    if source.row_from is None or source.row_to is None:
        locator = AstLocator(
            spans=(source.model_copy(update={"separator_before": ""}),),
            segment_ordinal=ordinal,
            parent_char_from=source.parent_char_from,
            parent_char_to=source.parent_char_to,
        )
        return locator, row_cursor
    parent_layout = _table_layout(parent.content)
    child_layout = _table_layout(child_content)
    if len(parent_layout) < 2 or len(child_layout) < 2:
        raise ChunkingError("table Child has no traceable body rows")
    parent_body = [row.cells for row in parent_layout[1:]]
    child_body = [row.cells for row in child_layout[1:]]
    local_start = _find_row_sequence(parent_body, child_body, row_cursor)
    local_to = local_start + len(child_body)
    canonical_from = source.row_from + local_start
    canonical_to = source.row_from + local_to
    parent_char_from = parent_layout[local_start + 1].char_from
    parent_char_to = parent_layout[local_to].char_to
    parent_body_char_from = parent_layout[1].char_from
    canonical_char_from = source.char_from + (
        parent_char_from - parent_body_char_from
    )
    canonical_char_to = source.char_from + (parent_char_to - parent_body_char_from)
    span = source.model_copy(
        update={
            "char_from": canonical_char_from,
            "char_to": canonical_char_to,
            "row_from": canonical_from,
            "row_to": canonical_to,
            "parent_char_from": parent_char_from,
            "parent_char_to": parent_char_to,
            "separator_before": "",
        }
    )
    locator = AstLocator(
        spans=(span,),
        segment_ordinal=ordinal,
        parent_char_from=span.parent_char_from,
        parent_char_to=span.parent_char_to,
    )
    return locator, local_to


def _find_row_sequence(
    rows: Sequence[Sequence[str]],
    wanted: Sequence[Sequence[str]],
    start: int,
) -> int:
    normalized = [tuple(cell.strip() for cell in row) for row in rows]
    target = [tuple(cell.strip() for cell in row) for row in wanted]
    for index in range(start, len(normalized) - len(target) + 1):
        if normalized[index : index + len(target)] == target:
            return index
    raise ChunkingError("Docling table Child rows cannot be mapped to its Parent")


def _raw_row_char_range(content: str, row_from: int, row_to: int) -> tuple[int, int]:
    lines = content.splitlines(keepends=True) or [content]
    if not 0 <= row_from < row_to <= len(lines):
        raise ChunkingError("row character range is outside its source")
    char_from = sum(len(line) for line in lines[:row_from])
    last_start = sum(len(line) for line in lines[: row_to - 1])
    char_to = last_start + len(lines[row_to - 1].rstrip("\r\n"))
    return char_from, char_to


def _intersecting_row_range(
    content: str,
    char_from: int,
    char_to: int,
    *,
    base_row: int,
) -> tuple[int, int]:
    """Return rows with at least one non-newline source character in the Child.

    The returned range is half-open. A Child beginning or ending inside a row
    includes that row; blank physical rows between two intersected rows remain
    structurally covered by the outer half-open range.
    """

    if not 0 <= char_from < char_to <= len(content):
        raise ChunkingError("Child row intersection is outside its Parent span")
    intersected: list[int] = []
    cursor = 0
    for row_index, line in enumerate(content.splitlines(keepends=True) or [content]):
        source_length = len(line.rstrip("\r\n"))
        line_from = cursor
        line_to = cursor + source_length
        if max(char_from, line_from) < min(char_to, line_to):
            intersected.append(row_index)
        cursor += len(line)
    if not intersected:
        raise ChunkingError("Child source contains no row characters")
    return base_row + min(intersected), base_row + max(intersected) + 1


def _list_child_locator(
    parent: ParentChunk,
    child_content: str,
    *,
    ordinal: int,
    char_cursor: int,
) -> tuple[AstLocator, int]:
    source_content = re.sub(r"(?m)^[*+-]\s+", "", child_content).strip()
    char_from = parent.content.find(source_content, char_cursor)
    if char_from < 0:
        raise ChunkingError("Docling list Child cannot be mapped to its Parent")
    char_to = char_from + len(source_content)
    return (
        _clip_locator(
            parent.ast_locator,
            char_from,
            char_to,
            segment_ordinal=ordinal,
            parent_content=parent.content,
        ),
        char_to,
    )


def _formatted_child_locator(
    parent: ParentChunk,
    child_content: str,
    *,
    ordinal: int,
    char_cursor: int,
) -> tuple[AstLocator, int]:
    unwrapped = child_content
    if parent.content_type == "code":
        unwrapped = re.sub(r"^```[^\n]*\n", "", unwrapped)
        unwrapped = re.sub(r"\n+```$", "", unwrapped)
    char_from = parent.content.find(unwrapped, char_cursor)
    if char_from < 0:
        raise ChunkingError(f"Docling Child cannot be mapped to Parent {parent.id}")
    char_to = char_from + len(unwrapped)
    return (
        _clip_locator(
            parent.ast_locator,
            char_from,
            char_to,
            segment_ordinal=ordinal,
            parent_content=parent.content,
        ),
        char_to,
    )


def _update_heading_path(
    path: list[str],
    locators: list[str],
    block: CanonicalBlock,
    block_index: int,
) -> None:
    level = min(block.heading_level or 1, len(path) + 1)
    del path[max(0, level - 1) :]
    del locators[max(0, level - 1) :]
    path.append(block.text.strip())
    locators.append(f"#/text_blocks/{block_index}")


def _content_type(block: CanonicalBlock) -> str:
    if block.kind == "other" and _looks_like_log(block.text):
        return "log"
    return block.kind


def _looks_like_log(text: str) -> bool:
    first_line = text.lstrip().partition("\n")[0]
    return first_line.startswith(("TRACE ", "DEBUG ", "INFO ", "WARN ", "ERROR "))


def _piece_from_block(
    block: CanonicalBlock,
    block_index: int,
    *,
    content_type: str,
    content: str | None = None,
    char_from: int | None = None,
    char_to: int | None = None,
    row_from: int | None = None,
    row_to: int | None = None,
    table_header_row_to: int = 0,
    oversize_reason: Literal["indivisible_row_exceeds_parent_hard_limit"] | None = None,
) -> _ParentPiece:
    return _ParentPiece(
        content=block.text if content is None else content,
        content_type=content_type,
        page_from=block.provenance.page_from,
        page_to=block.provenance.page_to,
        canonical_path=f"#/text_blocks/{block_index}",
        block_id=block.id,
        char_from=0 if char_from is None else char_from,
        char_to=len(block.text) if char_to is None else char_to,
        row_from=row_from,
        row_to=row_to,
        table_header_row_to=table_header_row_to,
        oversize_reason=oversize_reason,
    )


def _join_piece_content(pieces: Sequence[_ParentPiece]) -> str:
    return "\n\n".join(piece.content for piece in pieces)


def _locator_for_pieces(
    pieces: Sequence[_ParentPiece], *, segment_ordinal: int
) -> AstLocator:
    spans: list[AstSpan] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        separator = "" if index == 0 else "\n\n"
        parent_from = cursor + len(separator)
        parent_to = parent_from + len(piece.content)
        spans.append(
            AstSpan(
                canonical_path=piece.canonical_path,
                block_id=piece.block_id,
                page_from=piece.page_from,
                page_to=piece.page_to,
                char_from=piece.char_from,
                char_to=piece.char_to,
                row_from=piece.row_from,
                row_to=piece.row_to,
                table_header_row_to=piece.table_header_row_to,
                parent_char_from=parent_from,
                parent_char_to=parent_to,
                separator_before=separator,
            )
        )
        cursor = parent_to
    return AstLocator(
        spans=tuple(spans),
        segment_ordinal=segment_ordinal,
        parent_char_from=0,
        parent_char_to=cursor,
    )


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
