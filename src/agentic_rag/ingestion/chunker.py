"""Deterministic structure-semantic Parent/Child chunking.

Parent chunks retain canonical-AST provenance and favor complete sections.  Child
chunks are always produced by Docling's ``HybridChunker``; the small tokenizer
adapter below only narrows the configured tokenizer's limit and never provides a
second tokenization implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Sequence

from docling.chunking import HybridChunker  # type: ignore[import-not-found]
from docling_core.transforms.chunker.tokenizer.base import (  # type: ignore[import-not-found]
    BaseTokenizer,
)
from docling_core.types.doc import (  # type: ignore[import-not-found]
    DocItemLabel,
    DoclingDocument,
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
    ast_locator: str
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
    ast_locator: str
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
    block_from: int
    block_to: int
    page_from: int
    page_to: int
    ast_locator: str
    row_from: int | None = None
    row_to: int | None = None
    oversize_reason: Literal["indivisible_row_exceeds_parent_hard_limit"] | None = None


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
        if token_count <= self._hard_max:
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
        for segment_index, chunk in enumerate(chunks):
            if self._count(chunk.text) > self._hard_max:
                raise ChunkingError(
                    f"Docling exceeded the Parent hard limit for {block.id}"
                )
            pieces.append(
                _piece_from_block(
                    block,
                    block_index,
                    content_type=content_type,
                    content=chunk.text,
                    ast_locator=f"#/text_blocks/{block_index}/segments/{segment_index}",
                )
            )
        return pieces

    def _row_fallback(
        self, block: CanonicalBlock, block_index: int, *, content_type: str
    ) -> list[_ParentPiece]:
        rows = block.text.splitlines() or [block.text]
        output: list[_ParentPiece] = []
        current: list[str] = []
        current_from = 1

        def emit(row_to: int) -> None:
            nonlocal current
            if not current:
                return
            content = "\n".join(current)
            output.append(
                _piece_from_block(
                    block,
                    block_index,
                    content_type=content_type,
                    content=content,
                    ast_locator=(
                        f"#/text_blocks/{block_index}/rows/{current_from}:{row_to}"
                    ),
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

        for row_no, row in enumerate(rows, start=1):
            if not current:
                current = [row]
                current_from = row_no
                if self._count(row) > self._hard_max:
                    emit(row_no)
                continue
            candidate = "\n".join((*current, row))
            if self._count(candidate) <= self._hard_max:
                current.append(row)
                continue
            emit(row_no - 1)
            current = [row]
            current_from = row_no
            if self._count(row) > self._hard_max:
                emit(row_no)
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
        locator = _combined_locator(pieces)
        envelope = canonical_ast.envelope
        chunk_id = f"par_{content_id(envelope.document_version_id, str(ordinal), locator, digest)[:32]}"
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
            row_from=pieces[0].row_from if len(pieces) == 1 else None,
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
        doc = DoclingDocument(name=parent.id)
        for level, heading in enumerate(parent.heading_path, start=1):
            doc.add_heading(text=heading, level=level)
        doc.add_text(label=DocItemLabel.TEXT, text=parent.content)
        chunker = HybridChunker(tokenizer=self._tokenizer, merge_peers=True)
        docling_chunks = list(chunker.chunk(doc))
        if not docling_chunks:
            raise ChunkingError(f"Docling emitted no Child chunks for {parent.id}")

        children: list[ChildChunk] = []
        for ordinal, chunk in enumerate(docling_chunks):
            contextualized = chunker.contextualize(chunk=chunk)
            token_count = self._tokenizer.count_tokens(contextualized)
            if token_count > CHILD_MAX_TOKENS:
                raise ChunkingError(
                    f"Docling emitted {token_count} tokens for Child of {parent.id}"
                )
            digest = _sha256(contextualized)
            child_id = f"chi_{content_id(parent.document_version_id, parent.id, str(ordinal), digest)[:32]}"
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
                    page_from=parent.page_from,
                    page_to=parent.page_to,
                    ast_locator=parent.ast_locator,
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
    ast_locator: str | None = None,
    row_from: int | None = None,
    row_to: int | None = None,
    oversize_reason: Literal["indivisible_row_exceeds_parent_hard_limit"] | None = None,
) -> _ParentPiece:
    return _ParentPiece(
        content=block.text if content is None else content,
        content_type=content_type,
        block_from=block_index,
        block_to=block_index,
        page_from=block.provenance.page_from,
        page_to=block.provenance.page_to,
        ast_locator=ast_locator or f"#/text_blocks/{block_index}",
        row_from=row_from,
        row_to=row_to,
        oversize_reason=oversize_reason,
    )


def _join_piece_content(pieces: Sequence[_ParentPiece]) -> str:
    return "\n\n".join(piece.content for piece in pieces)


def _combined_locator(pieces: Sequence[_ParentPiece]) -> str:
    if len(pieces) == 1:
        return pieces[0].ast_locator
    first = min(piece.block_from for piece in pieces)
    last = max(piece.block_to for piece in pieces) + 1
    return f"#/text_blocks/{first}:{last}"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
