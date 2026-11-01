"""Deterministic source catalog; no semantic acceptance is performed here."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from agentic_rag.ingestion.assembler import CanonicalAst, CanonicalBlock
from agentic_rag.ingestion.chunker import ChildChunk
from agentic_rag.models.contextual import CanonicalSourceRef, text_sha256
from agentic_rag.runtime.ids import content_id


class SourceIntegrityError(ValueError):
    """Source identity, artifact bytes, or quoted coordinates are invalid.

    This is a build failure, unlike semantically insufficient claims which the
    verifier records and the renderer omits.
    """


@dataclass(frozen=True)
class SourceCatalog:
    user_id: str
    document_id: str
    document_version_id: str
    ast_sha256: str
    _blocks: Mapping[str, tuple[str, CanonicalBlock]]

    @classmethod
    def from_ast(cls, ast: CanonicalAst, ast_sha256: str) -> SourceCatalog:
        # Match ArtifactStore.put_json byte-for-byte; never trust a caller hash
        # as a declaration that the in-memory AST has been verified.
        encoded = json.dumps(
            ast.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != ast_sha256:
            raise SourceIntegrityError("canonical AST artifact hash mismatch")
        if len({block.id for block in ast.text_blocks}) != len(ast.text_blocks):
            raise SourceIntegrityError("duplicate canonical block IDs")
        envelope = ast.envelope
        blocks = {}
        for ordinal, block in enumerate(ast.text_blocks):
            path = f"#/text_blocks/{ordinal}"
            source_id = "src_" + content_id(
                envelope.user_id,
                envelope.document_id,
                envelope.document_version_id,
                ast_sha256,
                block.id,
                path,
            )
            blocks[source_id] = (path, block)
        return cls(
            envelope.user_id,
            envelope.document_id,
            envelope.document_version_id,
            ast_sha256,
            MappingProxyType(blocks),
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(self._blocks)

    def source_text(self, source_id: str) -> str:
        """Original block text for bounded source selection and prompt catalogs."""
        return self._lookup(source_id)[1].text

    def _lookup(self, source_id: str) -> tuple[str, CanonicalBlock]:
        try:
            return self._blocks[source_id]
        except KeyError as error:
            raise SourceIntegrityError(
                "source_id is outside the input catalog"
            ) from error

    def resolve_quote(self, source_id: str, quote: str) -> CanonicalSourceRef:
        path, block = self._lookup(source_id)
        if not quote:
            raise SourceIntegrityError("empty quote")
        start = block.text.find(quote)
        if start < 0:
            raise SourceIntegrityError("quote does not occur in source")
        if block.text.find(quote, start + 1) >= 0:
            raise SourceIntegrityError(
                "ambiguous quote; supply a distinguishing adjacent excerpt"
            )
        return CanonicalSourceRef(
            source_id=source_id,
            user_id=self.user_id,
            document_id=self.document_id,
            document_version_id=self.document_version_id,
            canonical_ast_sha256=self.ast_sha256,
            block_id=block.id,
            canonical_path=path,
            char_from=start,
            char_to=start + len(quote),
            quote_sha256=text_sha256(quote),
            page_from=block.provenance.page_from,
            page_to=block.provenance.page_to,
        )

    def verify_source(self, ref: CanonicalSourceRef) -> str:
        path, block = self._lookup(ref.source_id)
        if (
            ref.user_id,
            ref.document_id,
            ref.document_version_id,
            ref.canonical_ast_sha256,
        ) != (
            self.user_id,
            self.document_id,
            self.document_version_id,
            self.ast_sha256,
        ):
            raise SourceIntegrityError(
                "source owner/document/version/AST identity mismatch"
            )
        if ref.canonical_path != path or ref.block_id != block.id:
            raise SourceIntegrityError("source block/path mismatch")
        if (ref.page_from, ref.page_to) != (
            block.provenance.page_from,
            block.provenance.page_to,
        ):
            raise SourceIntegrityError(
                "source page range differs from canonical provenance"
            )
        if not 0 <= ref.char_from < ref.char_to <= len(block.text):
            raise SourceIntegrityError("source coordinates out of bounds")
        quote = block.text[ref.char_from : ref.char_to]
        if text_sha256(quote) != ref.quote_sha256:
            raise SourceIntegrityError("source quote hash mismatch")
        return quote

    def covers_child(
        self, refs: Sequence[CanonicalSourceRef], child: ChildChunk
    ) -> bool:
        for ref in refs:
            self.verify_source(ref)
        return covers_child_spans(refs, child)


def covers_child_spans(refs: Sequence[CanonicalSourceRef], child: ChildChunk) -> bool:
    """Coverage only; the owning catalog must separately verify source bytes."""
    identity = (child.user_id, child.document_id, child.document_version_id)
    if any(
        (ref.user_id, ref.document_id, ref.document_version_id) != identity
        for ref in refs
    ):
        return False
    for span in child.ast_locator.spans:
        intervals = sorted(
            (ref.char_from, ref.char_to)
            for ref in refs
            if ref.block_id == span.block_id
            and ref.canonical_path == span.canonical_path
        )
        cursor = span.char_from
        for start, end in intervals:
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor < span.char_to:
            return False
    return True
