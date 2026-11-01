"""Bounded whole-block selection; budget exhaustion never truncates conditions."""

from __future__ import annotations
from collections.abc import Callable
from pydantic import BaseModel, ConfigDict
from agentic_rag.ingestion.chunker import ChildChunk, ParentChunk
from agentic_rag.ingestion.contextualization.map import DocumentContextMap, SourceBlock
from agentic_rag.models.contextual import ContextScopeRegion


class ContextInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    target_source_child_id: str
    target_content: str
    sources: tuple[SourceBlock, ...]
    scopes: dict[str, tuple[str, ...]]
    context_budget_exhausted: bool = False


class ContextAssembler:
    def __init__(self, count_tokens: Callable[[str], int]):
        self._count = count_tokens

    def assemble(
        self,
        child: ChildChunk,
        parent: ParentChunk,
        mapping: DocumentContextMap,
        *,
        max_tokens: int = 6000,
        max_blocks: int = 16,
        max_hops: int = 2,
    ) -> tuple[ContextInput, dict[str, ContextScopeRegion]]:
        by_block = {
            r.block_id: r.source_id for r in mapping.regions["document"].source_refs
        }
        body = [by_block[s.block_id] for s in child.ast_locator.spans]
        parent_ids = [by_block[s.block_id] for s in parent.ast_locator.spans]
        regions = {
            k: v
            for k, v in mapping.regions.items()
            if all(s in {r.source_id for r in v.source_refs} for s in body)
        }
        headings = [
            v.source_refs[0].source_id for k, v in regions.items() if k != "document"
        ]
        ids = list(
            dict.fromkeys(body + headings + parent_ids + list(mapping.scope_candidates))
        )
        all_ids = [s.source_id for s in mapping.sources]
        for sid in body:
            pos = all_ids.index(sid)
            ids.extend(all_ids[max(0, pos - 1) : pos + 2])
        frontier = list(dict.fromkeys(ids))
        for _ in range(max_hops):
            next_ids = [
                t for sid in frontier for t in mapping.explicit_references.get(sid, ())
            ]
            ids.extend(next_ids)
            frontier = next_ids
        source_map = {s.source_id: s for s in mapping.sources}
        ordered = list(dict.fromkeys(ids))
        selected = []
        exhausted = False
        # Scope spans remain server-side. Retain the target-body intersection
        # of a structurally resolved region; this proves geometric coverage of
        # this child, not a business claim over every paragraph in the document.
        # All candidate applicability/exception paragraphs are separately selected
        # above. Failure to fit any required block marks the input exhausted.
        for sid in ordered:
            candidate = selected + [source_map[sid]]
            provisional = ContextInput(
                target_source_child_id=child.id,
                target_content=child.content,
                sources=tuple(candidate),
                scopes={},
            )
            if (
                len(candidate) > max_blocks
                or self._count(provisional.model_dump_json()) > max_tokens
            ):
                exhausted = True
                break
            selected = candidate
        selected_ids = {s.source_id for s in selected}
        visible = {
            k: ContextScopeRegion(
                kind=v.kind,
                source_refs=tuple(r for r in v.source_refs if r.source_id in body),
            )
            for k, v in regions.items()
            if set(body) <= selected_ids
        }
        result = ContextInput(
            target_source_child_id=child.id,
            target_content=child.content,
            sources=tuple(selected),
            scopes={
                k: tuple(r.source_id for r in v.source_refs) for k, v in visible.items()
            },
            context_budget_exhausted=exhausted,
        )
        if self._count(result.model_dump_json()) > max_tokens:
            result = result.model_copy(update={"context_budget_exhausted": True})
        return result, visible
