"""Original-source document/section map; never inferred business authority."""

from __future__ import annotations
import re
from pydantic import BaseModel, ConfigDict
from agentic_rag.ingestion.assembler import CanonicalAst
from agentic_rag.ingestion.contextualization.sources import SourceCatalog
from agentic_rag.models.contextual import ContextScopeRegion


class SourceBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: str
    text: str
    kind: str


class DocumentContextMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sources: tuple[SourceBlock, ...]
    regions: dict[str, ContextScopeRegion]
    explicit_references: dict[str, tuple[str, ...]]
    scope_candidates: tuple[str, ...]

    @classmethod
    def build(cls, ast: CanonicalAst, catalog: SourceCatalog) -> DocumentContextMap:
        sources = tuple(
            SourceBlock(source_id=sid, text=b.text, kind=b.kind)
            for sid, b in zip(catalog.source_ids, ast.text_blocks, strict=True)
        )
        refs = tuple(catalog.resolve_quote(s.source_id, s.text) for s in sources)
        regions = {"document": ContextScopeRegion(kind="document", source_refs=refs)}
        titles = {}
        for i, b in enumerate(ast.text_blocks):
            if b.kind == "table":
                regions[f"table:{sources[i].source_id}"] = ContextScopeRegion(
                    kind="table", source_refs=(refs[i],)
                )
            if b.kind != "heading" or b.heading_level is None:
                continue
            end = next(
                (
                    j
                    for j in range(i + 1, len(ast.text_blocks))
                    if ast.text_blocks[j].kind == "heading"
                    and ast.text_blocks[j].heading_level is not None
                    and ast.text_blocks[j].heading_level <= b.heading_level
                ),
                len(sources),
            )
            regions[f"section:{sources[i].source_id}"] = ContextScopeRegion(
                kind="section", source_refs=refs[i:end]
            )
            titles.setdefault(b.text, []).append(i)
            label = re.match(r"第[零〇一二三四五六七八九十百0-9]+章", b.text)
            if label and label.group(0) != b.text:
                titles.setdefault(label.group(0), []).append(i)
        edges = {}
        for source in sources:
            targets = []
            # Only explicit reference syntax and unique original headings; mere
            # neighbouring mentions do not create a reference edge.
            for title, indices in titles.items():
                if len(indices) == 1 and re.search(
                    r"(?:参见|按照|按|依据|见)\s*" + re.escape(title), source.text
                ):
                    region = regions[f"section:{sources[indices[0]].source_id}"]
                    targets.extend(r.source_id for r in region.source_refs)
            edges[source.source_id] = tuple(dict.fromkeys(targets))
        return cls(
            sources=sources,
            regions=regions,
            explicit_references=edges,
            scope_candidates=tuple(
                s.source_id
                for s in sources
                if re.search(
                    r"适用|不含|除外|仅限|定义|指的是|appl(?:y|ies|icable)|exclud",
                    s.text,
                    re.I,
                )
            ),
        )
